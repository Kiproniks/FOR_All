from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError

from apps.books.models import GlobalBookCache, UserBook
from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.hashing import sha256_bytes
from apps.books.services.llm_service import ensure_llm_ready, select_ollama_model
from apps.books.services.thought_chain.reports import write_sentence_preview_reports, write_thought_chain_reports
from apps.books.services.thought_chain.sentence_splitter import split_book_into_sentences
from apps.books.services.thought_chain.thought_extractor import extract_thought_from_sentence
from apps.books.services.thought_chain.thought_chain_runner import (
    run_thought_chain_analysis,
    run_thought_chain_preview,
)


class Command(BaseCommand):
    help = "Run sentence-by-sentence LLM thought-chain analysis."

    def add_arguments(self, parser):
        parser.add_argument("--file", dest="file_path", default="")
        parser.add_argument("--book-id", dest="book_id", type=int, default=None)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--full", action="store_true", help="Process the whole book without --max-sentences limit.")
        parser.add_argument("--strict", action="store_true", help="Enable strict quality gate metadata for production runs.")
        parser.add_argument(
            "--mode",
            choices=("greedy", "strict"),
            default="greedy",
            help="Block generation mode: greedy for production, strict for full pairwise.",
        )
        parser.add_argument(
            "--strict-pairwise-llm",
            action="store_true",
            help="Require LLM pairwise comparisons in persistent mode; deterministic prefilter stays preview-only.",
        )
        parser.add_argument(
            "--merge-same-title-blocks",
            action="store_true",
            help="Merge GlobalLogicalThoughtBlock rows with same/similar title after block creation.",
        )
        parser.add_argument("--no-llm", action="store_true", help="Only parse and split sentences; do not call Ollama or write DB analysis.")
        parser.add_argument("--preview-sentences", type=int, default=None, help="Number of sentences to show in --no-llm preview report.")
        parser.add_argument("--meaning-gate-noise-test", action="store_true", help="Run artificial meaning/noise gate test without parsing a book or writing DB.")
        parser.add_argument("--max-sentences", type=int, default=None)
        parser.add_argument("--max-pairs", type=int, default=None)
        parser.add_argument("--skip-pairwise", action="store_true")
        parser.add_argument("--skip-global-blocks", action="store_true")
        parser.add_argument("--force-refresh", action="store_true")
        parser.add_argument("--model", default="")
        parser.add_argument("--output-dir", default=".")

    def handle(self, *args: Any, **options: Any):
        file_path = str(options.get("file_path") or "").strip()
        book_id = options.get("book_id")
        dry_run = bool(options.get("dry_run"))
        no_llm = bool(options.get("no_llm"))
        preview_sentences = options.get("preview_sentences")
        max_sentences = options.get("max_sentences")
        max_pairs = options.get("max_pairs")
        full = bool(options.get("full"))
        strict = bool(options.get("strict"))
        strict_pairwise_llm = bool(options.get("strict_pairwise_llm"))
        analysis_mode = str(options.get("mode") or "greedy")
        if strict_pairwise_llm:
            analysis_mode = "strict"
        model_name = str(options.get("model") or "").strip()
        output_dir = options.get("output_dir") or "."

        if full:
            max_sentences = None

        if bool(options.get("meaning_gate_noise_test")):
            llm_state = ensure_llm_ready(require_enabled=True)
            if not llm_state.get("ok"):
                raise CommandError(f"LLM is required for meaning gate test: {llm_state.get('error')}")
            if not model_name:
                model_name = select_ollama_model("fast", available_models=llm_state.get("models", [])) or ""
            cases = [
                ("\u0413\u041b\u0410\u0412\u0410 2", True),
                ("1.1.", True),
                ("\u0418\u043b\u043b.", True),
                ("\u00a9 \u041e\u041e\u041e \u00ab\u0418\u0437\u0434\u0430\u0442\u0435\u043b\u044c\u0441\u0442\u0432\u043e \u0410\u0421\u0422\u00bb, 2018", True),
                ("[12]", True),
                ("ISBN 978-5-17-123456-7", True),
                ("\u041d\u0430\u0441\u0442\u043e\u044f\u0449\u0438\u0435 \u0444\u043e\u0442\u043e\u0433\u0440\u0430\u0444\u0438\u0438 \u0438\u0437 \u043a\u043e\u0441\u043c\u043e\u0441\u0430!", False),
                ("\u041d\u0430\u0443\u0447\u043d\u044b\u0439 \u043c\u0435\u0442\u043e\u0434 \u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u044d\u043a\u0441\u043f\u0435\u0440\u0438\u043c\u0435\u043d\u0442\u0430\u043b\u044c\u043d\u044b\u0445 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u0439.", False),
            ]
            rows = []
            for index, (text, expected_noise) in enumerate(cases, start=1):
                payload = extract_thought_from_sentence(text, model_name=model_name)
                actual_noise = bool(payload.get("noise")) or not bool(payload.get("is_meaningful"))
                rows.append(
                    {
                        "index": index,
                        "text": text,
                        "expected_noise": expected_noise,
                        "actual_noise": actual_noise,
                        "passed": expected_noise == actual_noise,
                        "is_meaningful": bool(payload.get("is_meaningful")),
                        "noise": bool(payload.get("noise")),
                        "skip_reason": payload.get("skip_reason", ""),
                        "thought": payload.get("thought", ""),
                        "terms": payload.get("terms", []),
                        "json_valid": bool(payload.get("json_valid")),
                        "fallback_used": bool(payload.get("fallback_used")),
                    }
                )
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / "meaning_gate_noise_test.json"
            md_path = out / "meaning_gate_noise_test.md"
            payload = {
                "model": model_name,
                "db_written": False,
                "total_cases": len(rows),
                "passed_cases": sum(1 for row in rows if row["passed"]),
                "failed_cases": sum(1 for row in rows if not row["passed"]),
                "rows": rows,
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            lines = [
                "# Meaning Gate Noise Test",
                "",
                f"- model: {model_name}",
                "- db_written: false",
                f"- total_cases: {payload['total_cases']}",
                f"- passed_cases: {payload['passed_cases']}",
                f"- failed_cases: {payload['failed_cases']}",
                "",
                "| # | Expected noise | Actual noise | Passed | Text | Thought / Reason |",
                "|---:|---|---|---|---|---|",
            ]
            for row in rows:
                reason = str(row["skip_reason"] or row["thought"])
                text_value = str(row["text"]).replace("|", "\\|")
                reason_value = reason.replace("|", "\\|")
                lines.append(
                    f"| {row['index']} | {row['expected_noise']} | {row['actual_noise']} | {row['passed']} | "
                    f"{text_value} | {reason_value} |"
                )
            md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.stdout.write(self.style.SUCCESS("meaning gate noise test completed"))
            self.stdout.write("db_written=False")
            self.stdout.write(f"report_json={json_path}")
            self.stdout.write(f"report_md={md_path}")
            self.stdout.write(f"passed={payload['passed_cases']}/{payload['total_cases']}")
            return

        parsed = None
        content = b""
        filename = ""
        user_book = None
        global_book = None

        if book_id:
            user_book = UserBook.objects.select_related("global_cache").get(id=book_id)
            if user_book.file:
                user_book.file.open("rb")
                content = user_book.file.read()
                user_book.file.close()
            elif not dry_run and not no_llm:
                raise CommandError("Book has no source file attached; provide --file or use dry-run with file.")
            filename = user_book.original_filename
            if content:
                parsed = parse_uploaded_book(content, filename)
            global_book = user_book.global_cache
        elif file_path:
            path = Path(file_path)
            if not path.exists():
                raise CommandError(f"File not found: {path}")
            content = path.read_bytes()
            filename = path.name
            parsed = parse_uploaded_book(content, filename)
        else:
            raise CommandError("Provide --file or --book-id.")

        if parsed is None:
            raise CommandError("Unable to parse source book.")

        if no_llm:
            preview_count = int(preview_sentences or max_sentences or 200)
            if preview_count <= 0:
                raise CommandError("--preview-sentences must be positive.")
            sentences = split_book_into_sentences(parsed)
            paths = write_sentence_preview_reports(
                file_name=filename,
                sentences=sentences,
                preview_count=preview_count,
                output_dir=output_dir,
            )
            self.stdout.write(self.style.SUCCESS("sentence preview completed"))
            self.stdout.write("llm_used=False")
            self.stdout.write("db_written=False")
            self.stdout.write(f"report_json={paths['json']}")
            self.stdout.write(f"report_md={paths['md']}")
            self.stdout.write(f"total_sentences={len(sentences)} preview_count={min(preview_count, len(sentences))}")
            return

        llm_state = ensure_llm_ready(require_enabled=True)
        if not llm_state.get("ok"):
            raise CommandError(f"LLM is required for llm_thought_chain: {llm_state.get('error')}")
        if not model_name:
            model_name = select_ollama_model("fast", available_models=llm_state.get("models", [])) or ""

        if dry_run:
            def progress(message: str) -> None:
                self.stdout.write(message)
                try:
                    self.stdout.flush()
                except Exception:
                    pass

            report = run_thought_chain_preview(
                parsed,
                max_sentences=max_sentences or 30,
                max_pairs=max_pairs,
                model_name=model_name,
                skip_pairwise=bool(options.get("skip_pairwise")),
                progress_callback=progress,
            )
            paths = write_thought_chain_reports(report, output_dir=output_dir)
            self.stdout.write(self.style.SUCCESS("llm_thought_chain dry-run completed"))
            self.stdout.write(f"report_json={paths['json']}")
            self.stdout.write(f"report_md={paths['md']}")
            self.stdout.write(
                f"sentences={report.get('total_sentences')} thoughts={report.get('thoughts_created')} "
                f"groups={report.get('sequential_groups_created')} pairs={report.get('pairwise_comparisons_done')}"
            )
            return

        if user_book is None:
            User = get_user_model()
            user, _ = User.objects.get_or_create(
                email="thought_chain_runner@local.local",
                defaults={"is_active": True},
            )
            if not user.has_usable_password():
                user.set_unusable_password()
                user.save(update_fields=["password"])
            file_hash = sha256_bytes(content)
            global_book, _ = GlobalBookCache.objects.get_or_create(
                file_hash=file_hash,
                defaults={"title": parsed.title, "authors": parsed.authors, "metadata": {}, "analysis_version": "thought_chain_v1"},
            )
            user_book = UserBook.objects.create(
                user=user,
                global_cache=global_book,
                title=parsed.title,
                authors=parsed.authors,
                original_filename=filename,
                file_hash=file_hash,
                status=UserBook.Status.PROCESSING,
                current_stage="thought_processing",
                progress_percent=1,
                analysis_mode="llm_thought_chain",
            )
            user_book.file.save(filename, ContentFile(content), save=True)
        elif global_book is None:
            file_hash = user_book.file_hash or sha256_bytes(content)
            global_book, _ = GlobalBookCache.objects.get_or_create(
                file_hash=file_hash,
                defaults={"title": parsed.title, "authors": parsed.authors, "metadata": {}, "analysis_version": "thought_chain_v1"},
            )
            user_book.global_cache = global_book
            user_book.save(update_fields=["global_cache"])

        def progress(message: str) -> None:
            self.stdout.write(message)
            try:
                self.stdout.flush()
            except Exception:
                pass

        report = run_thought_chain_analysis(
            book=user_book,
            parsed_book=parsed,
            global_book=global_book,
            model_name=model_name,
            max_sentences=max_sentences,
            max_pairs=max_pairs,
            skip_pairwise=bool(options.get("skip_pairwise")),
            skip_global_blocks=bool(options.get("skip_global_blocks")),
            force_refresh=bool(options.get("force_refresh")),
            resume=bool(options.get("resume")),
            strict=strict,
            strict_pairwise_llm=strict_pairwise_llm,
            analysis_mode=analysis_mode,
            merge_same_title_blocks=bool(options.get("merge_same_title_blocks")),
            progress_callback=progress,
        )
        paths = write_thought_chain_reports(report, output_dir=output_dir)
        self.stdout.write(self.style.SUCCESS("llm_thought_chain analysis completed"))
        self.stdout.write(f"book_id={user_book.id}")
        self.stdout.write(f"report_json={paths['json']}")
        self.stdout.write(f"report_md={paths['md']}")
        self.stdout.write(
            f"sentences={report.get('total_sentences')} thoughts={report.get('thoughts_created')} "
            f"groups={report.get('sequential_groups_created')} blocks={report.get('global_blocks_created')}"
        )
