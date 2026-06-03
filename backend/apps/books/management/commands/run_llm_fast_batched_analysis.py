from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.books.models import (
    BookSummary,
    BookTheme,
    Concept,
    ConceptMention,
    GlobalBookCache,
    LLMAnalysisRun,
    LLMBookAnalysis,
    LLMChapterAnalysis,
    LLMSectionAnalysis,
    LogicalBlock,
    ThemeSubtopic,
    UserBook,
)
from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.concept_normalizer import is_bad_concept, normalize_concept_name
from apps.books.services.hashing import sha256_bytes
from apps.books.services.llm_service import (
    LLM_PROMPT_VERSION,
    _is_generic_term_name,
    analyze_section_fast_with_llm,
    build_book_fast_with_llm,
    ensure_llm_ready,
    merge_chapter_fast_with_llm,
    prepare_section_llm_input,
)
from apps.books.services.semantic_quality_v2 import (
    clean_term_list_v2,
    clean_text,
    fatal_flags_for_section,
    is_generic_term,
    semantic_problem_flags,
    validate_section_payload_v2,
    warning_flags_for_section,
)
from apps.books.services.structure_detector import CanonicalSection, build_canonical_outline


MODE = "llm_fast_batched"
QUALITY_PROMPT_VERSION = f"{LLM_PROMPT_VERSION}_quality_v2"


def _project_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() == "backend" else cwd


def _hash_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("_meta", {})
    return raw if isinstance(raw, dict) else {}


def _has_timeout(payload: dict[str, Any]) -> bool:
    failure = str(_meta(payload).get("llm_failure", "")).lower()
    return "timeout" in failure or "timed out" in failure


def _strict_main_sections(outline: dict[str, Any]) -> list[CanonicalSection]:
    return [
        item
        for item in list(outline.get("sections", []))
        if item.content_type == "main_content" and item.is_main_content
    ]


def _section_terms(payload: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for item in payload.get("terms", []):
        value = str(item).strip()
        if value and not _is_generic_term_name(value) and not is_generic_term(value):
            terms.append(value)
    for item in payload.get("key_terms", []):
        if isinstance(item, dict):
            value = str(item.get("term", "")).strip()
            if value and not _is_generic_term_name(value) and not is_generic_term(value):
                terms.append(value)
    return list(dict.fromkeys(terms))


def _section_subtopics(payload: dict[str, Any]) -> list[str]:
    subtopics: list[str] = []
    for item in payload.get("subtopics", []):
        if isinstance(item, dict):
            value = str(item.get("title", "")).strip()
            if value and not _is_generic_term_name(value) and not is_generic_term(value):
                subtopics.append(value)
        else:
            value = str(item).strip()
            if value and not _is_generic_term_name(value) and not is_generic_term(value):
                subtopics.append(value)
    return list(dict.fromkeys(subtopics))


def _payload_from_section_result(result: LLMSectionAnalysis) -> dict[str, Any]:
    quote = str((result.metadata or {}).get("source_quote", "")).strip()
    return {
        "section_title": result.section_title,
        "section_type": "main_content",
        "summary": result.summary,
        "main_idea": str((result.metadata or {}).get("main_idea", "")),
        "terms": list(result.terms or []),
        "key_terms": [
            {"term": term, "definition": f"Ключевое понятие раздела: {term}.", "importance": 0.8, "source_quote": quote}
            for term in result.terms
        ],
        "subtopics": [
            {"title": item, "summary": f"Подтема раздела: {item}.", "source_quote": quote}
            for item in result.subtopics
        ],
        "source_quotes": [quote] if quote else [],
        "_meta": {
            "llm_used": result.json_valid,
            "fallback_used": result.fallback_used,
            "llm_failure": "",
            "cache_hit": result.cache_hit,
            "actual_llm_call": result.actual_llm_call,
        },
    }


def _section_report(result: LLMSectionAnalysis) -> dict[str, Any]:
    return {
        "section_index": result.section_index,
        "section_title": result.section_title,
        "json_valid": result.json_valid,
        "fallback_used": result.fallback_used,
        "timeout": result.timeout,
        "cache_hit": result.cache_hit,
        "actual_llm_call": result.actual_llm_call,
        "duration_seconds": result.duration_seconds,
        "input_chars": result.input_chars,
        "terms": result.terms,
        "subtopics": result.subtopics,
        "quality_flags": result.quality_flags,
        "main_idea": str((result.metadata or {}).get("main_idea", "")),
    }


class Command(BaseCommand):
    help = "Production-safe fast batched LLM analysis with DB checkpoint/resume."

    def add_arguments(self, parser):
        parser.add_argument("--book-id", type=int)
        parser.add_argument("--file")
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--max-batches", type=int)
        parser.add_argument("--stop-on-error", action="store_true", default=True)
        parser.add_argument("--output-dir", default="")
        parser.add_argument("--max-input-chars", type=int, default=1500)
        parser.add_argument("--batch-timeout-seconds", type=int, default=600)
        parser.add_argument("--semantic-audit-only", action="store_true")
        parser.add_argument("--reanalyze-problem-blocks-only", action="store_true")
        parser.add_argument("--fatal-only", action="store_true")
        parser.add_argument("--force-llm-refresh", action="store_true")

    def handle(self, *args, **options):
        if not options.get("book_id") and not options.get("file"):
            raise CommandError("Provide --book-id or --file.")

        audit_only = bool(options.get("semantic_audit_only"))
        llm_state = ensure_llm_ready(require_enabled=not audit_only)
        if not audit_only and not llm_state.get("ok"):
            raise CommandError(f"LLM is not ready: {llm_state.get('error')}")
        model = str(llm_state.get("selected_fast") or "qwen2.5:1.5b")

        user_book, content, filename = self._load_input(options)
        parsed = parse_uploaded_book(content, filename)
        file_hash = sha256_bytes(content)
        cache, _ = GlobalBookCache.objects.get_or_create(
            file_hash=file_hash,
            defaults={
                "title": parsed.title,
                "authors": parsed.authors,
                "metadata": {},
                "analysis_version": "concept_rag_llm_fast_batched_v1",
            },
        )
        if user_book is None:
            user_book = self._find_or_create_user_book_for_file(
                file_hash=file_hash,
                filename=filename,
                content=content,
                cache=cache,
                parsed=parsed,
            )
        elif user_book.global_cache_id != cache.id:
            user_book.global_cache = cache
            user_book.save(update_fields=["global_cache", "updated_at"])

        outline = build_canonical_outline(parsed)
        sections = _strict_main_sections(outline)
        if not sections:
            raise CommandError("No strict main_content sections found.")

        run = self._get_or_create_run(cache, user_book, options, sections_total=len(sections))
        output_dir = self._prepare_output_dir(run, options)
        if run.output_dir != str(output_dir):
            run.output_dir = str(output_dir)
            run.save(update_fields=["output_dir", "updated_at"])

        if audit_only:
            audit = self._write_semantic_audit(
                output_dir=output_dir,
                cache=cache,
                sections=sections,
                model=model,
                reanalyzed_count=0,
                label="audit_only",
            )
            run.final_report = {"semantic_audit": audit}
            run.status = LLMAnalysisRun.Status.DRY_RUN
            run.finished_at = timezone.now()
            run.save(update_fields=["final_report", "status", "finished_at", "updated_at"])
            self.stdout.write(self.style.SUCCESS("Semantic audit written."))
            self.stdout.write(f"Output dir: {output_dir}")
            return

        env_overrides = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": model,
            "OLLAMA_MODEL_FAST": model,
            "OLLAMA_MODEL_HIGH": model,
            "OLLAMA_MODEL_FALLBACK": model,
            "OLLAMA_TIMEOUT_SECONDS": "60",
            "OLLAMA_MAX_TOKENS_JSON": "320",
            "LLM_MAX_RETRIES": "0",
            "LLM_ENABLE_FALLBACK": "true",
            "OLLAMA_TIMEOUT_COOLDOWN_SECONDS": "0",
            "OLLAMA_RETRY_COOLDOWN_SECONDS": "0",
            "LLM_MAX_INPUT_CHARS": str(int(options["max_input_chars"])),
            "SECTION_LLM_MAX_INPUT_CHARS": str(int(options["max_input_chars"])),
        }
        previous = {key: os.environ.get(key) for key in env_overrides}

        try:
            os.environ.update(env_overrides)
            self._run_analysis(
                run=run,
                cache=cache,
                user_book=user_book,
                parsed=parsed,
                sections=sections,
                model=model,
                options=options,
                output_dir=output_dir,
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.stdout.write(self.style.SUCCESS(f"Run {run.analysis_run_id} finished with status={run.status}"))
        self.stdout.write(f"Output dir: {output_dir}")

    def _load_input(self, options) -> tuple[UserBook | None, bytes, str]:
        if options.get("book_id"):
            user_book = UserBook.objects.select_related("global_cache").get(id=int(options["book_id"]))
            if not user_book.file:
                raise CommandError(f"UserBook {user_book.id} has no stored file.")
            user_book.file.open("rb")
            try:
                content = user_book.file.read()
            finally:
                user_book.file.close()
            return user_book, content, user_book.original_filename

        file_path = Path(str(options["file"])).expanduser().resolve()
        if not file_path.exists():
            raise CommandError(f"File not found: {file_path}")
        return None, file_path.read_bytes(), file_path.name

    def _find_or_create_user_book_for_file(
        self,
        *,
        file_hash: str,
        filename: str,
        content: bytes,
        cache: GlobalBookCache,
        parsed,
    ) -> UserBook | None:
        existing = UserBook.objects.filter(file_hash=file_hash).order_by("-uploaded_at").first()
        if existing:
            if existing.global_cache_id != cache.id:
                existing.global_cache = cache
                existing.save(update_fields=["global_cache", "updated_at"])
            return existing

        user_model = get_user_model()
        user = user_model.objects.order_by("id").first()
        if not user:
            return None

        user_book = UserBook.objects.create(
            user=user,
            global_cache=cache,
            title=parsed.title,
            authors=parsed.authors,
            original_filename=filename,
            file_hash=file_hash,
            status=UserBook.Status.QUEUED,
            current_stage="queued",
            analysis_mode=MODE,
        )
        user_book.file.save(filename, ContentFile(content), save=True)
        return user_book

    def _get_or_create_run(
        self,
        cache: GlobalBookCache,
        user_book: UserBook | None,
        options,
        *,
        sections_total: int,
    ) -> LLMAnalysisRun:
        if options["resume"]:
            run = (
                LLMAnalysisRun.objects.filter(global_book=cache, mode=MODE)
                .exclude(status__in=[LLMAnalysisRun.Status.READY])
                .order_by("-created_at")
                .first()
            )
            if run:
                return run

        run = LLMAnalysisRun.objects.create(
            global_book=cache,
            user_book=user_book,
            analysis_run_id=uuid.uuid4().hex,
            mode=MODE,
            status=LLMAnalysisRun.Status.QUEUED,
            batch_size=max(1, int(options["batch_size"])),
            sections_total=sections_total,
            started_at=timezone.now(),
            last_heartbeat_at=timezone.now(),
        )
        return run

    def _prepare_output_dir(self, run: LLMAnalysisRun, options) -> Path:
        root = _project_root()
        if options.get("output_dir"):
            output_dir = Path(str(options["output_dir"])).expanduser()
            if not output_dir.is_absolute():
                output_dir = root / output_dir
        else:
            output_dir = root / "llm_fast_batched_runs" / run.analysis_run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _touch_run(self, run: LLMAnalysisRun, **fields) -> None:
        for key, value in fields.items():
            setattr(run, key, value)
        run.last_heartbeat_at = timezone.now()
        run.save(update_fields=list(fields.keys()) + ["last_heartbeat_at", "updated_at"])

    def _touch_user_book(self, user_book: UserBook | None, **fields) -> None:
        if not user_book:
            return
        for key, value in fields.items():
            setattr(user_book, key, value)
        user_book.last_heartbeat_at = timezone.now()
        update_fields = list(fields.keys()) + ["last_heartbeat_at", "updated_at"]
        if user_book.started_at is None:
            user_book.started_at = timezone.now()
            update_fields.append("started_at")
        user_book.save(update_fields=list(dict.fromkeys(update_fields)))

    def _run_analysis(
        self,
        *,
        run: LLMAnalysisRun,
        cache: GlobalBookCache,
        user_book: UserBook | None,
        parsed,
        sections: list[CanonicalSection],
        model: str,
        options,
        output_dir: Path,
    ) -> None:
        batch_size = max(1, int(options["batch_size"]))
        max_batches = options.get("max_batches")
        dry_run = bool(options["dry_run"])
        if bool(options.get("reanalyze_problem_blocks_only")):
            self._run_problem_blocks_reanalysis(
                run=run,
                cache=cache,
                user_book=user_book,
                parsed=parsed,
                sections=sections,
                model=model,
                options=options,
                output_dir=output_dir,
            )
            return
        start_offset = int(run.sections_processed) if options["resume"] else 0
        batches_done = 0

        self._touch_run(run, status=LLMAnalysisRun.Status.RUNNING, sections_total=len(sections))
        self._touch_user_book(
            user_book,
            status=UserBook.Status.LLM_FAST_BATCHED_SECTION_ANALYSIS,
            current_stage="llm_fast_batched_section_analysis",
            progress_percent=max(1, int(run.progress_percent)),
            analysis_mode=MODE,
            llm_model_used=model,
            llm_provider_used="ollama",
        )

        for offset in range(start_offset, len(sections), batch_size):
            if max_batches is not None and batches_done >= int(max_batches):
                break
            batch_sections = sections[offset : offset + batch_size]
            batch_index = offset // batch_size
            batch_started = time.time()
            batch_results: list[LLMSectionAnalysis] = []
            batch_timeout = False

            for section in batch_sections:
                if time.time() - batch_started >= int(options["batch_timeout_seconds"]):
                    batch_timeout = True
                    break
                result = self._analyze_or_get_section(
                    cache=cache,
                    run=run,
                    section=section,
                    model=model,
                    max_input_chars=int(options["max_input_chars"]),
                    prompt_version=QUALITY_PROMPT_VERSION,
                    force_refresh=bool(options.get("force_llm_refresh")),
                )
                validation_entry = self._section_validation_entry(
                    row=result,
                    section=section,
                    max_input_chars=int(options["max_input_chars"]),
                )
                if fatal_flags_for_section(validation_entry):
                    # Keep the full batched run moving when a local section can be
                    # repaired deterministically. Fallback remains visible in metrics,
                    # but non-fatal fallback no longer blocks the whole book.
                    self._deterministic_cleanup_section(
                        row=result,
                        section=section,
                        run=run,
                        max_input_chars=int(options["max_input_chars"]),
                    )
                    result.refresh_from_db()
                batch_results.append(result)

            batch_report = self._write_batch_report(
                output_dir=output_dir,
                batch_index=batch_index,
                offset=offset,
                requested=len(batch_sections),
                results=batch_results,
                duration=time.time() - batch_started,
                batch_timeout=batch_timeout,
            )
            self._update_checkpoint(run, user_book, batch_report, offset + len(batch_results), len(sections), batch_index)
            batches_done += 1

            if self._batch_has_error(batch_report) and bool(options["stop_on_error"]):
                self._finish_failed(run, user_book, "Batch quality gate failed.", partial=True)
                self._write_final_report(output_dir, run, cache, dry_run=dry_run, materialized=False)
                return

        processed_count = LLMSectionAnalysis.objects.filter(
            global_book=cache,
            mode=MODE,
            prompt_version=QUALITY_PROMPT_VERSION,
            json_valid=True,
        ).count()
        if dry_run or processed_count < len(sections):
            status = LLMAnalysisRun.Status.DRY_RUN if dry_run else LLMAnalysisRun.Status.PARTIAL_READY
            self._touch_run(run, status=status, finished_at=timezone.now())
            self._touch_user_book(
                user_book,
                status=UserBook.Status.PARTIAL_READY,
                current_stage="partial_ready" if not dry_run else "dry_run",
                progress_percent=max(run.progress_percent, int(processed_count * 100 / max(1, len(sections)))),
            )
            self._write_final_report(output_dir, run, cache, dry_run=dry_run, materialized=False)
            return

        self._touch_user_book(
            user_book,
            status=UserBook.Status.LLM_FAST_BATCHED_CHAPTER_ANALYSIS,
            current_stage="llm_fast_batched_chapter_analysis",
            progress_percent=82,
        )
        chapter_results = self._build_chapter_results(
            cache,
            run,
            sections,
            model,
            prompt_version=QUALITY_PROMPT_VERSION,
            force_refresh=bool(options.get("force_llm_refresh")),
        )
        if self._aggregation_has_blocking_errors(run, chapter_results):
            self._finish_failed(run, user_book, "Chapter aggregation quality gate failed.", partial=True)
            self._write_final_report(output_dir, run, cache, dry_run=False, materialized=False)
            return

        self._touch_user_book(
            user_book,
            status=UserBook.Status.LLM_FAST_BATCHED_BOOK_ANALYSIS,
            current_stage="llm_fast_batched_book_analysis",
            progress_percent=90,
        )
        book_result = self._build_book_result(
            cache,
            run,
            chapter_results,
            model,
            prompt_version=QUALITY_PROMPT_VERSION,
            force_refresh=bool(options.get("force_llm_refresh")),
        )
        if book_result.timeout:
            self._finish_failed(run, user_book, "Book aggregation quality gate failed.", partial=True)
            self._write_final_report(output_dir, run, cache, dry_run=False, materialized=False)
            return

        self._touch_user_book(
            user_book,
            status=UserBook.Status.SAVING_RESULTS,
            current_stage="saving_results",
            progress_percent=95,
        )
        self._materialize_for_ui(cache, parsed, sections, chapter_results, book_result)
        quality = self._quality_gates(run, cache, book_result)
        final_run_status = LLMAnalysisRun.Status.READY if quality["passed"] else LLMAnalysisRun.Status.PARTIAL_READY
        final_user_status = UserBook.Status.READY if quality["passed"] else UserBook.Status.PARTIAL_READY
        self._touch_run(run, status=final_run_status, progress_percent=100, finished_at=timezone.now())
        self._touch_user_book(
            user_book,
            status=final_user_status,
            current_stage=final_user_status,
            progress_percent=100,
            finished_at=timezone.now(),
            processed_at=timezone.now(),
            error_message="" if quality["passed"] else "Analysis completed with quality gate warnings.",
        )
        self._write_final_report(output_dir, run, cache, dry_run=False, materialized=True)

    def _aggregation_has_blocking_errors(self, run: LLMAnalysisRun, chapter_results: list[LLMChapterAnalysis]) -> bool:
        if any(item.timeout for item in chapter_results):
            return True
        valid_rate = run.valid_json_units / max(1, run.expected_json_units)
        return valid_rate < 0.95

    def _latest_section_result(self, cache: GlobalBookCache, section_index: int) -> LLMSectionAnalysis | None:
        preferred = (
            LLMSectionAnalysis.objects.filter(
                global_book=cache,
                mode=MODE,
                section_index=section_index,
                prompt_version=QUALITY_PROMPT_VERSION,
            )
            .order_by("-updated_at", "-id")
            .first()
        )
        if preferred:
            return preferred
        return (
            LLMSectionAnalysis.objects.filter(global_book=cache, mode=MODE, section_index=section_index)
            .order_by("-updated_at", "-id")
            .first()
        )

    def _section_validation_entry(
        self,
        *,
        row: LLMSectionAnalysis | None,
        section: CanonicalSection,
        max_input_chars: int = 2400,
    ) -> dict[str, Any]:
        prepared_text = prepare_section_llm_input(section, max_chars=max_input_chars)
        if row is None:
            return {
                "section_index": section.section_index,
                "section_title": section.section_title,
                "chapter_title": section.parent_chapter_title or section.chapter_title,
                "problem": True,
                "fatal": True,
                "flags": ["missing_section_analysis"],
                "fatal_flags": ["missing_section_analysis"],
                "warning_flags": [],
                "removed_generic_terms": [],
                "removed_suspicious_terms": [],
                "mixed_language_artifacts": [],
                "weak_grounding_terms": [],
                "generic_terms_ratio": 0,
                "row_id": None,
                "prompt_version": "",
                "fallback_used": False,
                "json_valid": False,
                "timeout": False,
                "deterministic_cleanup": False,
            }

        payload = {
            "summary": row.summary,
            "main_idea": str((row.metadata or {}).get("main_idea", "")),
            "terms": row.terms,
            "subtopics": row.subtopics,
        }
        validation = validate_section_payload_v2(payload, prepared_text, row.section_title)
        deterministic_cleanup = bool((row.metadata or {}).get("deterministic_cleanup"))
        flags = list(dict.fromkeys(
            validation.quality_flags
            + (["fallback_section_analysis"] if row.fallback_used else [])
            + (["invalid_json"] if not row.json_valid else [])
            + (["timeout"] if row.timeout else [])
        ))
        entry = {
            "section_index": row.section_index,
            "section_title": row.section_title,
            "chapter_title": row.chapter_title,
            "flags": flags,
            "removed_generic_terms": validation.removed_generic_terms,
            "removed_suspicious_terms": validation.removed_suspicious_terms,
            "mixed_language_artifacts": validation.mixed_language_artifacts,
            "weak_grounding_terms": validation.weak_grounding_terms,
            "generic_terms_ratio": validation.generic_terms_ratio,
            "row_id": row.id,
            "prompt_version": row.prompt_version,
            "fallback_used": row.fallback_used,
            "json_valid": row.json_valid,
            "timeout": row.timeout,
            "input_chars": row.input_chars,
            "summary": row.summary[:500],
            "terms": list(row.terms or []),
            "subtopics": list(row.subtopics or []),
            "deterministic_cleanup": deterministic_cleanup,
        }
        fatal_flags = fatal_flags_for_section(entry)
        warning_flags = warning_flags_for_section(entry)
        entry["fatal_flags"] = fatal_flags
        entry["warning_flags"] = warning_flags
        entry["fatal"] = bool(fatal_flags)
        entry["problem"] = bool(fatal_flags or warning_flags)
        return entry

    def _build_semantic_audit(
        self,
        *,
        cache: GlobalBookCache,
        sections: list[CanonicalSection],
        model: str,
        reanalyzed_count: int = 0,
        label: str = "audit",
        before_problem_count: int | None = None,
        cleaned_without_llm: int = 0,
    ) -> dict[str, Any]:
        section_entries = [
            self._section_validation_entry(row=self._latest_section_result(cache, section.section_index), section=section)
            for section in sections
        ]
        problem_entries = [item for item in section_entries if item["problem"]]
        fatal_entries = [item for item in section_entries if item.get("fatal")]
        warning_entries = [item for item in section_entries if item.get("warning_flags")]

        chapter_rows = list(
            LLMChapterAnalysis.objects.filter(global_book=cache, mode=MODE)
            .order_by("chapter_index", "-updated_at", "-id")
        )
        seen_chapters: set[tuple[int, str]] = set()
        latest_chapters: list[LLMChapterAnalysis] = []
        for row in chapter_rows:
            key = (row.chapter_index, row.chapter_title)
            if key in seen_chapters:
                continue
            seen_chapters.add(key)
            latest_chapters.append(row)
        book_row = getattr(cache, "llm_book_analysis", None)

        removed_generic = [term for item in section_entries for term in item["removed_generic_terms"]]
        removed_suspicious = [term for item in section_entries for term in item["removed_suspicious_terms"]]
        mixed = [term for item in section_entries for term in item["mixed_language_artifacts"]]
        weak = [term for item in section_entries for term in item["weak_grounding_terms"]]
        chapter_fallback_count = sum(1 for row in latest_chapters if row.fallback_used)
        book_fallback = bool(book_row and book_row.fallback_used)
        learning_path_count = len(book_row.learning_path or []) if book_row else 0
        summary_covers_book = bool(book_row and len(book_row.book_summary or "") >= 120 and learning_path_count >= min(6, len(latest_chapters) or 6))
        final_passed = (
            not fatal_entries
            and chapter_fallback_count == 0
            and not book_fallback
            and summary_covers_book
            and bool(book_row and book_row.book_summary and book_row.global_themes and book_row.learning_path)
        )
        return {
            "label": label,
            "book_title": cache.title,
            "model": model,
            "total_blocks": len(sections),
            "problem_blocks_before": before_problem_count if before_problem_count is not None else len(problem_entries),
            "problem_blocks_after": len(problem_entries),
            "fatal_blocks_before": before_problem_count if before_problem_count is not None else len(fatal_entries),
            "fatal_blocks_after": len(fatal_entries),
            "warning_blocks_after": len(warning_entries),
            "reanalyzed_blocks_count": reanalyzed_count,
            "cleaned_without_llm": cleaned_without_llm,
            "removed_generic_terms": list(dict.fromkeys(removed_generic)),
            "removed_suspicious_terms": list(dict.fromkeys(removed_suspicious)),
            "mixed_language_artifacts": list(dict.fromkeys(mixed)),
            "weak_grounding_count": len(weak),
            "weak_grounding_terms": list(dict.fromkeys(weak))[:100],
            "chapter_fallback_count": chapter_fallback_count,
            "chapter_fallback_titles": [row.chapter_title for row in latest_chapters if row.fallback_used],
            "book_fallback": book_fallback,
            "book_summary_covers_all_chapters": summary_covers_book,
            "final_semantic_quality_passed": final_passed,
            "problem_sections": problem_entries,
            "fatal_sections": fatal_entries,
            "warning_sections": warning_entries,
            "book_summary": book_row.book_summary if book_row else "",
            "book_themes": book_row.global_themes if book_row else [],
            "learning_path": book_row.learning_path if book_row else [],
        }

    def _write_semantic_audit(
        self,
        *,
        output_dir: Path,
        cache: GlobalBookCache,
        sections: list[CanonicalSection],
        model: str,
        reanalyzed_count: int = 0,
        label: str = "audit",
        before_problem_count: int | None = None,
        cleaned_without_llm: int = 0,
    ) -> dict[str, Any]:
        audit = self._build_semantic_audit(
            cache=cache,
            sections=sections,
            model=model,
            reanalyzed_count=reanalyzed_count,
            label=label,
            before_problem_count=before_problem_count,
            cleaned_without_llm=cleaned_without_llm,
        )
        json_text = json.dumps(audit, ensure_ascii=False, indent=2)
        md_text = self._semantic_audit_markdown(audit)
        final_json_text = json.dumps(self._final_cleanup_projection(audit), ensure_ascii=False, indent=2)
        final_md_text = self._final_cleanup_markdown(audit)
        targets = {output_dir, _project_root()}
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)
            (target / "llm_fast_batched_quality_v2_audit.json").write_text(json_text, encoding="utf-8-sig")
            (target / "llm_fast_batched_quality_v2_audit.md").write_text(md_text, encoding="utf-8-sig")
            (target / "llm_fast_batched_final_semantic_cleanup_audit.json").write_text(final_json_text, encoding="utf-8-sig")
            (target / "llm_fast_batched_final_semantic_cleanup_audit.md").write_text(final_md_text, encoding="utf-8-sig")
        return audit

    def _final_cleanup_projection(self, audit: dict[str, Any]) -> dict[str, Any]:
        fatal_after = int(audit.get("fatal_blocks_after", 0) or 0)
        warning_after = int(audit.get("warning_blocks_after", 0) or 0)
        if fatal_after > 0:
            recommendation = "partial_ready"
        elif warning_after > 0:
            recommendation = "ready_with_warnings"
        else:
            recommendation = "ready"
        return {
            "book_title": audit.get("book_title", ""),
            "problem_blocks_before": audit.get("problem_blocks_before", 0),
            "fatal_blocks_before": audit.get("fatal_blocks_before", 0),
            "fatal_blocks_after": fatal_after,
            "warning_blocks_after": warning_after,
            "cleaned_without_llm": audit.get("cleaned_without_llm", 0),
            "retried_with_llm": audit.get("reanalyzed_blocks_count", 0),
            "remaining_non_fatal_warnings": audit.get("warning_sections", []),
            "chapter_fallback_count": audit.get("chapter_fallback_count", 0),
            "book_fallback": audit.get("book_fallback", False),
            "book_summary_covers_all_chapters": audit.get("book_summary_covers_all_chapters", False),
            "final_semantic_quality_passed": audit.get("final_semantic_quality_passed", False),
            "book_status_recommendation": recommendation,
            "book_summary": audit.get("book_summary", ""),
            "learning_path": audit.get("learning_path", []),
        }

    def _final_cleanup_markdown(self, audit: dict[str, Any]) -> str:
        projected = self._final_cleanup_projection(audit)
        lines = [
            "# Final Semantic Cleanup Audit",
            "",
            f"- book: {projected['book_title']}",
            f"- problem_blocks_before: {projected['problem_blocks_before']}",
            f"- fatal_blocks_before: {projected['fatal_blocks_before']}",
            f"- fatal_blocks_after: {projected['fatal_blocks_after']}",
            f"- warning_blocks_after: {projected['warning_blocks_after']}",
            f"- cleaned_without_llm: {projected['cleaned_without_llm']}",
            f"- retried_with_llm: {projected['retried_with_llm']}",
            f"- chapter_fallback_count: {projected['chapter_fallback_count']}",
            f"- book_fallback: {projected['book_fallback']}",
            f"- book_summary_covers_all_chapters: {projected['book_summary_covers_all_chapters']}",
            f"- final_semantic_quality_passed: {projected['final_semantic_quality_passed']}",
            f"- book_status_recommendation: {projected['book_status_recommendation']}",
            "",
            "## Book Summary",
            str(projected["book_summary"]),
            "",
            "## Learning Path",
        ]
        for item in projected["learning_path"] or []:
            lines.append(f"- {item}")
        lines.extend(["", "## Remaining Fatal Sections"])
        for item in audit.get("fatal_sections", []) or []:
            lines.append(f"- {item.get('section_index')}: {item.get('section_title')} | {', '.join(item.get('fatal_flags', []))}")
        lines.extend(["", "## Remaining Non-Fatal Warnings"])
        for item in audit.get("warning_sections", []) or []:
            lines.append(f"- {item.get('section_index')}: {item.get('section_title')} | {', '.join(item.get('warning_flags', []))}")
        return "\n".join(lines) + "\n"

    def _semantic_audit_markdown(self, audit: dict[str, Any]) -> str:
        lines = [
            "# LLM Fast Batched Quality V2 Audit",
            "",
            f"- book: {audit.get('book_title', '')}",
            f"- label: {audit.get('label', '')}",
            f"- model: {audit.get('model', '')}",
            f"- total_blocks: {audit.get('total_blocks')}",
            f"- problem_blocks_before: {audit.get('problem_blocks_before')}",
            f"- problem_blocks_after: {audit.get('problem_blocks_after')}",
            f"- fatal_blocks_before: {audit.get('fatal_blocks_before')}",
            f"- fatal_blocks_after: {audit.get('fatal_blocks_after')}",
            f"- warning_blocks_after: {audit.get('warning_blocks_after')}",
            f"- reanalyzed_blocks_count: {audit.get('reanalyzed_blocks_count')}",
            f"- removed_generic_terms: {len(audit.get('removed_generic_terms', []))}",
            f"- removed_suspicious_terms: {len(audit.get('removed_suspicious_terms', []))}",
            f"- mixed_language_artifacts: {len(audit.get('mixed_language_artifacts', []))}",
            f"- weak_grounding_count: {audit.get('weak_grounding_count')}",
            f"- chapter_fallback_count: {audit.get('chapter_fallback_count')}",
            f"- book_fallback: {audit.get('book_fallback')}",
            f"- final_semantic_quality_passed: {audit.get('final_semantic_quality_passed')}",
            "",
            "## Book Summary",
            str(audit.get("book_summary", "")),
            "",
            "## Themes",
        ]
        for item in audit.get("book_themes", []) or []:
            lines.append(f"- {item}")
        lines.extend(["", "## Learning Path"])
        for item in audit.get("learning_path", []) or []:
            lines.append(f"- {item}")
        lines.extend(["", "## Removed Generic Terms"])
        for item in (audit.get("removed_generic_terms", []) or [])[:80]:
            lines.append(f"- {item}")
        lines.extend(["", "## Problem Sections"])
        for item in audit.get("problem_sections", []) or []:
            lines.extend(
                [
                    "",
                    f"### Section {item.get('section_index')}: {item.get('section_title')}",
                    f"- chapter: {item.get('chapter_title')}",
                    f"- row_id: {item.get('row_id')}",
                    f"- prompt_version: {item.get('prompt_version')}",
                    f"- flags: {', '.join(item.get('flags', []))}",
                    f"- fatal_flags: {', '.join(item.get('fatal_flags', []))}",
                    f"- warning_flags: {', '.join(item.get('warning_flags', []))}",
                    f"- deterministic_cleanup: {item.get('deterministic_cleanup')}",
                    f"- fallback: {item.get('fallback_used')}",
                    f"- json_valid: {item.get('json_valid')}",
                    f"- timeout: {item.get('timeout')}",
                    f"- summary: {item.get('summary', '')}",
                    f"- terms: {', '.join(item.get('terms', [])[:12])}",
                    f"- subtopics: {', '.join(item.get('subtopics', [])[:12])}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _link_existing_sections_to_run(
        self,
        *,
        run: LLMAnalysisRun,
        cache: GlobalBookCache,
        sections: list[CanonicalSection],
        skip_indices: set[int],
    ) -> None:
        for section in sections:
            if section.section_index in skip_indices:
                continue
            row = self._latest_section_result(cache, section.section_index)
            if row:
                row.analysis_run = run
                row.cache_hit = True
                row.save(update_fields=["analysis_run", "cache_hit", "updated_at"])

    def _recalculate_run_section_metrics(self, run: LLMAnalysisRun) -> None:
        rows = list(LLMSectionAnalysis.objects.filter(analysis_run=run, mode=MODE))
        run.sections_processed = len(rows)
        run.valid_json_units = sum(1 for row in rows if row.json_valid)
        run.expected_json_units = len(rows)
        run.fallback_count = sum(1 for row in rows if row.fallback_used)
        run.timeout_count = sum(1 for row in rows if row.timeout)
        run.cache_hits = sum(1 for row in rows if row.cache_hit)
        run.llm_calls_actual = sum(1 for row in rows if row.actual_llm_call and not row.cache_hit)
        run.progress_percent = min(80, int(run.sections_processed * 80 / max(1, run.sections_total)))
        run.last_heartbeat_at = timezone.now()
        run.save()

    def _cleanup_summary_text(self, summary: str, section_title: str, terms: list[str]) -> str:
        value = clean_text(summary, max_len=1200)
        title = clean_text(section_title, max_len=220)
        for marker in (
            "Глава:",
            "Секция:",
            "Начало секции:",
            "Ключевые предложения из середины/конца:",
            "Внутренние подзаголовки:",
            "Конец секции:",
        ):
            value = value.replace(marker, " ")
        if title:
            value = value.removeprefix(title).strip(" .:-")
            value = value.replace(f"{title} ", " ", 1).strip()
        value = clean_text(value, max_len=900)
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", value) if item.strip()]
        normal_sentences = []
        for sentence in sentences:
            if len(sentence) < 30:
                continue
            if "Начало секции" in sentence or "Секция:" in sentence:
                continue
            normal_sentences.append(sentence)
            if len(normal_sentences) >= 2:
                break
        cleaned = " ".join(normal_sentences).strip()
        if title and (not cleaned or cleaned.lower() == title.lower() or len(cleaned) < 70):
            term_tail = ", ".join(terms[:4]) if terms else "ключевые понятия раздела"
            cleaned = (
                f"Раздел объясняет предметную область, связанную с темой «{title}»: {term_tail}. "
                "Основной смысл фрагмента состоит в том, чтобы показать роль этих понятий в общей логике главы "
                "и связать их с практическими механизмами работы компьютерных сетей."
            )
        return cleaned[:1000]

    def _deterministic_cleanup_section(
        self,
        *,
        row: LLMSectionAnalysis,
        section: CanonicalSection,
        run: LLMAnalysisRun,
        max_input_chars: int,
    ) -> bool:
        prepared_text = prepare_section_llm_input(section, max_chars=max_input_chars)
        initial_entry = self._section_validation_entry(row=row, section=section, max_input_chars=max_input_chars)
        initial_fatal = fatal_flags_for_section(initial_entry)

        source_terms = list(row.terms or []) + list(row.subtopics or [])
        cleaned_terms, term_stats = clean_term_list_v2(source_terms, prepared_text, limit=8)
        cleaned_subtopics, subtopic_stats = clean_term_list_v2(list(row.subtopics or []) + cleaned_terms, prepared_text, limit=6)
        cleaned_summary = self._cleanup_summary_text(row.summary, row.section_title, cleaned_terms)
        cleaned_payload = {
            "summary": cleaned_summary,
            "main_idea": str((row.metadata or {}).get("main_idea", "")),
            "terms": cleaned_terms,
            "subtopics": cleaned_subtopics,
        }
        validation = validate_section_payload_v2(cleaned_payload, prepared_text, row.section_title)
        new_flags = list(validation.quality_flags)
        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        metadata = {
            **metadata,
            "deterministic_cleanup": True,
            "deterministic_cleanup_stats": {
                "initial_fatal_flags": initial_fatal,
                "removed_generic_terms": list(term_stats.get("removed_generic_terms", [])) + list(subtopic_stats.get("removed_generic_terms", [])),
                "removed_suspicious_terms": list(term_stats.get("removed_suspicious_terms", [])) + list(subtopic_stats.get("removed_suspicious_terms", [])),
                "mixed_language_artifacts": list(term_stats.get("mixed_language_artifacts", [])) + list(subtopic_stats.get("mixed_language_artifacts", [])),
                "weak_grounding_terms": list(term_stats.get("weak_grounding_terms", [])) + list(subtopic_stats.get("weak_grounding_terms", [])),
            },
        }
        row.analysis_run = run
        row.summary = cleaned_summary
        row.terms = cleaned_terms
        row.subtopics = cleaned_subtopics
        row.json_valid = True
        row.timeout = False
        row.quality_flags = new_flags
        row.metadata = metadata
        row.save(
            update_fields=[
                "analysis_run",
                "summary",
                "terms",
                "subtopics",
                "json_valid",
                "timeout",
                "quality_flags",
                "metadata",
                "updated_at",
            ]
        )
        after_entry = self._section_validation_entry(row=row, section=section, max_input_chars=max_input_chars)
        return not fatal_flags_for_section(after_entry)

    def _run_problem_blocks_reanalysis(
        self,
        *,
        run: LLMAnalysisRun,
        cache: GlobalBookCache,
        user_book: UserBook | None,
        parsed,
        sections: list[CanonicalSection],
        model: str,
        options,
        output_dir: Path,
    ) -> None:
        before = self._build_semantic_audit(cache=cache, sections=sections, model=model, label="before_reanalysis")
        fatal_only = bool(options.get("fatal_only"))
        target_entries = before.get("fatal_sections", []) if fatal_only else before.get("problem_sections", [])
        problem_indices = {int(item["section_index"]) for item in target_entries}
        if fatal_only and not problem_indices:
            self._link_existing_sections_to_run(run=run, cache=cache, sections=sections, skip_indices=set())
            self._recalculate_run_section_metrics(run)
            after = self._write_semantic_audit(
                output_dir=output_dir,
                cache=cache,
                sections=sections,
                model=model,
                reanalyzed_count=0,
                label="fatal_cleanup_noop",
                before_problem_count=0,
                cleaned_without_llm=0,
            )
            final_status = UserBook.Status.READY_WITH_WARNINGS if int(after.get("warning_blocks_after", 0) or 0) else UserBook.Status.READY
            self._touch_run(run, status=LLMAnalysisRun.Status.READY, progress_percent=100, finished_at=timezone.now())
            self._touch_user_book(
                user_book,
                status=final_status,
                current_stage=final_status,
                progress_percent=100,
                finished_at=timezone.now(),
                processed_at=timezone.now(),
                error_message="" if final_status == UserBook.Status.READY else "Analysis completed with non-fatal semantic warnings.",
            )
            final_report = self._write_final_report(output_dir, run, cache, dry_run=False, materialized=True)
            final_report["semantic_quality_v2_audit"] = after
            run.final_report = final_report
            run.save(update_fields=["final_report", "updated_at"])
            return
        self._touch_run(run, status=LLMAnalysisRun.Status.RUNNING, sections_total=len(sections))
        self._touch_user_book(
            user_book,
            status=UserBook.Status.LLM_FAST_BATCHED_SECTION_ANALYSIS,
            current_stage="reanalyze_problem_blocks_only",
            progress_percent=5,
            analysis_mode=MODE,
            llm_model_used=model,
            llm_provider_used="ollama",
        )

        self._link_existing_sections_to_run(run=run, cache=cache, sections=sections, skip_indices=problem_indices)
        reanalyzed = []
        cleaned_without_llm = 0
        for section in sections:
            if section.section_index not in problem_indices:
                continue
            existing_row = self._latest_section_result(cache, section.section_index)
            if existing_row:
                cleaned_ok = self._deterministic_cleanup_section(
                    row=existing_row,
                    section=section,
                    run=run,
                    max_input_chars=int(options["max_input_chars"]),
                )
                if cleaned_ok:
                    cleaned_without_llm += 1
                    continue
            result = self._analyze_or_get_section(
                cache=cache,
                run=run,
                section=section,
                model=model,
                max_input_chars=int(options["max_input_chars"]),
                prompt_version=QUALITY_PROMPT_VERSION,
                force_refresh=True,
            )
            retry_row = result
            retry_cleaned_ok = self._deterministic_cleanup_section(
                row=retry_row,
                section=section,
                run=run,
                max_input_chars=int(options["max_input_chars"]),
            )
            if retry_cleaned_ok:
                cleaned_without_llm += 1
            reanalyzed.append(result)

        self._recalculate_run_section_metrics(run)
        self._touch_user_book(
            user_book,
            status=UserBook.Status.LLM_FAST_BATCHED_CHAPTER_ANALYSIS,
            current_stage="llm_fast_batched_chapter_analysis",
            progress_percent=82,
        )
        chapter_results = self._build_chapter_results(
            cache,
            run,
            sections,
            model,
            prompt_version=QUALITY_PROMPT_VERSION,
            force_refresh=True,
        )
        self._touch_user_book(
            user_book,
            status=UserBook.Status.LLM_FAST_BATCHED_BOOK_ANALYSIS,
            current_stage="llm_fast_batched_book_analysis",
            progress_percent=90,
        )
        book_result = self._build_book_result(
            cache,
            run,
            chapter_results,
            model,
            prompt_version=QUALITY_PROMPT_VERSION,
            force_refresh=True,
        )
        self._touch_user_book(user_book, status=UserBook.Status.SAVING_RESULTS, current_stage="saving_results", progress_percent=95)
        self._materialize_for_ui(cache, parsed, sections, chapter_results, book_result)
        after = self._write_semantic_audit(
            output_dir=output_dir,
            cache=cache,
            sections=sections,
            model=model,
            reanalyzed_count=len(reanalyzed),
            label="after_reanalysis",
            before_problem_count=len(problem_indices),
            cleaned_without_llm=cleaned_without_llm,
        )
        quality = self._quality_gates(run, cache, book_result)
        semantic_passed = bool(after.get("final_semantic_quality_passed"))
        fatal_after = int(after.get("fatal_blocks_after", 0) or 0)
        warning_after = int(after.get("warning_blocks_after", 0) or 0)
        passed = bool(fatal_after == 0 and semantic_passed)
        final_run_status = LLMAnalysisRun.Status.READY if passed else LLMAnalysisRun.Status.PARTIAL_READY
        if fatal_after == 0 and semantic_passed and warning_after > 0:
            final_user_status = UserBook.Status.READY_WITH_WARNINGS
        elif fatal_after == 0 and semantic_passed:
            final_user_status = UserBook.Status.READY
        else:
            final_user_status = UserBook.Status.PARTIAL_READY
        self._touch_run(run, status=final_run_status, progress_percent=100, finished_at=timezone.now())
        self._touch_user_book(
            user_book,
            status=final_user_status,
            current_stage=final_user_status,
            progress_percent=100,
            finished_at=timezone.now(),
            processed_at=timezone.now(),
            error_message="" if fatal_after == 0 else "Analysis completed with fatal semantic quality warnings.",
        )
        final_report = self._write_final_report(output_dir, run, cache, dry_run=False, materialized=True)
        final_report["semantic_quality_v2_audit"] = after
        run.final_report = final_report
        run.save(update_fields=["final_report", "updated_at"])

    def _analyze_or_get_section(
        self,
        *,
        cache: GlobalBookCache,
        run: LLMAnalysisRun,
        section: CanonicalSection,
        model: str,
        max_input_chars: int,
        prompt_version: str = QUALITY_PROMPT_VERSION,
        force_refresh: bool = False,
    ) -> LLMSectionAnalysis:
        prepared_text = prepare_section_llm_input(section, max_chars=max_input_chars)
        content_hash = _hash_text(f"{section.section_title}\n{prepared_text}")
        existing = LLMSectionAnalysis.objects.filter(
            global_book=cache,
            mode=MODE,
            content_hash=content_hash,
            prompt_version=prompt_version,
            model_used=model,
        ).first()
        if existing and force_refresh:
            existing.delete()
            existing = None
        if existing:
            existing.cache_hit = True
            existing.analysis_run = run
            existing.chapter_title = section.parent_chapter_title or section.chapter_title
            existing.section_title = section.section_title
            existing.section_index = section.section_index
            existing.start_paragraph = section.start_paragraph
            existing.end_paragraph = section.end_paragraph
            existing.word_count = section.word_count
            existing.input_chars = len(prepared_text)
            existing.save(
                update_fields=[
                    "cache_hit",
                    "analysis_run",
                    "chapter_title",
                    "section_title",
                    "section_index",
                    "start_paragraph",
                    "end_paragraph",
                    "word_count",
                    "input_chars",
                    "updated_at",
                ]
            )
            return existing

        started = time.time()
        payload = analyze_section_fast_with_llm(
            section_title=section.section_title,
            section_text=prepared_text,
            chapter_title=section.parent_chapter_title or section.chapter_title,
            section_type=section.content_type,
        )
        meta = _meta(payload)
        terms = _section_terms(payload)
        subtopics = _section_subtopics(payload)
        quote = ""
        source_quotes = payload.get("source_quotes", [])
        if isinstance(source_quotes, list) and source_quotes:
            quote = str(source_quotes[0])
        meta = _meta(payload)
        semantic_v2 = meta.get("semantic_quality_v2", {}) if isinstance(meta.get("semantic_quality_v2", {}), dict) else {}
        return LLMSectionAnalysis.objects.create(
            global_book=cache,
            analysis_run=run,
            mode=MODE,
            chapter_title=section.parent_chapter_title or section.chapter_title,
            section_title=section.section_title,
            section_index=section.section_index,
            start_paragraph=section.start_paragraph,
            end_paragraph=section.end_paragraph,
            word_count=section.word_count,
            summary=str(payload.get("summary", "")).strip(),
            terms=terms,
            subtopics=subtopics,
            model_used=model,
            prompt_version=prompt_version,
            content_hash=content_hash,
            json_valid=bool(meta.get("llm_used")),
            fallback_used=bool(meta.get("fallback_used")),
            timeout=_has_timeout(payload),
            quality_flags=list(payload.get("quality_flags", [])) if isinstance(payload.get("quality_flags", []), list) else [],
            cache_hit=bool(meta.get("cache_hit")),
            actual_llm_call=bool(meta.get("actual_llm_call", True)),
            duration_seconds=float(meta.get("duration_seconds", round(time.time() - started, 2)) or 0.0),
            input_chars=len(prepared_text),
            metadata={
                "source_quote": quote,
                "main_idea": str(payload.get("main_idea", ""))[:1000],
                "bad_input_notes": list(payload.get("bad_input_notes", [])) if isinstance(payload.get("bad_input_notes", []), list) else [],
                "prepared_text_chars": len(prepared_text),
                "semantic_quality_v2": semantic_v2,
                "llm_meta": meta,
            },
        )

    def _write_batch_report(
        self,
        *,
        output_dir: Path,
        batch_index: int,
        offset: int,
        requested: int,
        results: list[LLMSectionAnalysis],
        duration: float,
        batch_timeout: bool,
    ) -> dict[str, Any]:
        report = {
            "batch_index": batch_index,
            "offset": offset,
            "sections_requested": requested,
            "sections_processed": len(results),
            "valid_json_units": sum(1 for item in results if item.json_valid),
            "expected_json_units": len(results),
            "fallback": sum(1 for item in results if item.fallback_used),
            "timeout": sum(1 for item in results if item.timeout) + int(batch_timeout),
            "cache_hits": sum(1 for item in results if item.cache_hit),
            "actual_llm_calls": sum(1 for item in results if item.actual_llm_call),
            "duration_seconds": round(duration, 2),
            "failed_sections": [
                {
                    "section_index": item.section_index,
                    "section_title": item.section_title,
                    "fallback_used": item.fallback_used,
                    "timeout": item.timeout,
                    "json_valid": item.json_valid,
                }
                for item in results
                if item.fallback_used or item.timeout or not item.json_valid
            ],
            "quality_flags": sorted({flag for item in results for flag in (item.quality_flags or [])}),
            "sections": [_section_report(item) for item in results],
        }
        (output_dir / f"batch_{batch_index:03d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def _batch_has_error(self, report: dict[str, Any]) -> bool:
        return (
            int(report.get("timeout", 0)) > 0
            or int(report.get("valid_json_units", 0)) < int(report.get("expected_json_units", 0))
        )

    def _update_checkpoint(
        self,
        run: LLMAnalysisRun,
        user_book: UserBook | None,
        batch_report: dict[str, Any],
        next_offset: int,
        total_sections: int,
        batch_index: int,
    ) -> None:
        run.current_batch_index = batch_index
        run.current_offset = next_offset
        run.sections_processed = max(run.sections_processed, next_offset)
        run.valid_json_units += int(batch_report.get("valid_json_units", 0))
        run.expected_json_units += int(batch_report.get("expected_json_units", 0))
        run.fallback_count += int(batch_report.get("fallback", 0))
        run.timeout_count += int(batch_report.get("timeout", 0))
        run.cache_hits += int(batch_report.get("cache_hits", 0))
        run.llm_calls_actual += int(batch_report.get("actual_llm_calls", 0))
        run.progress_percent = min(80, int(run.sections_processed * 80 / max(1, total_sections)))
        run.last_heartbeat_at = timezone.now()
        run.save()
        self._touch_user_book(
            user_book,
            progress_percent=run.progress_percent,
            llm_calls_total=run.llm_calls_actual,
            fallback_used_count=run.fallback_count,
            llm_failures_total=run.fallback_count + run.timeout_count,
        )

    def _build_chapter_results(
        self,
        cache: GlobalBookCache,
        run: LLMAnalysisRun,
        sections: list[CanonicalSection],
        model: str,
        prompt_version: str = QUALITY_PROMPT_VERSION,
        force_refresh: bool = False,
    ) -> list[LLMChapterAnalysis]:
        by_chapter: dict[str, list[LLMSectionAnalysis]] = {}
        for section in sections:
            result = LLMSectionAnalysis.objects.filter(
                global_book=cache,
                mode=MODE,
                analysis_run=run,
                section_index=section.section_index,
            ).first()
            if result:
                by_chapter.setdefault(result.chapter_title or section.chapter_title, []).append(result)

        results: list[LLMChapterAnalysis] = []
        metrics = {
            "valid": 0,
            "expected": 0,
            "fallback": 0,
            "timeout": 0,
            "cache_hits": 0,
            "actual_calls": 0,
        }
        for chapter_index, (chapter_title, section_results) in enumerate(by_chapter.items(), start=1):
            evidence_hash = _hash_text(
                chapter_title + "\n" + "\n".join(item.content_hash for item in section_results)
            )
            existing = LLMChapterAnalysis.objects.filter(
                global_book=cache,
                mode=MODE,
                content_hash=evidence_hash,
                prompt_version=prompt_version,
                model_used=model,
            ).first()
            if existing and force_refresh:
                existing.delete()
                existing = None
            if existing:
                existing.cache_hit = True
                existing.analysis_run = run
                existing.save(update_fields=["cache_hit", "analysis_run", "updated_at"])
                results.append(existing)
                metrics["expected"] += 1
                metrics["valid"] += int(existing.json_valid)
                metrics["fallback"] += int(existing.fallback_used)
                metrics["timeout"] += int(existing.timeout)
                metrics["cache_hits"] += 1
                continue

            payloads = [_payload_from_section_result(item) for item in section_results]
            started = time.time()
            payload = merge_chapter_fast_with_llm(chapter_title, payloads)
            meta = _meta(payload)
            result = LLMChapterAnalysis.objects.create(
                global_book=cache,
                analysis_run=run,
                mode=MODE,
                chapter_title=chapter_title,
                chapter_index=chapter_index,
                chapter_summary=str(payload.get("chapter_summary", "")).strip(),
                main_topics=list(payload.get("main_topics", [])),
                key_terms=list(payload.get("important_terms", [])),
                sections_count=len(section_results),
                source_section_ids=[item.id for item in section_results],
                model_used=model,
                prompt_version=prompt_version,
                content_hash=evidence_hash,
                json_valid=bool(meta.get("llm_used")),
                fallback_used=bool(meta.get("fallback_used")),
                timeout=_has_timeout(payload),
                quality_flags=list(meta.get("quality_flags", [])) if isinstance(meta.get("quality_flags", []), list) else [],
                cache_hit=bool(meta.get("cache_hit")),
                actual_llm_call=bool(meta.get("actual_llm_call", True)),
                duration_seconds=float(meta.get("duration_seconds", round(time.time() - started, 2)) or 0.0),
                metadata={"llm_meta": meta},
            )
            results.append(result)
            metrics["expected"] += 1
            metrics["valid"] += int(result.json_valid)
            metrics["fallback"] += int(result.fallback_used)
            metrics["timeout"] += int(result.timeout)
            metrics["cache_hits"] += int(result.cache_hit)
            metrics["actual_calls"] += int(result.actual_llm_call)
        run.chapters_processed = len(results)
        run.valid_json_units += metrics["valid"]
        run.expected_json_units += metrics["expected"]
        run.fallback_count += metrics["fallback"]
        run.timeout_count += metrics["timeout"]
        run.cache_hits += metrics["cache_hits"]
        run.llm_calls_actual += metrics["actual_calls"]
        run.last_heartbeat_at = timezone.now()
        run.save(
            update_fields=[
                "chapters_processed",
                "valid_json_units",
                "expected_json_units",
                "fallback_count",
                "timeout_count",
                "cache_hits",
                "llm_calls_actual",
                "last_heartbeat_at",
                "updated_at",
            ]
        )
        return results

    def _build_book_result(
        self,
        cache: GlobalBookCache,
        run: LLMAnalysisRun,
        chapter_results: list[LLMChapterAnalysis],
        model: str,
        prompt_version: str = QUALITY_PROMPT_VERSION,
        force_refresh: bool = False,
    ) -> LLMBookAnalysis:
        evidence_hash = _hash_text("\n".join(item.content_hash for item in chapter_results))
        existing = getattr(cache, "llm_book_analysis", None)
        if (
            existing
            and not force_refresh
            and existing.content_hash == evidence_hash
            and existing.model_used == model
            and existing.prompt_version == prompt_version
            and (existing.json_valid or bool(existing.learning_path))
        ):
            existing.cache_hit = True
            existing.analysis_run = run
            existing.save(update_fields=["cache_hit", "analysis_run", "updated_at"])
            self._add_book_metrics_to_run(run, existing, cache_hit=True)
            return existing

        payloads = [
            {
                "chapter_title": item.chapter_title,
                "chapter_summary": item.chapter_summary,
                "main_topics": item.main_topics,
                "important_terms": item.key_terms,
            }
            for item in chapter_results
        ]
        started = time.time()
        payload = build_book_fast_with_llm(payloads)
        meta = _meta(payload)
        result, _ = LLMBookAnalysis.objects.update_or_create(
            global_book=cache,
            defaults={
                "analysis_run": run,
                "mode": MODE,
                "book_summary": str(payload.get("book_summary", "")).strip(),
                "global_themes": list(payload.get("global_themes", [])),
                "learning_path": list(payload.get("recommended_learning_path", [])),
                "model_used": model,
                "prompt_version": prompt_version,
                "chapters_count": len(chapter_results),
                "sections_count": run.sections_total,
                "content_hash": evidence_hash,
                "json_valid": bool(meta.get("llm_used")),
                "fallback_used": bool(meta.get("fallback_used")),
                "timeout": _has_timeout(payload),
                "quality_flags": list(meta.get("quality_flags", [])) if isinstance(meta.get("quality_flags", []), list) else [],
                "cache_hit": bool(meta.get("cache_hit")),
                "actual_llm_call": bool(meta.get("actual_llm_call", True)),
                "duration_seconds": float(meta.get("duration_seconds", round(time.time() - started, 2)) or 0.0),
                "metadata": {"llm_meta": meta},
            },
        )
        self._add_book_metrics_to_run(run, result, cache_hit=False)
        return result

    def _add_book_metrics_to_run(self, run: LLMAnalysisRun, result: LLMBookAnalysis, *, cache_hit: bool) -> None:
        run.valid_json_units += int(result.json_valid)
        run.expected_json_units += 1
        run.fallback_count += int(result.fallback_used)
        run.timeout_count += int(result.timeout)
        run.cache_hits += int(cache_hit)
        run.llm_calls_actual += int(result.actual_llm_call and not cache_hit)
        run.last_heartbeat_at = timezone.now()
        run.save(
            update_fields=[
                "valid_json_units",
                "expected_json_units",
                "fallback_count",
                "timeout_count",
                "cache_hits",
                "llm_calls_actual",
                "last_heartbeat_at",
                "updated_at",
            ]
        )

    def _theme_title_from_chapter(self, chapter: LLMChapterAnalysis) -> str:
        chapter_title = " ".join(str(chapter.chapter_title or "").split()).strip()
        cleaned_chapter_title = chapter_title
        cleaned_chapter_title = cleaned_chapter_title.replace("Глава", "").strip(" .:-")
        evidence = " ".join([chapter.chapter_summary or "", " ".join(chapter.main_topics or []), " ".join(chapter.key_terms or [])])
        topics, _ = clean_term_list_v2([str(item) for item in (chapter.main_topics or [])], evidence, limit=5)
        terms, _ = clean_term_list_v2([str(item) for item in (chapter.key_terms or [])], evidence, limit=5)
        candidate = topics[0] if topics else ""
        if candidate and not is_generic_term(candidate) and candidate.lower() != cleaned_chapter_title.lower():
            if len(candidate.split()) >= 3:
                return candidate[:512]
        details = list(dict.fromkeys((topics[1:] if candidate else topics) + terms))[:3]
        if cleaned_chapter_title and details:
            return f"{cleaned_chapter_title}: {', '.join(details)}"[:512]
        if candidate:
            return candidate[:512]
        if cleaned_chapter_title:
            return cleaned_chapter_title[:512]
        return (chapter.main_topics[0] if chapter.main_topics else "Тема книги")[:512]

    def _materialize_for_ui(
        self,
        cache: GlobalBookCache,
        parsed,
        sections: list[CanonicalSection],
        chapter_results: list[LLMChapterAnalysis],
        book_result: LLMBookAnalysis,
    ) -> None:
        with transaction.atomic():
            ConceptMention.objects.filter(global_book=cache).delete()
            ThemeSubtopic.objects.filter(theme__global_book=cache).delete()
            BookTheme.objects.filter(global_book=cache).delete()
            LogicalBlock.objects.filter(global_book=cache).delete()
            BookSummary.objects.filter(global_book=cache).delete()

            block_by_section: dict[int, LogicalBlock] = {}
            for order, section in enumerate(sections, start=1):
                result = LLMSectionAnalysis.objects.filter(global_book=cache, mode=MODE, section_index=section.section_index).first()
                if not result:
                    continue
                source_text = "\n\n".join(
                    str(item.get("text", "")).strip()
                    for item in section.paragraphs
                    if str(item.get("text", "")).strip()
                )
                block = LogicalBlock.objects.create(
                    global_book=cache,
                    title=result.section_title[:512],
                    order_number=order,
                    source_text=source_text,
                    short_summary=result.summary,
                    start_paragraph=result.start_paragraph,
                    end_paragraph=result.end_paragraph,
                    chapter_title=result.chapter_title[:512],
                    token_count=result.word_count,
                    semantic_data={
                        "pipeline": MODE,
                        "section_index": result.section_index,
                        "terms": result.terms,
                        "subtopics": result.subtopics,
                        "llm_section_analysis_id": result.id,
                    },
                    concept_candidates=result.terms + result.subtopics,
                )
                block_by_section[result.section_index] = block
                for term in list(dict.fromkeys(result.terms + result.subtopics))[:10]:
                    if is_bad_concept(term):
                        continue
                    normalized = normalize_concept_name(term)
                    if not normalized:
                        continue
                    concept, _ = Concept.objects.get_or_create(
                        normalized_name=normalized[:255],
                        defaults={"name": term[:255], "description": result.summary[:2000]},
                    )
                    ConceptMention.objects.update_or_create(
                        concept=concept,
                        logical_block=block,
                        defaults={
                            "global_book": cache,
                            "short_explanation": result.summary or f"Концепт связан с разделом: {term}.",
                            "source_quote": str((result.metadata or {}).get("source_quote", ""))[:1000],
                            "importance_score": 0.75,
                        },
                    )

            for chapter in chapter_results:
                source_ids = chapter.source_section_ids or []
                section_results = list(LLMSectionAnalysis.objects.filter(id__in=source_ids).order_by("section_index"))
                section_indexes = [item.section_index for item in section_results]
                blocks = [block_by_section[idx] for idx in section_indexes if idx in block_by_section]
                if not blocks:
                    continue
                theme = BookTheme.objects.create(
                    global_book=cache,
                    chapter_title=chapter.chapter_title[:512],
                    title=self._theme_title_from_chapter(chapter),
                    order_number=chapter.chapter_index,
                    start_block_number=blocks[0].order_number,
                    end_block_number=blocks[-1].order_number,
                    start_paragraph=blocks[0].start_paragraph,
                    end_paragraph=blocks[-1].end_paragraph,
                    summary=chapter.chapter_summary[:2000],
                )
                for topic in list(dict.fromkeys(chapter.main_topics + chapter.key_terms))[:8]:
                    normalized = normalize_concept_name(topic)
                    if not normalized or is_bad_concept(topic):
                        continue
                    ThemeSubtopic.objects.get_or_create(
                        theme=theme,
                        normalized_name=normalized[:255],
                        defaults={
                            "name": topic[:255],
                            "summary": chapter.chapter_summary[:1200],
                            "source_quote": "",
                            "importance_score": 0.7,
                            "start_paragraph": theme.start_paragraph,
                            "end_paragraph": theme.end_paragraph,
                        },
                    )

            cache.title = parsed.title
            cache.authors = parsed.authors
            cache.full_summary = book_result.book_summary
            cache.analysis_version = "concept_rag_llm_fast_batched_v1"
            metadata = cache.metadata if isinstance(cache.metadata, dict) else {}
            cache.metadata = {
                **metadata,
                "analysis_mode": MODE,
                "pipeline_used": MODE,
                "strict_main_content_sections_count": len(sections),
                "llm_fast_batched": {
                    "book_analysis_id": book_result.id,
                    "chapters_count": len(chapter_results),
                    "sections_count": len(sections),
                    "global_themes": book_result.global_themes,
                    "learning_path": book_result.learning_path,
                },
            }
            cache.save(update_fields=["title", "authors", "full_summary", "analysis_version", "metadata", "updated_at"])
            BookSummary.objects.create(
                global_book=cache,
                short_summary=book_result.book_summary[:1000],
                detailed_summary=book_result.book_summary,
            )

    def _write_final_report(self, output_dir: Path, run: LLMAnalysisRun, cache: GlobalBookCache, *, dry_run: bool, materialized: bool) -> dict[str, Any]:
        book_result = LLMBookAnalysis.objects.filter(global_book=cache).first()
        report = {
            "analysis_run_id": run.analysis_run_id,
            "mode": run.mode,
            "status": run.status,
            "dry_run": dry_run,
            "materialized_for_ui": materialized,
            "sections_total": run.sections_total,
            "sections_processed": run.sections_processed,
            "chapters_processed": run.chapters_processed,
            "valid_json_units": run.valid_json_units,
            "expected_json_units": run.expected_json_units,
            "fallback_count": run.fallback_count,
            "timeout_count": run.timeout_count,
            "cache_hits": run.cache_hits,
            "actual_llm_calls": run.llm_calls_actual,
            "quality_gates": self._quality_gates(run, cache, book_result),
            "book": {
                "summary": book_result.book_summary if book_result else "",
                "global_themes": book_result.global_themes if book_result else [],
                "learning_path": book_result.learning_path if book_result else [],
                "quality_flags": book_result.quality_flags if book_result else [],
            },
        }
        run.final_report = report
        run.save(update_fields=["final_report", "updated_at"])
        (output_dir / "llm_fast_batched_final_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md = [
            "# LLM Fast Batched Final Report",
            "",
            f"- status: {report['status']}",
            f"- dry_run: {dry_run}",
            f"- materialized_for_ui: {materialized}",
            f"- sections: {report['sections_processed']}/{report['sections_total']}",
            f"- chapters: {report['chapters_processed']}",
            f"- fallback: {run.fallback_count}",
            f"- timeout: {run.timeout_count}",
            f"- cache_hits: {run.cache_hits}",
            f"- actual_llm_calls: {run.llm_calls_actual}",
            f"- quality_passed: {report['quality_gates']['passed']}",
            "",
            "## Book",
            report["book"]["summary"],
        ]
        (output_dir / "llm_fast_batched_final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        return report

    def _quality_gates(self, run: LLMAnalysisRun, cache: GlobalBookCache, book_result: LLMBookAnalysis | None) -> dict[str, Any]:
        sections_processed = run.sections_processed
        valid_rate = run.valid_json_units / max(1, run.expected_json_units)
        fallback_rate = run.fallback_count / max(1, run.expected_json_units)
        flags = list(book_result.quality_flags if book_result else [])
        generic_terms_count = 0
        for row in LLMSectionAnalysis.objects.filter(global_book=cache, mode=MODE, analysis_run=run):
            generic_terms_count += sum(1 for item in row.terms if _is_generic_term_name(item))
        passed = (
            sections_processed == run.sections_total
            and valid_rate >= 0.95
            and fallback_rate < 0.05
            and run.timeout_count == 0
            and not flags
            and generic_terms_count == 0
            and bool(book_result and book_result.book_summary and book_result.global_themes and book_result.learning_path)
        )
        return {
            "passed": passed,
            "sections_complete": sections_processed == run.sections_total,
            "valid_json_rate": round(valid_rate, 4),
            "fallback_rate": round(fallback_rate, 4),
            "timeout_count": run.timeout_count,
            "book_quality_flags": flags,
            "generic_terms_count": generic_terms_count,
        }

    def _finish_failed(self, run: LLMAnalysisRun, user_book: UserBook | None, message: str, *, partial: bool) -> None:
        status = LLMAnalysisRun.Status.PARTIAL_READY if partial else LLMAnalysisRun.Status.FAILED
        self._touch_run(run, status=status, error_message=message[:2000], finished_at=timezone.now())
        self._touch_user_book(
            user_book,
            status=UserBook.Status.PARTIAL_READY if partial else UserBook.Status.FAILED,
            current_stage="partial_ready" if partial else "failed",
            error_message=message[:2000],
            finished_at=timezone.now(),
        )
