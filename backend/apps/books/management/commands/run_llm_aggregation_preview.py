from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.llm_service import (
    analyze_section_with_llm,
    build_book_analysis_with_llm,
    ensure_llm_ready,
    get_llm_runtime_config,
    merge_chapter_analyses_with_llm,
)
from apps.books.services.logical_block_splitter import split_into_logical_blocks_improved
from apps.books.services.structure_detector import build_canonical_outline


def _project_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() == "backend" else cwd


def _clean_section_text(blocks, section_title: str, max_chars: int) -> str:
    parts: list[str] = []
    for block in blocks:
        if (block.section_title or block.chapter_title) != section_title:
            continue
        text = (block.clean_text_for_analysis or block.source_text or "").strip()
        if text:
            parts.append(text)
    value = "\n\n".join(parts).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rsplit(" ", 1)[0]


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("_meta", {})
    return raw if isinstance(raw, dict) else {}


def _is_timeout_problem(meta: dict[str, Any], valid_json: bool, duration_seconds: float) -> bool:
    """A slow successful JSON response is not a timeout failure."""
    failure = str(meta.get("llm_failure", "")).lower()
    if "timeout" in failure or "timed out" in failure:
        return True
    return duration_seconds >= 59 and (not valid_json or bool(meta.get("fallback_used")))


class Command(BaseCommand):
    help = "Run mini preview for chapter/book LLM aggregation only; no full analysis."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True)
        parser.add_argument("--limit-main-sections", type=int, default=3)
        parser.add_argument("--max-input-chars", type=int, default=1600)
        parser.add_argument("--output", default="llm_aggregation_preview_report")

    def handle(self, *args, **options):
        file_path = Path(str(options["file"])).expanduser().resolve()
        if not file_path.exists():
            raise CommandError(f"File not found: {file_path}")

        root = _project_root()
        output_base = str(options["output"]).strip() or "llm_aggregation_preview_report"
        json_path = root / f"{output_base}.json"
        md_path = root / f"{output_base}.md"

        cfg = get_llm_runtime_config()
        llm_state = ensure_llm_ready(require_enabled=True)
        report: dict[str, Any] = {
            "status": "failed_precheck",
            "book_file": str(file_path),
            "llm": {
                "ready": bool(llm_state.get("ok")),
                "model_used": llm_state.get("selected_fast"),
                "fallback_model": llm_state.get("selected_fallback"),
                "available_models": llm_state.get("models", []),
                "timeout_seconds": cfg.get("timeout_seconds"),
                "error": llm_state.get("error", ""),
            },
            "sections": [],
            "chapter": {},
            "book": {},
            "metrics": {},
            "can_run_limited_llm_full": False,
        }
        if not llm_state.get("ok"):
            self._write_reports(json_path, md_path, report)
            self.stdout.write(self.style.WARNING("LLM provider is not ready."))
            return

        parsed = parse_uploaded_book(file_path.read_bytes(), file_path.name)
        outline = build_canonical_outline(parsed)
        main_sections = [
            item
            for item in list(outline.get("sections", []))
            if item.content_type == "main_content" and item.is_main_content
        ]
        selected = main_sections[: max(2, min(3, int(options["limit_main_sections"])))]
        selected_indexes = {item.section_index - 1 for item in selected}
        mini_parsed = SimpleNamespace(
            title=parsed.title,
            authors=parsed.authors,
            metadata=parsed.metadata,
            chapters=[parsed.chapters[idx] for idx in sorted(selected_indexes)],
        )
        blocks, _splitter_diag = split_into_logical_blocks_improved(mini_parsed)

        env_overrides = {
            "OLLAMA_MODEL_FAST": str(llm_state.get("selected_fast") or ""),
            "OLLAMA_MODEL_HIGH": str(llm_state.get("selected_fast") or ""),
            "OLLAMA_MODEL_FALLBACK": str(llm_state.get("selected_fallback") or llm_state.get("selected_fast") or ""),
            "OLLAMA_TIMEOUT_SECONDS": "60",
            "OLLAMA_MAX_TOKENS_JSON": "160",
            "SECTION_LLM_MAX_INPUT_CHARS": str(int(options["max_input_chars"])),
            "LLM_MAX_RETRIES": "0",
        }
        previous = {key: os.environ.get(key) for key in env_overrides}
        started = time.time()

        section_payloads: list[dict[str, Any]] = []
        section_reports: list[dict[str, Any]] = []
        try:
            os.environ.update(env_overrides)
            for section in selected:
                payload = analyze_section_with_llm(
                    section_title=section.section_title,
                    section_text=_clean_section_text(blocks, section.section_title, int(options["max_input_chars"])),
                    chapter_title=section.parent_chapter_title or section.chapter_title,
                    section_type=section.content_type,
                )
                section_payloads.append(payload)
                terms = [
                    str(item.get("term", "")).strip()
                    for item in payload.get("key_terms", [])
                    if isinstance(item, dict)
                ]
                section_reports.append(
                    {
                        "section_title": section.section_title,
                        "json_valid": bool(_meta(payload).get("llm_used")),
                        "fallback_used": bool(_meta(payload).get("fallback_used")),
                        "summary": str(payload.get("summary", "")).strip(),
                        "key_terms": terms[:10],
                    }
                )

            chapter_started = time.time()
            chapter_payload = merge_chapter_analyses_with_llm("Preview chapter", section_payloads)
            chapter_duration = round(time.time() - chapter_started, 2)

            book_started = time.time()
            book_payload = build_book_analysis_with_llm([chapter_payload])
            book_duration = round(time.time() - book_started, 2)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        chapter_meta = _meta(chapter_payload)
        book_meta = _meta(book_payload)
        chapter_valid = bool(chapter_meta.get("llm_used"))
        book_valid = bool(book_meta.get("llm_used"))
        fallback_count = int(bool(chapter_meta.get("fallback_used"))) + int(bool(book_meta.get("fallback_used")))
        chapter_timeout = _is_timeout_problem(chapter_meta, chapter_valid, chapter_duration)
        book_timeout = _is_timeout_problem(book_meta, book_valid, book_duration)
        timeout_count = int(chapter_timeout) + int(book_timeout)
        can_full = chapter_valid and book_valid and fallback_count == 0 and timeout_count == 0

        report.update(
            {
                "status": "completed",
                "book_title": parsed.title,
                "sections": section_reports,
                "chapter": {
                    "valid_json": chapter_valid,
                    "fallback_used": bool(chapter_meta.get("fallback_used")),
                    "timeout": chapter_timeout,
                    "duration_seconds": chapter_duration,
                    "chapter_summary": str(chapter_payload.get("chapter_summary", "")).strip(),
                    "main_topics": chapter_payload.get("main_topics", []),
                    "key_terms": chapter_payload.get("important_terms", []),
                    "failure": str(chapter_meta.get("llm_failure", "")).strip(),
                },
                "book": {
                    "valid_json": book_valid,
                    "fallback_used": bool(book_meta.get("fallback_used")),
                    "timeout": book_timeout,
                    "duration_seconds": book_duration,
                    "book_summary": str(book_payload.get("book_summary", "")).strip(),
                    "global_themes": book_payload.get("global_themes", []),
                    "learning_path": book_payload.get("recommended_learning_path", []),
                    "failure": str(book_meta.get("llm_failure", "")).strip(),
                },
                "metrics": {
                    "section_calls": len(selected) * 3,
                    "aggregation_calls": 4,
                    "total_llm_json_calls": len(selected) * 3 + 4,
                    "chapter_valid_json": chapter_valid,
                    "book_valid_json": book_valid,
                    "fallback_used_count": fallback_count,
                    "timeout_count": timeout_count,
                    "duration_seconds": round(time.time() - started, 2),
                },
                "can_run_limited_llm_full": can_full,
            }
        )
        self._write_reports(json_path, md_path, report)
        self.stdout.write(self.style.SUCCESS("LLM aggregation preview completed."))
        self.stdout.write(f"JSON report: {json_path}")
        self.stdout.write(f"Markdown report: {md_path}")
        self.stdout.write(
            f"chapter_json={chapter_valid}, book_json={book_valid}, fallback={fallback_count}, "
            f"timeouts={timeout_count}, can_limited_full={can_full}"
        )

    def _write_reports(self, json_path: Path, md_path: Path, report: dict[str, Any]) -> None:
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown(report), encoding="utf-8")

    def _markdown(self, report: dict[str, Any]) -> str:
        lines = ["# LLM Aggregation Preview Report", ""]
        llm = report.get("llm", {})
        lines.append(f"- Status: **{report.get('status')}**")
        lines.append(f"- Ollama ready: **{llm.get('ready')}**")
        lines.append(f"- Model used: `{llm.get('model_used')}`")
        lines.append(f"- can_run_limited_llm_full: **{report.get('can_run_limited_llm_full', False)}**")
        lines.append("")
        metrics = report.get("metrics", {})
        lines.append("## Metrics")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        chapter = report.get("chapter", {})
        lines.append("## Chapter Aggregation")
        lines.append(f"- valid_json: {chapter.get('valid_json')}")
        lines.append(f"- fallback: {chapter.get('fallback_used')}")
        lines.append(f"- timeout: {chapter.get('timeout')}")
        lines.append(f"- summary: {str(chapter.get('chapter_summary', ''))[:600]}")
        lines.append(f"- main_topics: {', '.join(chapter.get('main_topics', [])[:10]) or '-'}")
        lines.append(f"- key_terms: {', '.join(chapter.get('key_terms', [])[:12]) or '-'}")
        lines.append("")
        book = report.get("book", {})
        lines.append("## Book Aggregation")
        lines.append(f"- valid_json: {book.get('valid_json')}")
        lines.append(f"- fallback: {book.get('fallback_used')}")
        lines.append(f"- timeout: {book.get('timeout')}")
        lines.append(f"- summary: {str(book.get('book_summary', ''))[:600]}")
        lines.append(f"- global_themes: {', '.join(book.get('global_themes', [])[:10]) or '-'}")
        lines.append(f"- learning_path: {', '.join(book.get('learning_path', [])[:10]) or '-'}")
        return "\n".join(lines).strip() + "\n"
