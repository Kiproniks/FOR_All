from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.llm_service import (
    _is_generic_term_name,
    analyze_section_fast_with_llm,
    build_book_fast_with_llm,
    ensure_llm_ready,
    get_llm_runtime_config,
    merge_chapter_fast_with_llm,
    prepare_section_llm_input,
)
from apps.books.services.structure_detector import CanonicalSection, build_canonical_outline


def _project_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() == "backend" else cwd


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("_meta", {})
    return raw if isinstance(raw, dict) else {}


def _terms_from_section_payload(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for item in payload.get("key_terms", []):
        if isinstance(item, dict):
            value = str(item.get("term", "")).strip()
            if value:
                result.append(value)
    return result


def _subtopics_from_section_payload(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for item in payload.get("subtopics", []):
        if isinstance(item, dict):
            value = str(item.get("title", "")).strip()
            if value:
                result.append(value)
    return result


def _has_timeout(payload: dict[str, Any]) -> bool:
    failure = str(_meta(payload).get("llm_failure", "")).lower()
    return "timeout" in failure or "timed out" in failure


def _select_limited_sections(outline: dict[str, Any], limit: int, offset: int = 0) -> list[CanonicalSection]:
    main_sections = [
        item
        for item in list(outline.get("sections", []))
        if item.content_type == "main_content" and item.is_main_content
    ]
    if not main_sections:
        return []
    offset = max(0, int(offset))
    if offset:
        return main_sections[offset : offset + limit]

    canonical_chapters = list(outline.get("canonical_chapters", []))
    if canonical_chapters:
        first_chapter = canonical_chapters[0]
        indexes = {
            int(item.get("section_index", -1))
            for item in first_chapter.get("sections", [])
            if str(item.get("section_index", "")).isdigit()
        }
        chapter_sections = [item for item in main_sections if item.section_index in indexes]
        if chapter_sections:
            selected = chapter_sections[:limit]
            # Some parsed books expose the chapter heading as a separate section.
            # In that case a "one chapter" batch would contain only one small
            # item, so extend with following main-content sections up to limit.
            if len(selected) < min(5, limit):
                selected_ids = {item.section_index for item in selected}
                for section in main_sections:
                    if section.section_index in selected_ids:
                        continue
                    selected.append(section)
                    selected_ids.add(section.section_index)
                    if len(selected) >= limit:
                        break
            return selected[:limit]

    return main_sections[:limit]


def _strict_main_content_count(outline: dict[str, Any]) -> int:
    return len(
        [
            item
            for item in list(outline.get("sections", []))
            if item.content_type == "main_content" and item.is_main_content
        ]
    )


class Command(BaseCommand):
    help = "Run fast limited LLM full pipeline on N main-content sections; no DB writes."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True)
        parser.add_argument("--sections", type=int, default=None, help="Alias for --limit-main-sections")
        parser.add_argument("--limit-main-sections", type=int, default=5)
        parser.add_argument("--offset-main-sections", type=int, default=0)
        parser.add_argument("--mode", choices=["fast"], default="fast")
        parser.add_argument("--max-input-chars", type=int, default=1200)
        parser.add_argument("--max-chunks-per-section", type=int, default=1)
        parser.add_argument("--batch-timeout-seconds", type=int, default=None)
        parser.add_argument("--output", default="limited_llm_full_report")

    def handle(self, *args, **options):
        file_path = Path(str(options["file"])).expanduser().resolve()
        if not file_path.exists():
            raise CommandError(f"File not found: {file_path}")

        root = _project_root()
        output_base = str(options["output"]).strip() or "limited_llm_full_report"
        json_path = root / f"{output_base}.json"
        md_path = root / f"{output_base}.md"

        runtime = get_llm_runtime_config()
        llm_state = ensure_llm_ready(require_enabled=True)
        report: dict[str, Any] = {
            "status": "failed_precheck",
            "book_file": str(file_path),
            "llm": {
                "ready": bool(llm_state.get("ok")),
                "provider": runtime.get("provider"),
                "model_used": llm_state.get("selected_fast"),
                "fallback_model": llm_state.get("selected_fallback"),
                "available_models": llm_state.get("models", []),
                "per_call_timeout_seconds": 60,
                "fallback_enabled": True,
                "error": llm_state.get("error", ""),
            },
            "selected_sections": [],
            "sections": [],
            "chapters": [],
            "book": {},
            "metrics": {},
            "quality": {},
        }
        if not llm_state.get("ok"):
            self._write_reports(json_path, md_path, report)
            self.stdout.write(self.style.WARNING("LLM is not ready; limited full was not run."))
            return

        parsed = parse_uploaded_book(file_path.read_bytes(), file_path.name)
        outline = build_canonical_outline(parsed)
        strict_main_content_count = _strict_main_content_count(outline)
        requested_sections = options["sections"] if options["sections"] is not None else options["limit_main_sections"]
        limit = max(1, min(10, int(requested_sections)))
        offset = max(0, int(options["offset_main_sections"]))
        selected_sections = _select_limited_sections(outline, limit, offset)
        if not selected_sections:
            raise CommandError("No main_content sections detected.")
        batch_timeout = options["batch_timeout_seconds"]
        if batch_timeout is None:
            batch_timeout = 300 if limit <= 5 else 600
        batch_timeout = max(60, int(batch_timeout))

        env_overrides = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": str(llm_state.get("selected_fast") or "qwen2.5:1.5b"),
            "OLLAMA_MODEL_FAST": str(llm_state.get("selected_fast") or "qwen2.5:1.5b"),
            "OLLAMA_MODEL_HIGH": str(llm_state.get("selected_fast") or "qwen2.5:1.5b"),
            "OLLAMA_MODEL_FALLBACK": str(llm_state.get("selected_fallback") or llm_state.get("selected_fast") or "qwen2.5:1.5b"),
            "OLLAMA_TIMEOUT_SECONDS": "60",
            "OLLAMA_MAX_TOKENS_JSON": "220",
            "LLM_MAX_RETRIES": "0",
            "LLM_ENABLE_FALLBACK": "true",
            "LLM_MAX_INPUT_CHARS": str(int(options["max_input_chars"])),
            "SECTION_LLM_MAX_INPUT_CHARS": str(int(options["max_input_chars"])),
            "LLM_MAX_CHUNKS_PER_SECTION": str(max(1, min(2, int(options["max_chunks_per_section"])))),
            "LLM_MAX_CALLS_PER_CHAPTER": "12",
            "LLM_MAX_CALLS_PER_BOOK": "60",
        }
        previous = {key: os.environ.get(key) for key in env_overrides}
        started = time.time()
        batch_timeout_exceeded = False
        stopped_at_section: int | None = None
        section_payloads: list[dict[str, Any]] = []
        processed_sections: list[CanonicalSection] = []
        chapter_payload: dict[str, Any] | None = None
        book_payload: dict[str, Any] | None = None
        try:
            os.environ.update(env_overrides)
            for section in selected_sections:
                if time.time() - started >= batch_timeout:
                    batch_timeout_exceeded = True
                    stopped_at_section = section.section_index
                    break
                prepared_text = prepare_section_llm_input(section, max_chars=int(options["max_input_chars"]))
                section_started = time.time()
                section_payloads.append(
                    analyze_section_fast_with_llm(
                        section_title=section.section_title,
                        section_text=prepared_text,
                        chapter_title=section.parent_chapter_title or section.chapter_title,
                        section_type=section.content_type,
                    )
                )
                section_payloads[-1].setdefault("_meta", {})["section_duration_seconds"] = round(time.time() - section_started, 2)
                section_payloads[-1].setdefault("_meta", {})["input_chars"] = len(prepared_text)
                processed_sections.append(section)

            if not batch_timeout_exceeded and section_payloads and time.time() - started < batch_timeout:
                chapter_title = processed_sections[0].parent_chapter_title or processed_sections[0].chapter_title
                chapter_payload = merge_chapter_fast_with_llm(chapter_title, section_payloads)

            if chapter_payload and not batch_timeout_exceeded and time.time() - started < batch_timeout:
                book_payload = build_book_fast_with_llm([chapter_payload])
            elif not book_payload and time.time() - started >= batch_timeout:
                batch_timeout_exceeded = True
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        section_reports: list[dict[str, Any]] = []
        generic_terms: set[str] = set()
        valid_json_count = 0
        fallback_count = 0
        timeout_count = 0
        actual_llm_calls = 0
        cache_hits = 0
        failed_sections: list[dict[str, Any]] = []
        slowest_sections: list[dict[str, Any]] = []
        for section, payload in zip(processed_sections, section_payloads, strict=False):
            meta = _meta(payload)
            if meta.get("actual_llm_call"):
                actual_llm_calls += 1
            if meta.get("cache_hit"):
                cache_hits += 1
            terms = _terms_from_section_payload(payload)
            generics = [term for term in terms if _is_generic_term_name(term)]
            generic_terms.update(generics)
            if meta.get("llm_used"):
                valid_json_count += 1
            if meta.get("fallback_used"):
                fallback_count += 1
            if _has_timeout(payload):
                timeout_count += 1
            section_report = {
                "section_index": section.section_index,
                "section_title": section.section_title,
                "chapter_title": section.parent_chapter_title or section.chapter_title,
                "json_valid": bool(meta.get("llm_used")),
                "fallback_used": bool(meta.get("fallback_used")),
                "timeout": _has_timeout(payload),
                "summary": str(payload.get("summary", "")).strip(),
                "key_terms": terms,
                "generic_terms": generics,
                "subtopics": _subtopics_from_section_payload(payload),
                "quality_flags": payload.get("quality_flags", []),
                "llm_failure": str(meta.get("llm_failure", "")).strip(),
                "cache_hit": bool(meta.get("cache_hit")),
                "actual_llm_call": bool(meta.get("actual_llm_call")),
                "duration_seconds": float(meta.get("section_duration_seconds", meta.get("duration_seconds", 0.0)) or 0.0),
                "input_chars": int(meta.get("input_chars", 0) or 0),
            }
            section_reports.append(section_report)
            slowest_sections.append(
                {
                    "section_index": section_report["section_index"],
                    "section_title": section_report["section_title"],
                    "duration_seconds": section_report["duration_seconds"],
                }
            )
            if section_report["fallback_used"] or section_report["timeout"] or section_report["llm_failure"]:
                failed_sections.append(section_report)

        chapter_reports: list[dict[str, Any]] = []
        for payload in ([chapter_payload] if chapter_payload else []):
            meta = _meta(payload)
            if meta.get("actual_llm_call"):
                actual_llm_calls += 1
            if meta.get("cache_hit"):
                cache_hits += 1
            terms = [str(item).strip() for item in payload.get("important_terms", []) if str(item).strip()]
            generics = [term for term in terms if _is_generic_term_name(term)]
            generic_terms.update(generics)
            if meta.get("llm_used"):
                valid_json_count += 1
            if meta.get("fallback_used"):
                fallback_count += 1
            if _has_timeout(payload):
                timeout_count += 1
            chapter_reports.append(
                {
                    "chapter_title": str(payload.get("chapter_title", "")).strip(),
                    "json_valid": bool(meta.get("llm_used")),
                    "fallback_used": bool(meta.get("fallback_used")),
                    "timeout": _has_timeout(payload),
                    "chapter_summary": str(payload.get("chapter_summary", "")).strip(),
                    "main_topics": payload.get("main_topics", []),
                    "key_terms": terms,
                    "generic_terms": generics,
                    "llm_failure": str(meta.get("llm_failure", "")).strip(),
                }
            )

        book_payload = book_payload or {}
        book_meta = _meta(book_payload)
        if book_payload:
            if book_meta.get("actual_llm_call"):
                actual_llm_calls += 1
            if book_meta.get("cache_hit"):
                cache_hits += 1
            if book_meta.get("llm_used"):
                valid_json_count += 1
            if book_meta.get("fallback_used"):
                fallback_count += 1
            if _has_timeout(book_payload):
                timeout_count += 1
        if batch_timeout_exceeded:
            timeout_count += 1

        expected_json_units = len(section_reports) + len(chapter_reports) + (1 if book_payload else 0)
        fallback_is_problem = fallback_count > 0
        clean_result = expected_json_units > 0 and valid_json_count == expected_json_units and fallback_count == 0 and timeout_count == 0
        book_quality_flags = list(book_meta.get("quality_flags", [])) if isinstance(book_meta.get("quality_flags", []), list) else []
        validator_cleanup_flags = list(book_meta.get("validator_cleanup_flags", [])) if isinstance(book_meta.get("validator_cleanup_flags", []), list) else []
        hallucination_flags = [flag for flag in book_quality_flags if flag in {"hallucinated_topic", "irrelevant_learning_path", "weak_grounding", "generic_book_theme"}]
        can_expand = clean_result and len(section_reports) >= 5 and not hallucination_flags

        report.update(
            {
                "status": "failed_timeout" if batch_timeout_exceeded else "completed",
                "book_title": parsed.title,
                "mode": options["mode"],
                "sections_requested": limit,
                "offset_main_sections": offset,
                "total_main_content_sections_detected": strict_main_content_count,
                "outline_main_sections_count": int(outline.get("main_sections_count", 0)),
                "selected_sections": [
                    {
                        "section_index": item.section_index,
                        "section_title": item.section_title,
                        "chapter_title": item.parent_chapter_title or item.chapter_title,
                        "word_count": item.word_count,
                        "paragraph_range": [item.start_paragraph, item.end_paragraph],
                    }
                    for item in processed_sections
                ],
                "sections": section_reports,
                "chapters": chapter_reports,
                "book": {
                    "json_valid": bool(book_meta.get("llm_used")),
                    "fallback_used": bool(book_meta.get("fallback_used")),
                    "timeout": _has_timeout(book_payload) if book_payload else False,
                    "book_summary": str(book_payload.get("book_summary", "")).strip(),
                    "global_themes": book_payload.get("global_themes", []),
                    "learning_path": book_payload.get("recommended_learning_path", []),
                    "llm_failure": str(book_meta.get("llm_failure", "")).strip(),
                    "quality_flags": book_quality_flags,
                    "validator_cleanup_flags": validator_cleanup_flags,
                },
                "metrics": {
                    "sections_requested": limit,
                    "offset_main_sections": offset,
                    "sections_processed": len(section_reports),
                    "total_main_content_sections_detected": strict_main_content_count,
                    "outline_main_sections_count": int(outline.get("main_sections_count", 0)),
                    "chapters_processed": len(chapter_reports),
                    "llm_calls_expected": limit + 2,
                    "llm_calls_actual": actual_llm_calls,
                    "cache_hits": cache_hits,
                    "llm_calls_pipeline_units": len(section_reports) + len(chapter_reports) + (1 if book_payload else 0),
                    "estimated_json_calls": len(section_reports) + len(chapter_reports) + (1 if book_payload else 0),
                    "valid_json_units": valid_json_count,
                    "expected_json_units": expected_json_units,
                    "fallback_used_count": fallback_count,
                    "timeout_count": timeout_count,
                    "batch_timeout_exceeded": batch_timeout_exceeded,
                    "batch_timeout_seconds": batch_timeout,
                    "stopped_at_section": stopped_at_section,
                    "duration_seconds": round(time.time() - started, 2),
                    "slowest_sections": sorted(slowest_sections, key=lambda item: item["duration_seconds"], reverse=True)[:5],
                    "failed_sections": [
                        {
                            "section_index": item["section_index"],
                            "section_title": item["section_title"],
                            "llm_failure": item["llm_failure"],
                            "fallback_used": item["fallback_used"],
                            "timeout": item["timeout"],
                        }
                        for item in failed_sections
                    ],
                },
                "quality": {
                    "fallback_is_problem": fallback_is_problem,
                    "generic_terms_found": sorted(generic_terms),
                    "generic_terms_count": len(generic_terms),
                    "hallucination_flags": hallucination_flags,
                    "validator_cleanup_flags": validator_cleanup_flags,
                    "irrelevant_learning_path": "irrelevant_learning_path" in hallucination_flags,
                    "can_expand_next": can_expand,
                    "can_expand_to_10_sections": can_expand and limit >= 5,
                    "can_expand_to_2_3_chapters": can_expand,
                    "recommendation": (
                        "Можно проверить 10 секций." if can_expand and limit == 5 else
                        "Можно оценивать запуск по главам." if can_expand and limit >= 10 else
                        "Нельзя расширять: есть timeout/fallback/hallucination или батч слишком мал."
                    ),
                    "notes": (
                        "Limited batch is clean; safe to try 2-3 chapters."
                        if can_expand
                        else (
                            "Clean result, but batch is too small to prove scaling."
                            if clean_result
                            else "Do not expand yet: fallback/timeout/generic term problems remain."
                        )
                    ),
                },
            }
        )
        self._write_reports(json_path, md_path, report)
        self.stdout.write(self.style.SUCCESS("Limited LLM full completed."))
        self.stdout.write(f"JSON report: {json_path}")
        self.stdout.write(f"Markdown report: {md_path}")
        self.stdout.write(
            f"sections={len(section_reports)}, valid_units={valid_json_count}/{expected_json_units}, "
            f"fallback={fallback_count}, timeouts={timeout_count}, cache_hits={cache_hits}, "
            f"actual_calls={actual_llm_calls}, can_expand={can_expand}"
        )

    def _write_reports(self, json_path: Path, md_path: Path, report: dict[str, Any]) -> None:
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown(report), encoding="utf-8")

    def _markdown(self, report: dict[str, Any]) -> str:
        lines: list[str] = ["# Limited LLM Full Report", ""]
        lines.append(f"- Status: **{report.get('status')}**")
        lines.append(f"- Book: **{report.get('book_title', '')}**")
        llm = report.get("llm", {})
        lines.append(f"- Model: `{llm.get('model_used')}`")
        lines.append(f"- Fallback enabled: **{llm.get('fallback_enabled')}**")
        lines.append("")

        metrics = report.get("metrics", {})
        lines.append("## Metrics")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

        lines.append("## Sections")
        for item in report.get("sections", []):
            lines.append(f"### {item.get('section_title')}")
            lines.append(f"- JSON valid: {item.get('json_valid')}, fallback: {item.get('fallback_used')}, timeout: {item.get('timeout')}")
            lines.append(f"- Summary: {str(item.get('summary', ''))[:500]}")
            lines.append(f"- Key terms: {', '.join(item.get('key_terms', [])[:12]) or '-'}")
            lines.append(f"- Subtopics: {', '.join(item.get('subtopics', [])[:10]) or '-'}")
            lines.append(f"- Generic terms: {', '.join(item.get('generic_terms', [])) or '-'}")
            if item.get("llm_failure"):
                lines.append(f"- Failure: `{item.get('llm_failure')}`")
            lines.append("")

        lines.append("## Chapter Aggregation")
        for item in report.get("chapters", []):
            lines.append(f"### {item.get('chapter_title')}")
            lines.append(f"- JSON valid: {item.get('json_valid')}, fallback: {item.get('fallback_used')}, timeout: {item.get('timeout')}")
            lines.append(f"- Summary: {str(item.get('chapter_summary', ''))[:500]}")
            lines.append(f"- Main topics: {', '.join(item.get('main_topics', [])[:10]) or '-'}")
            lines.append(f"- Key terms: {', '.join(item.get('key_terms', [])[:12]) or '-'}")
            lines.append("")

        book = report.get("book", {})
        lines.append("## Book Aggregation")
        lines.append(f"- JSON valid: {book.get('json_valid')}, fallback: {book.get('fallback_used')}, timeout: {book.get('timeout')}")
        lines.append(f"- Summary: {str(book.get('book_summary', ''))[:700]}")
        lines.append(f"- Global themes: {', '.join(book.get('global_themes', [])[:10]) or '-'}")
        lines.append(f"- Learning path: {', '.join(book.get('learning_path', [])[:10]) or '-'}")
        lines.append("")

        quality = report.get("quality", {})
        lines.append("## Quality")
        lines.append(f"- fallback_is_problem: **{quality.get('fallback_is_problem')}**")
        lines.append(f"- generic_terms_found: {', '.join(quality.get('generic_terms_found', [])) or '-'}")
        lines.append(f"- can_expand_to_2_3_chapters: **{quality.get('can_expand_to_2_3_chapters')}**")
        lines.append(f"- notes: {quality.get('notes', '')}")
        return "\n".join(lines).strip() + "\n"
