from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.llm_hierarchical_pipeline import run_hierarchical_llm_pipeline
from apps.books.services.llm_service import ensure_llm_ready, get_llm_runtime_config
from apps.books.services.logical_block_splitter import split_into_logical_blocks_improved
from apps.books.services.structure_detector import CanonicalSection, build_canonical_outline

GENERIC_TERMS = {
    "глава",
    "раздел",
    "материал",
    "текст",
    "книга",
    "тема",
    "section",
    "chapter",
    "material",
    "text",
}


def _output_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() == "backend" else cwd


def _section_payload_summary(item) -> dict[str, Any]:
    payload = item.payload
    terms = [str(term.get("term", "")).strip() for term in payload.get("key_terms", []) if isinstance(term, dict)]
    subtopics = [str(sub.get("title", "")).strip() for sub in payload.get("subtopics", []) if isinstance(sub, dict)]
    quotes = [str(term.get("source_quote", "")).strip() for term in payload.get("key_terms", []) if isinstance(term, dict)]
    quotes += [str(sub.get("source_quote", "")).strip() for sub in payload.get("subtopics", []) if isinstance(sub, dict)]
    quotes = [item for item in quotes if item]
    generic_terms = [term for term in terms if term.lower() in GENERIC_TERMS]
    meta = payload.get("_meta", {}) if isinstance(payload.get("_meta", {}), dict) else {}
    return {
        "section_index": item.section.section_index,
        "section_title": item.section.section_title,
        "chapter_title": item.section.parent_chapter_title or item.section.chapter_title,
        "summary": str(payload.get("summary", "")).strip(),
        "key_terms": terms,
        "subtopics": subtopics,
        "quotes_count": len(quotes),
        "generic_terms": generic_terms,
        "json_valid": bool(meta.get("llm_used", False)),
        "fallback_used": bool(meta.get("fallback_used", False)),
        "llm_failure": str(meta.get("llm_failure", "")).strip(),
        "quality_flags": payload.get("quality_flags", []),
    }


def _preview_quality_score(section_items: list[dict[str, Any]], metrics: dict[str, Any]) -> float:
    if not section_items:
        return 0.0

    score = 1.0
    fallback_ratio = sum(1 for item in section_items if item["fallback_used"]) / len(section_items)
    json_invalid_ratio = sum(1 for item in section_items if not item["json_valid"]) / len(section_items)
    generic_ratio = (
        sum(len(item["generic_terms"]) for item in section_items)
        / max(1, sum(len(item["key_terms"]) for item in section_items))
    )
    empty_summary_ratio = sum(1 for item in section_items if not item["summary"]) / len(section_items)
    empty_quotes_ratio = sum(1 for item in section_items if item["quotes_count"] == 0) / len(section_items)
    timeout_penalty = 0.0
    if int(metrics.get("timeout_count", 0)) > 0:
        timeout_penalty = 0.2

    score -= fallback_ratio * 0.30
    score -= json_invalid_ratio * 0.25
    score -= generic_ratio * 0.20
    score -= empty_summary_ratio * 0.15
    score -= empty_quotes_ratio * 0.10
    score -= timeout_penalty
    return round(max(0.0, min(1.0, score)), 4)


class Command(BaseCommand):
    help = "Run safe limited LLM preview (2-3 main sections) and save JSON/MD report."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True, help="Path to FB2/PDF file")
        parser.add_argument("--limit-main-sections", type=int, default=3)
        parser.add_argument("--max-calls", type=int, default=20)
        parser.add_argument("--max-blocks", type=int, default=10)
        parser.add_argument("--output", default="llm_preview_report", help="Output basename without extension")

    def handle(self, *args, **options):
        file_path = Path(str(options["file"])).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            raise CommandError(f"File not found: {file_path}")

        output_base = str(options["output"]).strip() or "llm_preview_report"
        output_root = _output_root()
        json_path = output_root / f"{output_base}.json"
        md_path = output_root / f"{output_base}.md"

        llm_state = ensure_llm_ready(require_enabled=True)
        runtime_cfg = get_llm_runtime_config()
        preview_started = time.time()
        report: dict[str, Any] = {
            "status": "failed_precheck",
            "book_file": str(file_path),
            "llm": {
                "provider": runtime_cfg.get("provider"),
                "base_url": runtime_cfg.get("base_url"),
                "selected_fast": llm_state.get("selected_fast"),
                "selected_high": llm_state.get("selected_high"),
                "selected_fallback": llm_state.get("selected_fallback"),
                "timeout_seconds": int(runtime_cfg.get("timeout_seconds", 30)),
                "max_retries": int(runtime_cfg.get("max_retries", 1)),
                "enable_fallback": bool(runtime_cfg.get("enable_fallback", True)),
                "available_models": llm_state.get("models", []),
                "ready": bool(llm_state.get("ok")),
                "error": llm_state.get("error", ""),
            },
            "preview": {},
            "metrics": {},
            "quality": {},
        }

        if not llm_state.get("ok"):
            self._write_reports(json_path, md_path, report)
            self.stdout.write(self.style.WARNING("LLM preview was not run: Ollama/provider is not ready."))
            self.stdout.write(f"JSON report: {json_path}")
            self.stdout.write(f"Markdown report: {md_path}")
            return

        parsed = parse_uploaded_book(file_path.read_bytes(), file_path.name)
        outline = build_canonical_outline(parsed)
        sections: list[CanonicalSection] = list(outline.get("sections", []))
        main_sections = [item for item in sections if item.content_type == "main_content" and item.is_main_content]
        if not main_sections:
            main_sections = [item for item in sections if item.is_main_content]
        if not main_sections:
            raise CommandError("No main content sections detected for llm_preview.")

        section_limit = max(2, min(3, int(options["limit_main_sections"])))
        selected_sections = main_sections[:section_limit]
        selected_indexes = {item.section_index - 1 for item in selected_sections}
        mini_parsed = SimpleNamespace(
            title=parsed.title,
            authors=parsed.authors,
            metadata=parsed.metadata,
            chapters=[parsed.chapters[idx] for idx in sorted(selected_indexes) if 0 <= idx < len(parsed.chapters)],
        )

        blocks, splitter_diag = split_into_logical_blocks_improved(mini_parsed)
        preview_blocks = blocks[: max(1, int(options["max_blocks"]))]

        env_keys = {
            "LLM_MAX_CALLS_PER_BOOK": str(min(20, max(1, int(options["max_calls"])))),
            "LLM_MAX_CALLS_PER_CHAPTER": "10",
            "LLM_MAX_CHUNKS_PER_SECTION": "2",
            "OLLAMA_TIMEOUT_SECONDS": "30",
            "LLM_MAX_INPUT_CHARS": "2500",
            "OLLAMA_MAX_TOKENS_JSON": "180",
            "OLLAMA_MAX_TOKENS_TEXT": "120",
            "OLLAMA_MODEL_HIGH": str(llm_state.get("selected_fast") or llm_state.get("selected_fallback") or ""),
        }
        old_values = {key: os.environ.get(key) for key in env_keys}

        try:
            os.environ.update(env_keys)
            result = run_hierarchical_llm_pipeline(mini_parsed, mode="llm_preview")
        finally:
            for key, old in old_values.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

        section_reports = [_section_payload_summary(item) for item in result.get("section_results", [])]
        chapter_payloads = result.get("chapter_payloads", [])
        book_payload = result.get("book_payload", {})
        metrics = result.get("metrics", {}) if isinstance(result.get("metrics", {}), dict) else {}

        failures: list[str] = []
        timeout_count = 0
        for item in section_reports:
            failure = item.get("llm_failure", "")
            if failure:
                failures.append(failure)
                if "timeout" in failure.lower():
                    timeout_count += 1
        if timeout_count == 0 and int(metrics.get("llm_failures_total", 0)) > 0 and int(metrics.get("llm_calls_total", 0)) > 0:
            # Section-level meta can be merged and lose low-level timeout marker.
            timeout_count = int(metrics.get("llm_failures_total", 0))

        quality_score = _preview_quality_score(section_reports, {"timeout_count": timeout_count})

        report["status"] = "completed"
        report["preview"] = {
            "book_title": parsed.title,
            "authors": parsed.authors,
            "sections_selected": [
                {
                    "section_index": sec.section_index,
                    "section_title": sec.section_title,
                    "chapter_title": sec.parent_chapter_title or sec.chapter_title,
                    "word_count": sec.word_count,
                    "paragraph_range": [sec.start_paragraph, sec.end_paragraph],
                }
                for sec in selected_sections
            ],
            "logical_blocks_count": len(blocks),
            "logical_blocks_preview": [
                {
                    "order_number": block.order_number,
                    "title": block.title,
                    "chapter_title": block.chapter_title,
                    "words": block.token_count,
                    "paragraph_range": [block.start_paragraph, block.end_paragraph],
                }
                for block in preview_blocks
            ],
            "sections_analysis": section_reports,
            "chapter_payloads_count": len(chapter_payloads),
            "book_summary_preview": str(book_payload.get("book_summary", "")).strip()[:1800],
        }
        report["metrics"] = {
            "llm_calls_total": int(metrics.get("llm_calls_total", 0)),
            "llm_success_calls": max(0, int(metrics.get("llm_calls_total", 0)) - int(metrics.get("llm_failures_total", 0))),
            "llm_failures_total": int(metrics.get("llm_failures_total", 0)),
            "fallback_used_count": int(metrics.get("fallback_used_count", 0)),
            "timeout_count": timeout_count,
            "sections_analyzed": len(section_reports),
            "duration_seconds": round(time.time() - preview_started, 2),
        }
        report["quality"] = {
            "llm_preview_quality_score": quality_score,
            "generic_terms_found": sorted(
                {
                    term
                    for item in section_reports
                    for term in item.get("generic_terms", [])
                    if term
                }
            ),
            "json_valid_sections": sum(1 for item in section_reports if item.get("json_valid")),
            "fallback_sections": sum(1 for item in section_reports if item.get("fallback_used")),
            "has_front_matter_noise": any(
                token in (item.get("section_title", "").lower())
                for item in section_reports
                for token in ("предислов", "благодар", "об авторах", "издатель")
            ),
            "ready_for_llm_full": quality_score >= 0.8 and int(metrics.get("llm_failures_total", 0)) == 0,
            "issues": failures[:20],
        }
        report["splitter_diagnostics"] = splitter_diag

        self._write_reports(json_path, md_path, report)
        self.stdout.write(self.style.SUCCESS("LLM preview completed."))
        self.stdout.write(f"JSON report: {json_path}")
        self.stdout.write(f"Markdown report: {md_path}")
        self.stdout.write(
            f"Score: {quality_score}, calls={report['metrics']['llm_calls_total']}, "
            f"fallback={report['metrics']['fallback_used_count']}, timeouts={timeout_count}"
        )

    def _write_reports(self, json_path: Path, md_path: Path, report: dict[str, Any]):
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._to_markdown(report), encoding="utf-8")

    def _to_markdown(self, report: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("# LLM Preview Report")
        lines.append("")
        lines.append(f"- Status: **{report.get('status')}**")
        lines.append(f"- Book file: `{report.get('book_file', '')}`")
        llm = report.get("llm", {})
        lines.append(f"- Ollama ready: **{llm.get('ready', False)}**")
        lines.append(f"- FAST model: `{llm.get('selected_fast')}`")
        lines.append(f"- HIGH model: `{llm.get('selected_high')}`")
        lines.append(f"- FALLBACK model: `{llm.get('selected_fallback')}`")
        if llm.get("error"):
            lines.append(f"- LLM error: `{llm.get('error')}`")
        lines.append("")

        preview = report.get("preview", {})
        if preview:
            lines.append("## Selected Sections")
            for sec in preview.get("sections_selected", []):
                lines.append(
                    f"- #{sec.get('section_index')} {sec.get('section_title')} "
                    f"(words={sec.get('word_count')}, p={sec.get('paragraph_range')})"
                )
            lines.append("")
            lines.append("## Section Results")
            for sec in preview.get("sections_analysis", []):
                lines.append(f"### {sec.get('section_title')}")
                lines.append(f"- JSON valid: {sec.get('json_valid')}, fallback: {sec.get('fallback_used')}")
                lines.append(f"- Summary: {sec.get('summary', '')[:420]}")
                lines.append(f"- Key terms: {', '.join(sec.get('key_terms', [])[:12]) or '-'}")
                lines.append(f"- Subtopics: {', '.join(sec.get('subtopics', [])[:10]) or '-'}")
                lines.append(f"- Generic terms: {', '.join(sec.get('generic_terms', [])) or '-'}")
                lines.append(f"- Quotes count: {sec.get('quotes_count', 0)}")
                lines.append("")

        metrics = report.get("metrics", {})
        quality = report.get("quality", {})
        lines.append("## Metrics")
        lines.append(f"- LLM calls total: {metrics.get('llm_calls_total', 0)}")
        lines.append(f"- LLM success: {metrics.get('llm_success_calls', 0)}")
        lines.append(f"- LLM failures: {metrics.get('llm_failures_total', 0)}")
        lines.append(f"- Fallback used: {metrics.get('fallback_used_count', 0)}")
        lines.append(f"- Timeout count: {metrics.get('timeout_count', 0)}")
        lines.append(f"- Duration (sec): {metrics.get('duration_seconds', 0)}")
        lines.append("")
        lines.append("## Quality")
        lines.append(f"- llm_preview_quality_score: **{quality.get('llm_preview_quality_score', 0.0)}**")
        lines.append(f"- ready_for_llm_full: **{quality.get('ready_for_llm_full', False)}**")
        lines.append(f"- generic_terms_found: {', '.join(quality.get('generic_terms_found', [])) or '-'}")
        lines.append(f"- json_valid_sections: {quality.get('json_valid_sections', 0)}")
        lines.append(f"- fallback_sections: {quality.get('fallback_sections', 0)}")
        lines.append(f"- has_front_matter_noise: {quality.get('has_front_matter_noise', False)}")
        issues = quality.get("issues", [])
        if issues:
            lines.append("- issues:")
            for issue in issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines).strip() + "\n"
