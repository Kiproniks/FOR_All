from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.llm_service import analyze_section_with_llm, ensure_llm_ready, get_llm_runtime_config
from apps.books.services.logical_block_splitter import split_into_logical_blocks_improved
from apps.books.services.structure_detector import build_canonical_outline

GENERIC_TERMS = {
    "глава",
    "раздел",
    "материал",
    "текст",
    "книга",
    "тема",
    "chapter",
    "section",
    "material",
    "text",
}


def _project_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() == "backend" else cwd


def _clean_section_text(blocks, section_title: str, max_chars: int) -> str:
    parts: list[str] = []
    for block in blocks:
        block_section = block.section_title or block.chapter_title
        if block_section != section_title:
            continue
        text = (block.clean_text_for_analysis or block.source_text or "").strip()
        if text:
            parts.append(text)
    joined = "\n\n".join(parts).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars].rsplit(" ", 1)[0]


def _section_report(section, payload: dict[str, Any], duration: float) -> dict[str, Any]:
    meta = payload.get("_meta", {}) if isinstance(payload.get("_meta", {}), dict) else {}
    key_terms = [
        {
            "term": str(item.get("term", "")).strip(),
            "definition": str(item.get("definition", "")).strip(),
            "source_quote": str(item.get("source_quote", "")).strip(),
        }
        for item in payload.get("key_terms", [])
        if isinstance(item, dict) and str(item.get("term", "")).strip()
    ]
    subtopics = [
        {
            "title": str(item.get("title", "")).strip(),
            "summary": str(item.get("summary", "")).strip(),
            "source_quote": str(item.get("source_quote", "")).strip(),
        }
        for item in payload.get("subtopics", [])
        if isinstance(item, dict) and str(item.get("title", "")).strip()
    ]
    generic_terms = [item["term"] for item in key_terms if item["term"].lower() in GENERIC_TERMS]
    return {
        "section_index": section.section_index,
        "section_title": section.section_title,
        "word_count": section.word_count,
        "paragraph_range": [section.start_paragraph, section.end_paragraph],
        "duration_seconds": round(duration, 2),
        "json_valid": bool(meta.get("llm_used", False)),
        "fallback_used": bool(meta.get("fallback_used", False)),
        "llm_failure": str(meta.get("llm_failure", "")).strip(),
        "summary": str(payload.get("summary", "")).strip(),
        "key_terms": key_terms,
        "subtopics": subtopics,
        "generic_terms": generic_terms,
        "quality_flags": payload.get("quality_flags", []),
    }


class Command(BaseCommand):
    help = "Run section-level LLM preview only; no chapter/book/full analysis."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True)
        parser.add_argument("--limit-main-sections", type=int, default=3)
        parser.add_argument("--max-input-chars", type=int, default=2200)
        parser.add_argument("--output", default="llm_section_preview_report")
        parser.add_argument("--benchmark-models", default="", help="Comma-separated Ollama models for one-section benchmark")

    def handle(self, *args, **options):
        file_path = Path(str(options["file"])).expanduser().resolve()
        if not file_path.exists():
            raise CommandError(f"File not found: {file_path}")

        root = _project_root()
        output_base = str(options["output"]).strip() or "llm_section_preview_report"
        json_path = root / f"{output_base}.json"
        md_path = root / f"{output_base}.md"

        cfg = get_llm_runtime_config()
        llm_state = ensure_llm_ready(require_enabled=True)
        report: dict[str, Any] = {
            "status": "failed_precheck",
            "book_file": str(file_path),
            "llm": {
                "ready": bool(llm_state.get("ok")),
                "provider": cfg.get("provider"),
                "base_url": cfg.get("base_url"),
                "model_used": llm_state.get("selected_fast"),
                "fallback_model": llm_state.get("selected_fallback"),
                "available_models": llm_state.get("models", []),
                "timeout_seconds": cfg.get("timeout_seconds"),
                "error": llm_state.get("error", ""),
            },
            "metrics": {},
            "sections": [],
            "can_run_llm_full": False,
        }
        if not llm_state.get("ok"):
            self._write_reports(json_path, md_path, report)
            self.stdout.write(self.style.WARNING("Ollama/provider is not ready; section preview was not run."))
            return

        parsed = parse_uploaded_book(file_path.read_bytes(), file_path.name)
        outline = build_canonical_outline(parsed)
        sections = list(outline.get("sections", []))
        main_sections = [item for item in sections if item.content_type == "main_content" and item.is_main_content]
        selected = main_sections[: max(2, min(3, int(options["limit_main_sections"])))]
        selected_indexes = {item.section_index - 1 for item in selected}
        mini_parsed = SimpleNamespace(
            title=parsed.title,
            authors=parsed.authors,
            metadata=parsed.metadata,
            chapters=[parsed.chapters[idx] for idx in sorted(selected_indexes)],
        )
        blocks, splitter_diag = split_into_logical_blocks_improved(mini_parsed)

        env_overrides = {
            "OLLAMA_MODEL_HIGH": str(llm_state.get("selected_fast") or ""),
            "OLLAMA_MODEL_FAST": str(llm_state.get("selected_fast") or ""),
            "OLLAMA_MODEL_FALLBACK": str(llm_state.get("selected_fallback") or llm_state.get("selected_fast") or ""),
            "OLLAMA_TIMEOUT_SECONDS": "30",
            "OLLAMA_MAX_TOKENS_JSON": "180",
            "LLM_MAX_INPUT_CHARS": str(max(800, int(options["max_input_chars"]))),
        }
        previous = {key: os.environ.get(key) for key in env_overrides}

        benchmark_reports: list[dict[str, Any]] = []
        if options.get("benchmark_models"):
            first_section = selected[0]
            first_text = _clean_section_text(blocks, first_section.section_title, int(options["max_input_chars"]))
            for model_name in [item.strip() for item in str(options["benchmark_models"]).split(",") if item.strip()]:
                if model_name not in set(llm_state.get("models", [])):
                    benchmark_reports.append(
                        {
                            "model": model_name,
                            "valid_json": False,
                            "timeout": False,
                            "fallback_used": True,
                            "duration_seconds": 0.0,
                            "terms": [],
                            "error": "model_not_installed",
                        }
                    )
                    continue
                old_model_env = {
                    "OLLAMA_MODEL_FAST": os.environ.get("OLLAMA_MODEL_FAST"),
                    "OLLAMA_MODEL_HIGH": os.environ.get("OLLAMA_MODEL_HIGH"),
                    "OLLAMA_MODEL_FALLBACK": os.environ.get("OLLAMA_MODEL_FALLBACK"),
                    "LLM_ENABLE_FALLBACK": os.environ.get("LLM_ENABLE_FALLBACK"),
                    "LLM_MAX_RETRIES": os.environ.get("LLM_MAX_RETRIES"),
                }
                os.environ.update(
                    {
                        "OLLAMA_MODEL_FAST": model_name,
                        "OLLAMA_MODEL_HIGH": model_name,
                        "OLLAMA_MODEL_FALLBACK": model_name,
                        "LLM_ENABLE_FALLBACK": "false",
                        "LLM_MAX_RETRIES": "0",
                    }
                )
                bench_started = time.time()
                payload = analyze_section_with_llm(
                    section_title=first_section.section_title,
                    section_text=first_text,
                    chapter_title=first_section.parent_chapter_title or first_section.chapter_title,
                    section_type=first_section.content_type,
                )
                bench_duration = round(time.time() - bench_started, 2)
                meta = payload.get("_meta", {}) if isinstance(payload.get("_meta", {}), dict) else {}
                benchmark_reports.append(
                    {
                        "model": model_name,
                        "valid_json": bool(meta.get("llm_used", False)),
                        "timeout": bench_duration >= int(cfg.get("timeout_seconds", 30)) - 1,
                        "fallback_used": bool(meta.get("fallback_used", False)),
                        "duration_seconds": bench_duration,
                        "terms": [
                            str(item.get("term", "")).strip()
                            for item in payload.get("key_terms", [])
                            if isinstance(item, dict)
                        ][:8],
                        "error": str(meta.get("llm_failure", "")).strip(),
                    }
                )
                for key, value in old_model_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        section_reports: list[dict[str, Any]] = []
        started = time.time()
        try:
            os.environ.update(env_overrides)
            for section in selected:
                section_text = _clean_section_text(blocks, section.section_title, int(options["max_input_chars"]))
                call_started = time.time()
                payload = analyze_section_with_llm(
                    section_title=section.section_title,
                    section_text=section_text,
                    chapter_title=section.parent_chapter_title or section.chapter_title,
                    section_type=section.content_type,
                )
                section_reports.append(_section_report(section, payload, time.time() - call_started))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        calls_total = len(section_reports) * 3
        fallback_count = sum(1 for item in section_reports if item["fallback_used"])
        valid_sections = sum(1 for item in section_reports if item["json_valid"])
        valid_count = valid_sections * 3
        timeout_count = sum(
            1
            for item in section_reports
            if item["fallback_used"] and item["duration_seconds"] >= int(cfg.get("timeout_seconds", 30)) - 1
        )
        generic_terms = sorted({term for item in section_reports for term in item["generic_terms"]})
        ready_for_full = calls_total > 0 and valid_count == calls_total and fallback_count == 0 and not generic_terms

        report.update(
            {
                "status": "completed",
                "book": {"title": parsed.title, "authors": parsed.authors},
                "sections": section_reports,
                "splitter": {
                    "logical_blocks_count": len(blocks),
                    "diagnostics": splitter_diag,
                },
                "metrics": {
                    "llm_calls_total": calls_total,
                    "llm_success_calls": valid_count,
                    "fallback_used_count": fallback_count,
                    "timeout_count": timeout_count,
                    "json_valid_sections": valid_sections,
                    "duration_seconds": round(time.time() - started, 2),
                },
                "generic_terms_found": generic_terms,
                "benchmark": benchmark_reports,
                "can_run_llm_full": ready_for_full,
            }
        )
        self._write_reports(json_path, md_path, report)
        self.stdout.write(self.style.SUCCESS("Section-level LLM preview completed."))
        self.stdout.write(f"JSON report: {json_path}")
        self.stdout.write(f"Markdown report: {md_path}")
        self.stdout.write(
            f"calls={calls_total}, json_valid_sections={valid_sections}, fallback={fallback_count}, "
            f"timeouts={timeout_count}, can_run_llm_full={ready_for_full}"
        )

    def _write_reports(self, json_path: Path, md_path: Path, report: dict[str, Any]) -> None:
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown(report), encoding="utf-8")

    def _markdown(self, report: dict[str, Any]) -> str:
        lines = ["# LLM Section Preview Report", ""]
        llm = report.get("llm", {})
        lines.append(f"- Status: **{report.get('status')}**")
        lines.append(f"- Ollama ready: **{llm.get('ready')}**")
        lines.append(f"- Model used: `{llm.get('model_used')}`")
        lines.append(f"- Fallback model: `{llm.get('fallback_model')}`")
        lines.append("")
        metrics = report.get("metrics", {})
        lines.append("## Metrics")
        for key in ("llm_calls_total", "llm_success_calls", "fallback_used_count", "timeout_count", "json_valid_sections", "duration_seconds"):
            lines.append(f"- {key}: {metrics.get(key, 0)}")
        lines.append(f"- can_run_llm_full: **{report.get('can_run_llm_full', False)}**")
        if report.get("benchmark"):
            lines.append("")
            lines.append("## Mini Benchmark")
            for item in report.get("benchmark", []):
                lines.append(
                    f"- `{item.get('model')}`: valid_json={item.get('valid_json')}, "
                    f"fallback={item.get('fallback_used')}, timeout={item.get('timeout')}, "
                    f"duration={item.get('duration_seconds')}s, terms={', '.join(item.get('terms', [])[:6]) or '-'}"
                )
        lines.append("")
        lines.append("## Sections")
        for section in report.get("sections", []):
            terms = [item["term"] for item in section.get("key_terms", [])][:8]
            subtopics = [item["title"] for item in section.get("subtopics", [])][:8]
            lines.append(f"### {section.get('section_title')}")
            lines.append(f"- JSON valid: {section.get('json_valid')}, fallback: {section.get('fallback_used')}, duration: {section.get('duration_seconds')}s")
            lines.append(f"- Summary: {section.get('summary', '')[:500]}")
            lines.append(f"- Key terms: {', '.join(terms) or '-'}")
            lines.append(f"- Subtopics: {', '.join(subtopics) or '-'}")
            lines.append(f"- Failure: `{section.get('llm_failure') or '-'}`")
            lines.append("")
        return "\n".join(lines).strip() + "\n"
