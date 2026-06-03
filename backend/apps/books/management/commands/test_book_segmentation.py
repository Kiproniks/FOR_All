from __future__ import annotations

import json
import re
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from django.core.management.base import BaseCommand, CommandError

from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.content_filter import is_front_matter_title
from apps.books.services.llm_service import ensure_llm_ready, mini_check_logical_block
from apps.books.services.logical_block_splitter import count_words, split_into_logical_blocks_improved
from apps.books.services.rag_service import cosine_similarity, create_embedding
from apps.books.services.structure_detector import CanonicalSection, build_canonical_outline

GENERIC_TITLE_RE = re.compile(
    r"^(?:глава|chapter|section|раздел|material|тема|часть|блок)\b[\s\-:]*\d*$",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"[.!?…»\"]$")
FRONT_MATTER_WORDS_RE = re.compile(
    r"(?:благодарност|об авторах|от издательства|copyright|all rights reserved|isbn|переводчик|предисловие)",
    re.IGNORECASE,
)
TRANSITION_MARKERS = (
    "однако",
    "с другой стороны",
    "таким образом",
    "далее",
    "итак",
    "в итоге",
    "следовательно",
)
FLAG_WEIGHTS = {
    "too_short": 0.08,
    "too_long": 0.10,
    "mixed_topics": 0.14,
    "generic_title": 0.08,
    "front_matter_leak": 0.20,
    "bad_boundary": 0.07,
    "split_needed": 0.08,
    "merge_needed": 0.05,
}


@dataclass(slots=True)
class BlockQualityResult:
    block_id: int
    order_number: int
    title: str
    chapter_title: str
    section_title: str
    dominant_content_type: str
    words: int
    chars: int
    start_paragraph: int
    end_paragraph: int
    excerpt: str
    boundary_reason: str
    flags: list[str]
    score: float
    llm_check: dict[str, Any]


def _safe_excerpt(text: str, limit: int = 500) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + "..."


def _dominant_content_type(content_types: dict[str, int] | None) -> str:
    if not content_types:
        return "unknown"
    sorted_items = sorted(content_types.items(), key=lambda item: item[1], reverse=True)
    return sorted_items[0][0] if sorted_items else "unknown"


def _is_bad_boundary(source_text: str) -> bool:
    lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    if not lines:
        return True
    first = lines[0]
    last = lines[-1]
    if first and first[0].islower():
        return True
    if last and not SENTENCE_END_RE.search(last):
        return True
    return False


def _topic_mixed_score(block) -> float:
    records = block.paragraph_records or []
    paragraphs = [str(row.get("text", "")).strip() for row in records if str(row.get("text", "")).strip()]
    if len(paragraphs) < 4:
        return 0.0
    vectors = [create_embedding(text[:1200]) for text in paragraphs[:14]]
    if len(vectors) < 2:
        return 0.0
    similarities: list[float] = []
    for i in range(1, len(vectors)):
        similarities.append(cosine_similarity(vectors[i - 1], vectors[i]))
    if not similarities:
        return 0.0
    avg_sim = mean(similarities)
    transition_hits = 0
    for paragraph in paragraphs:
        low = paragraph.lower()
        if any(marker in low for marker in TRANSITION_MARKERS):
            transition_hits += 1
    penalty = 0.0
    if avg_sim < 0.57:
        penalty += 0.55
    elif avg_sim < 0.64:
        penalty += 0.35
    if transition_hits >= 2:
        penalty += 0.20
    return min(1.0, penalty)


def _merge_needed_with_neighbors(current, prev_block, next_block) -> bool:
    if current is None:
        return False
    curr_words = count_words(current.clean_text_for_analysis or current.source_text)
    if curr_words >= 90:
        return False

    curr_vector = create_embedding((current.clean_text_for_analysis or current.source_text)[:1800])
    neighbors = [item for item in (prev_block, next_block) if item is not None]
    if not neighbors:
        return False
    for item in neighbors:
        neighbor_vector = create_embedding((item.clean_text_for_analysis or item.source_text)[:1800])
        if cosine_similarity(curr_vector, neighbor_vector) >= 0.86:
            return True
    return False


def _boundary_reason(block, *, min_words: int, max_words: int) -> str:
    reasons: list[str] = []
    records = block.paragraph_records or []
    if records:
        first_type = str(records[0].get("content_type", ""))
        if first_type in {"title", "subtitle"}:
            reasons.append("starts_with_heading")
    words = count_words(block.clean_text_for_analysis or block.source_text)
    if words >= int(max_words * 0.88):
        reasons.append("size_limit_chunking")
    if words <= max(70, int(min_words * 0.45)):
        reasons.append("short_local_fragment")
    if not reasons:
        reasons.append("semantic_continuity_within_section")
    return ", ".join(reasons)


def _section_to_dict(section: CanonicalSection) -> dict[str, Any]:
    return {
        "section_index": section.section_index,
        "chapter_title": section.chapter_title,
        "section_title": section.section_title,
        "parent_chapter_title": section.parent_chapter_title,
        "content_type": section.content_type,
        "is_main_content": section.is_main_content,
        "start_paragraph": section.start_paragraph,
        "end_paragraph": section.end_paragraph,
        "word_count": section.word_count,
        "level": section.level,
    }


def _quality_band(score: float) -> str:
    if score >= 0.9:
        return "excellent"
    if score >= 0.75:
        return "good_with_minor_fixes"
    if score >= 0.5:
        return "weak_needs_splitter_fixes"
    return "bad_do_not_run_llm"


class Command(BaseCommand):
    help = "Mini-test for structure detection + logical segmentation quality (no full LLM analysis)."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True, help="Path to FB2/PDF file")
        parser.add_argument("--limit-main-sections", type=int, default=3)
        parser.add_argument("--output", default="segmentation_mini_report", help="Output basename without extension")
        parser.add_argument("--min-words", type=int, default=220)
        parser.add_argument("--target-words", type=int, default=700)
        parser.add_argument("--max-words", type=int, default=1200)

    def handle(self, *args, **options):
        file_path = Path(str(options["file"])).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            raise CommandError(f"File not found: {file_path}")

        output_base = str(options["output"]).strip() or "segmentation_mini_report"
        output_root = Path.cwd().parent if Path.cwd().name.lower() == "backend" else Path.cwd()
        json_path = output_root / f"{output_base}.json"
        md_path = output_root / f"{output_base}.md"

        parsed = parse_uploaded_book(file_path.read_bytes(), file_path.name)
        outline = build_canonical_outline(parsed)
        sections: list[CanonicalSection] = list(outline.get("sections", []))

        main_sections = [s for s in sections if s.content_type == "main_content" and s.is_main_content]
        if not main_sections:
            main_sections = [s for s in sections if s.is_main_content]
        if not main_sections:
            main_sections = sections[:]

        selected_sections = main_sections[: max(1, int(options["limit_main_sections"]))]
        if not selected_sections:
            raise CommandError("No sections available for segmentation mini-test.")

        selected_indexes = {max(0, sec.section_index - 1) for sec in selected_sections}
        mini_chapters = [parsed.chapters[idx] for idx in sorted(selected_indexes) if 0 <= idx < len(parsed.chapters)]
        mini_parsed = SimpleNamespace(
            title=parsed.title,
            authors=parsed.authors,
            metadata=dict(parsed.metadata or {}),
            chapters=mini_chapters,
        )

        blocks, splitter_diag = split_into_logical_blocks_improved(
            mini_parsed,
            min_words=int(options["min_words"]),
            target_words=int(options["target_words"]),
            max_words=int(options["max_words"]),
        )
        llm_mini_check_enabled = False
        llm_state: dict[str, Any] = {"ok": False, "provider": "unknown", "error": "llm_not_checked"}
        if _is_ollama_endpoint_reachable():
            llm_state = ensure_llm_ready(require_enabled=False)
            llm_mini_check_enabled = bool(llm_state.get("ok"))
        else:
            llm_state = {"ok": False, "provider": "ollama", "error": "ollama_unreachable"}

        block_results: list[BlockQualityResult] = []
        llm_checked_count = 0
        for idx, block in enumerate(blocks, start=1):
            words = count_words(block.clean_text_for_analysis or block.source_text)
            chars = len(block.source_text or "")
            dominant_type = _dominant_content_type(block.content_types)
            flags: list[str] = []

            if words < 70 and dominant_type not in {"definition", "title", "subtitle"}:
                flags.append("too_short")
            if words > 1400:
                flags.append("too_long")
            if _topic_mixed_score(block) >= 0.45:
                flags.append("mixed_topics")
            if GENERIC_TITLE_RE.search((block.title or "").strip()):
                flags.append("generic_title")
            if is_front_matter_title(block.chapter_title) or FRONT_MATTER_WORDS_RE.search(block.source_text or ""):
                flags.append("front_matter_leak")
            if _is_bad_boundary(block.source_text or ""):
                flags.append("bad_boundary")
            if "mixed_topics" in flags or "too_long" in flags:
                flags.append("split_needed")

            prev_block = blocks[idx - 2] if idx > 1 else None
            next_block = blocks[idx] if idx < len(blocks) else None
            if _merge_needed_with_neighbors(block, prev_block, next_block):
                flags.append("merge_needed")

            llm_check: dict[str, Any] = {}
            if llm_mini_check_enabled and llm_checked_count < 5:
                llm_check = mini_check_logical_block(block.title, block.clean_text_for_analysis or block.source_text)
                if llm_check.get("llm_used"):
                    llm_checked_count += 1
                    if not llm_check.get("single_idea", True) and "mixed_topics" not in flags:
                        flags.append("mixed_topics")
                    if llm_check.get("split_recommended") and "split_needed" not in flags:
                        flags.append("split_needed")
                    if not llm_check.get("title_ok", True) and "generic_title" not in flags:
                        flags.append("generic_title")

            flags = sorted(set(flags))
            penalties = sum(FLAG_WEIGHTS.get(flag, 0.0) for flag in flags)
            block_score = max(0.0, 1.0 - penalties)
            if not flags:
                flags = ["good_block"]
                block_score = 1.0

            block_results.append(
                BlockQualityResult(
                    block_id=idx,
                    order_number=block.order_number,
                    title=block.title,
                    chapter_title=block.chapter_title,
                    section_title=block.section_title or block.chapter_title,
                    dominant_content_type=dominant_type,
                    words=words,
                    chars=chars,
                    start_paragraph=block.start_paragraph,
                    end_paragraph=block.end_paragraph,
                    excerpt=_safe_excerpt(block.source_text, limit=500),
                    boundary_reason=_boundary_reason(
                        block,
                        min_words=int(options["min_words"]),
                        max_words=int(options["max_words"]),
                    ),
                    flags=flags,
                    score=round(block_score, 4),
                    llm_check=llm_check,
                )
            )

        block_scores = [item.score for item in block_results] or [0.0]
        score = round(sum(block_scores) / max(1, len(block_scores)), 4)
        quality_band = _quality_band(score)

        front_matter_count = sum(1 for sec in sections if sec.content_type in {"front_matter", "preface", "acknowledgements", "author_bio", "publisher_note", "copyright", "abbreviation_list", "toc"})
        report_json = {
            "book": {
                "title": parsed.title,
                "authors": parsed.authors,
                "file": str(file_path),
            },
            "mini_test_config": {
                "limit_main_sections": int(options["limit_main_sections"]),
                "min_words": int(options["min_words"]),
                "target_words": int(options["target_words"]),
                "max_words": int(options["max_words"]),
            },
            "structure_detection": {
                "sections_total": int(outline.get("sections_total", 0)),
                "front_matter_sections": front_matter_count,
                "main_content_sections": int(outline.get("main_sections_count", 0)),
                "main_content_start": int(outline.get("main_content_start", 0) or 0),
                "section_types": outline.get("stats", {}).get("by_type", {}),
                "selected_main_sections": [_section_to_dict(sec) for sec in selected_sections],
            },
            "splitter_diagnostics": splitter_diag,
            "blocks": [asdict(item) for item in block_results],
            "quality": {
                "segmentation_quality_score": score,
                "quality_band": quality_band,
                "acceptance_ready_for_llm_preview": score >= 0.8,
                "flags_summary": _flags_summary(block_results),
            },
            "llm_mini_check": {
                "enabled": llm_mini_check_enabled,
                "provider_ok": llm_state.get("ok", False),
                "provider": llm_state.get("provider"),
                "error": llm_state.get("error", ""),
                "checked_blocks": llm_checked_count,
            },
        }

        md_content = self._build_markdown_report(report_json)
        json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(md_content, encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("Segmentation mini-test completed."))
        self.stdout.write(f"JSON report: {json_path}")
        self.stdout.write(f"Markdown report: {md_path}")
        self.stdout.write(
            f"Score: {score} ({quality_band}), ready_for_llm_preview={report_json['quality']['acceptance_ready_for_llm_preview']}"
        )

    def _build_markdown_report(self, report: dict[str, Any]) -> str:
        book = report["book"]
        structure = report["structure_detection"]
        quality = report["quality"]
        blocks = report["blocks"]
        filtered = report.get("splitter_diagnostics", {}).get("filtered_candidates", [])

        lines: list[str] = []
        lines.append("# Segmentation Mini Report")
        lines.append("")
        lines.append(f"- Book: **{book.get('title') or 'Untitled'}**")
        lines.append(f"- Authors: {book.get('authors') or '-'}")
        lines.append(f"- File: `{book.get('file')}`")
        lines.append("")
        lines.append("## Structure Detection")
        lines.append(f"- Total sections: {structure.get('sections_total', 0)}")
        lines.append(f"- Front matter sections: {structure.get('front_matter_sections', 0)}")
        lines.append(f"- Main content sections: {structure.get('main_content_sections', 0)}")
        lines.append(f"- Main content starts at section: {structure.get('main_content_start', 0)}")
        lines.append("")
        lines.append("### Selected Main Sections for Mini-Test")
        for sec in structure.get("selected_main_sections", []):
            lines.append(
                f"- #{sec.get('section_index')} | {sec.get('section_title')} "
                f"| type={sec.get('content_type')} | p{sec.get('start_paragraph')}-{sec.get('end_paragraph')} | words={sec.get('word_count')}"
            )
        lines.append("")
        lines.append("## Logical Blocks")
        for block in blocks:
            lines.append(f"### Block {block['order_number']}: {block['title']}")
            lines.append(
                f"- Chapter/Section: {block['chapter_title']} / {block['section_title']} | "
                f"p{block['start_paragraph']}-{block['end_paragraph']}"
            )
            lines.append(
                f"- Size: {block['words']} words, {block['chars']} chars | dominant_type={block['dominant_content_type']}"
            )
            lines.append(f"- Boundary reason: {block['boundary_reason']}")
            lines.append(f"- Flags: {', '.join(block['flags'])}")
            lines.append(f"- Block score: {block['score']}")
            llm_check = block.get("llm_check", {}) or {}
            if llm_check.get("llm_used"):
                lines.append(
                    "- LLM mini-check: "
                    f"single_idea={llm_check.get('single_idea')} | "
                    f"split_recommended={llm_check.get('split_recommended')} | "
                    f"title_ok={llm_check.get('title_ok')} | "
                    f"themes={', '.join(llm_check.get('themes', [])) or '-'}"
                )
            lines.append(f"- Excerpt: {block['excerpt']}")
            lines.append("")
        lines.append("## Filtered-Out Block Candidates")
        if not filtered:
            lines.append("- None")
        else:
            for row in filtered[:120]:
                lines.append(
                    f"- {row.get('chapter_title', '-')}: chunk={row.get('chunk_index')} "
                    f"reason={row.get('reason')} p{row.get('start_paragraph', '-')}-{row.get('end_paragraph', '-')}"
                )
        lines.append("")
        lines.append("## Quality Summary")
        lines.append(f"- segmentation_quality_score: **{quality.get('segmentation_quality_score')}**")
        lines.append(f"- quality_band: **{quality.get('quality_band')}**")
        lines.append(f"- ready_for_llm_preview: **{quality.get('acceptance_ready_for_llm_preview')}**")
        lines.append("- flags_summary:")
        for flag, count in sorted((quality.get("flags_summary") or {}).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  - {flag}: {count}")
        return "\n".join(lines).strip() + "\n"


def _flags_summary(results: list[BlockQualityResult]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in results:
        for flag in item.flags:
            summary[flag] = summary.get(flag, 0) + 1
    return summary


def _is_ollama_endpoint_reachable() -> bool:
    from os import getenv

    parsed = urlparse(getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 11434)
    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except OSError:
        return False
