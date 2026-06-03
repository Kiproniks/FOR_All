from __future__ import annotations

import re
from typing import Any

from apps.books.services.content_filter import is_front_matter_title

SERVICE_NOISE_RE = re.compile(
    r"(?:\bISBN\b|©|copyright|all rights reserved|все права защищены|переводч|издательств|тираж)",
    re.IGNORECASE,
)
GENERIC_THEME_RE = re.compile(r"(?:без названия|часть\s*\d+|source material)", re.IGNORECASE)
GENERIC_SUBTOPIC_RE = re.compile(
    r"(?:такой образ|данный случай|следующий раздел|эта глава|этот пример|другой способ|важный вопрос|основная проблема)$",
    re.IGNORECASE,
)

DISALLOWED_QUOTE_TYPES = {
    "code",
    "copyright",
    "exercise",
    "question",
    "figure_caption",
    "table_caption",
    "dedication",
    "acknowledgements",
    "acronym_list",
    "toc",
    "index",
    "bibliography",
}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _block_is_service_only(block: Any) -> bool:
    semantic_data = getattr(block, "semantic_data", None) or {}
    rows = semantic_data.get("paragraph_records") if isinstance(semantic_data, dict) else None
    if not isinstance(rows, list) or not rows:
        return False

    service_rows = 0
    text_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _normalize_text(str(row.get("text", "")))
        if not text:
            continue
        text_rows += 1
        ctype = str(row.get("content_type", ""))
        if ctype in DISALLOWED_QUOTE_TYPES:
            service_rows += 1
    if text_rows == 0:
        return True
    return service_rows / text_rows >= 0.8


def _quote_looks_from_disallowed_type(quote: str, block: Any) -> bool:
    quote = _normalize_text(quote)
    if not quote:
        return False
    semantic_data = getattr(block, "semantic_data", None) or {}
    rows = semantic_data.get("paragraph_records") if isinstance(semantic_data, dict) else None
    if not isinstance(rows, list):
        return False

    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _normalize_text(str(row.get("text", "")))
        ctype = str(row.get("content_type", ""))
        if not text or ctype not in DISALLOWED_QUOTE_TYPES:
            continue
        if quote.lower() in text.lower() or text.lower() in quote.lower():
            return True
    return False


def evaluate_analysis_quality(
    *,
    summary_text: str,
    blocks: list[Any],
    themes: list[Any],
    subtopics: list[Any],
    concept_mentions: list[Any],
    parser_metadata: dict[str, Any] | None = None,
    content_filter_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parser_metadata = parser_metadata or {}
    content_filter_stats = content_filter_stats or {}

    problems: list[str] = []

    total_blocks = len(blocks)
    untitled_blocks = sum(1 for block in blocks if "без названия" in _normalize_text(getattr(block, "chapter_title", "")).lower())
    untitled_ratio = (untitled_blocks / total_blocks) if total_blocks else 0.0
    if total_blocks >= 8 and untitled_ratio > 0.35:
        problems.append("too_many_untitled_chapters")

    front_matter_blocks = 0
    for block in blocks:
        semantic_data = getattr(block, "semantic_data", None) or {}
        section_title = ""
        if isinstance(semantic_data, dict):
            if semantic_data.get("front_matter"):
                front_matter_blocks += 1
                continue
            section_title = str(semantic_data.get("section_title", ""))
        chapter_title = _normalize_text(getattr(block, "chapter_title", ""))
        if is_front_matter_title(section_title or chapter_title):
            front_matter_blocks += 1

    front_ratio = (front_matter_blocks / total_blocks) if total_blocks else 0.0
    if (total_blocks >= 3 and front_ratio >= 0.4) or (total_blocks > 0 and front_ratio >= 0.9):
        problems.append("front_matter_dominates_blocks")

    if SERVICE_NOISE_RE.search(summary_text or ""):
        problems.append("summary_contains_copyright")

    themes_count = len(themes)
    chapters_count = int(parser_metadata.get("sections_count") or parser_metadata.get("chapters_count") or 0)
    if chapters_count >= 8 and themes_count <= 4:
        problems.append("themes_are_too_coarse")

    if any(GENERIC_THEME_RE.search(_normalize_text(getattr(theme, "title", ""))) for theme in themes):
        problems.append("generic_theme_titles")

    if themes and sum(1 for theme in themes if is_front_matter_title(_normalize_text(getattr(theme, "title", "")))) / len(themes) >= 0.5:
        problems.append("themes_from_front_matter")

    if any(GENERIC_SUBTOPIC_RE.search(_normalize_text(getattr(subtopic, "name", ""))) for subtopic in subtopics):
        problems.append("subtopics_are_generic")

    quote_from_code = 0
    for mention in concept_mentions:
        block = getattr(mention, "logical_block", None)
        if block and _quote_looks_from_disallowed_type(getattr(mention, "source_quote", ""), block):
            quote_from_code += 1
    if quote_from_code > 0:
        problems.append("quotes_from_code")

    if any(_block_is_service_only(block) for block in blocks):
        problems.append("service_only_blocks_present")

    total_words = sum(len(_normalize_text(getattr(block, "source_text", "")).split()) for block in blocks)
    if total_words >= 18000 and len(concept_mentions) < 12:
        problems.append("too_few_concepts")

    if len(_normalize_text(summary_text)) < 120:
        problems.append("summary_too_short")

    filtered_ratio = 0.0
    kept_count = int(content_filter_stats.get("kept_count", 0))
    removed_count = int(content_filter_stats.get("removed_count", 0))
    total_paragraphs = int(content_filter_stats.get("total_paragraphs", 0))
    if total_paragraphs > 0:
        filtered_ratio = removed_count / total_paragraphs

    penalty_by_problem = {
        "summary_contains_copyright": 0.22,
        "front_matter_dominates_blocks": 0.22,
        "themes_from_front_matter": 0.18,
        "themes_are_too_coarse": 0.16,
        "generic_theme_titles": 0.12,
        "subtopics_are_generic": 0.12,
        "quotes_from_code": 0.16,
        "too_few_concepts": 0.15,
        "service_only_blocks_present": 0.12,
        "summary_too_short": 0.10,
        "too_many_untitled_chapters": 0.10,
    }

    quality_score = 1.0
    for problem in sorted(set(problems)):
        quality_score -= penalty_by_problem.get(problem, 0.08)
    if filtered_ratio > 0.75:
        quality_score -= 0.08
    quality_score = max(0.0, round(quality_score, 4))

    return {
        "quality_score": quality_score,
        "problems": sorted(set(problems)),
        "stats": {
            "total_blocks": total_blocks,
            "untitled_blocks": untitled_blocks,
            "front_matter_blocks": front_matter_blocks,
            "themes_count": themes_count,
            "subtopics_count": len(subtopics),
            "concept_mentions_count": len(concept_mentions),
            "filtered_paragraphs": removed_count,
            "kept_paragraphs": kept_count,
            "filtered_ratio": round(filtered_ratio, 4),
            "quote_from_disallowed": quote_from_code,
            "total_words": total_words,
        },
    }
