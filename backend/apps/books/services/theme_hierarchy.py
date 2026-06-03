from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from apps.books.services.concept_extractor import build_theme_subtopics_from_blocks
from apps.books.services.llm_service import ensure_grounded_summary, extractive_theme_summary_from_digests

GENERIC_THEME_TITLES = {
    "без названия главы",
    "раздел без названия",
    "часть 1",
    "часть 2",
    "часть 3",
    "source material",
}


@dataclass
class ThemeSubtopicData:
    name: str
    summary: str
    source_quote: str
    importance_score: float
    start_block_number: int
    end_block_number: int
    start_paragraph: int
    end_paragraph: int


@dataclass
class BookThemeData:
    chapter_title: str
    title: str
    order_number: int
    start_block_number: int
    end_block_number: int
    start_paragraph: int
    end_paragraph: int
    summary: str
    subtopics: list[ThemeSubtopicData]


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _is_generic_theme_title(title: str) -> bool:
    low = _normalize_text(title).lower()
    if not low:
        return True
    if low in GENERIC_THEME_TITLES:
        return True
    if re.fullmatch(r"часть\s*\d+", low):
        return True
    if "без названия" in low:
        return True
    return False


def _theme_group_key(block: Any) -> str:
    semantic_data = getattr(block, "semantic_data", None) or {}
    if isinstance(semantic_data, dict):
        section_path = semantic_data.get("section_path") or []
        if isinstance(section_path, list) and section_path:
            # Prefer top-most meaningful section title.
            top = section_path[0]
            if isinstance(top, dict):
                top_title = _normalize_text(str(top.get("title", "")))
                if top_title:
                    return top_title

        section_title = _normalize_text(str(semantic_data.get("section_title", "")))
        if section_title:
            return section_title

    chapter_title = _normalize_text(str(getattr(block, "chapter_title", "")))
    if chapter_title:
        return chapter_title

    return "Основной раздел"


def _theme_title_from_group(raw_title: str, subtopics: list[dict[str, Any]]) -> str:
    title = _normalize_text(raw_title)
    if title and not _is_generic_theme_title(title):
        return title[:512]

    if subtopics:
        top_names = [item.get("name", "") for item in subtopics if item.get("name")]
        top_names = [item for item in top_names if item][:2]
        if top_names:
            return "; ".join(top_names)[:512]

    return "Основная тема раздела"


def _theme_summary(blocks: list[Any]) -> str:
    digests = []
    evidence_parts = []
    for block in blocks:
        short_summary = _normalize_text(getattr(block, "short_summary", ""))
        semantic_data = getattr(block, "semantic_data", None) or {}
        clean_text = semantic_data.get("clean_text_for_analysis") if isinstance(semantic_data, dict) else None
        source_text = _normalize_text(clean_text or getattr(block, "source_text", ""))

        if short_summary:
            digests.append({"summary": short_summary})
        elif source_text:
            digests.append({"summary": source_text[:240]})

        if source_text:
            evidence_parts.append(source_text[:1600])

    fallback = extractive_theme_summary_from_digests(digests, limit_sentences=3)
    evidence_text = "\n\n".join(evidence_parts)[:12000]
    return ensure_grounded_summary(fallback, evidence_text, fallback)


def _theme_subtopics(blocks: list[Any]) -> list[ThemeSubtopicData]:
    raw_subtopics = build_theme_subtopics_from_blocks(
        theme_title=_theme_group_key(blocks[0]),
        blocks=blocks,
        max_items=6,
    )

    if not raw_subtopics:
        return []

    subtopics: list[ThemeSubtopicData] = []
    for item in raw_subtopics:
        try:
            score = float(item.get("importance_score", 0.5))
        except (TypeError, ValueError):
            score = 0.5

        start_block = int(item.get("source_block_order", blocks[0].order_number))
        end_block = start_block

        anchor = next((block for block in blocks if block.order_number == start_block), blocks[0])
        subtopics.append(
            ThemeSubtopicData(
                name=_normalize_text(str(item.get("name", "")))[:255],
                summary=_normalize_text(str(item.get("summary", "")))[:1000],
                source_quote=_normalize_text(str(item.get("source_quote", "")))[:500],
                importance_score=max(0.0, min(1.0, score)),
                start_block_number=start_block,
                end_block_number=end_block,
                start_paragraph=int(getattr(anchor, "start_paragraph", 0)),
                end_paragraph=int(getattr(anchor, "end_paragraph", 0)),
            )
        )

    return [item for item in subtopics if item.name][:4]


def build_theme_hierarchy(blocks: list[Any]) -> list[BookThemeData]:
    """
    Improved theme hierarchy:
    - prefers real chapter/section titles
    - avoids coarse equal-range splitting
    - builds subtopics from clean block text
    """

    if not blocks:
        return []

    ordered_blocks = sorted(blocks, key=lambda item: item.order_number)
    non_front_blocks = []
    for block in ordered_blocks:
        semantic_data = getattr(block, "semantic_data", None) or {}
        is_front = bool(semantic_data.get("front_matter")) if isinstance(semantic_data, dict) else False
        if not is_front:
            non_front_blocks.append(block)
    if non_front_blocks:
        ordered_blocks = non_front_blocks

    groups: list[tuple[str, list[Any]]] = []
    current_title = _theme_group_key(ordered_blocks[0])
    current_blocks: list[Any] = []

    for block in ordered_blocks:
        title = _theme_group_key(block)
        if title != current_title and current_blocks:
            groups.append((current_title, current_blocks))
            current_title = title
            current_blocks = []
        current_blocks.append(block)

    if current_blocks:
        groups.append((current_title, current_blocks))

    # Merge very small neighboring groups only when titles are generic,
    # otherwise keep explicit chapter/section boundaries.
    merged_groups: list[tuple[str, list[Any]]] = []
    for title, group_blocks in groups:
        if (
            merged_groups
            and len(group_blocks) == 1
            and _is_generic_theme_title(title)
            and _is_generic_theme_title(merged_groups[-1][0])
        ):
            prev_title, prev_blocks = merged_groups[-1]
            merged_groups[-1] = (prev_title, prev_blocks + group_blocks)
        else:
            merged_groups.append((title, group_blocks))

    themes: list[BookThemeData] = []
    for order, (raw_title, group_blocks) in enumerate(merged_groups, start=1):
        start_block = group_blocks[0].order_number
        end_block = group_blocks[-1].order_number
        start_paragraph = group_blocks[0].start_paragraph
        end_paragraph = group_blocks[-1].end_paragraph

        subtopics = _theme_subtopics(group_blocks)
        title = _theme_title_from_group(raw_title, [
            {"name": item.name} for item in subtopics
        ])
        summary = _theme_summary(group_blocks)

        chapter_title = _normalize_text(str(getattr(group_blocks[0], "chapter_title", ""))) or title
        themes.append(
            BookThemeData(
                chapter_title=chapter_title[:512],
                title=title[:512],
                order_number=order,
                start_block_number=start_block,
                end_block_number=end_block,
                start_paragraph=start_paragraph,
                end_paragraph=end_paragraph,
                summary=summary[:2000],
                subtopics=subtopics,
            )
        )

    return themes
