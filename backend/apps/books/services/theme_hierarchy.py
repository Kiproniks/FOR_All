from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymorphy3

from apps.books.models import LogicalBlock
from apps.books.services.concept_normalizer import normalize_concept_name
from apps.books.services.llm_service import (
    ensure_grounded_summary,
    extract_theme_hierarchy_for_chapter,
    extractive_theme_summary_from_digests,
)

morph = pymorphy3.MorphAnalyzer()
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
STOP_TOPIC_WORDS = {
    "книга",
    "автор",
    "глава",
    "тема",
    "раздел",
    "пример",
    "задача",
    "текст",
    "данный",
    "этот",
    "такой",
    "вопрос",
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


def _quote_from_text(text: str, max_chars: int = 220) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0].strip()
    if len(sentence) < 20:
        sentence = cleaned
    return sentence[:max_chars]


def _theme_count_for_chapter(blocks_count: int) -> int:
    if blocks_count <= 2:
        return 1
    if blocks_count <= 5:
        return 2
    if blocks_count <= 9:
        return 3
    return 4


def _split_contiguous(total: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, total))
    ranges: list[tuple[int, int]] = []
    base = total // parts
    rem = total % parts
    start = 1
    for idx in range(parts):
        length = base + (1 if idx < rem else 0)
        end = start + length - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def _fallback_topic_phrases(text: str, chapter_title: str, max_items: int = 4) -> list[str]:
    words = [item.lower() for item in WORD_RE.findall(text)]
    phrases = Counter()

    for idx in range(len(words)):
        for span in (2, 3):
            chunk = words[idx : idx + span]
            if len(chunk) < span:
                continue
            if any(token in STOP_TOPIC_WORDS for token in chunk):
                continue

            parsed = [morph.parse(token)[0] for token in chunk]
            if "NOUN" not in parsed[-1].tag:
                continue
            if not all(("NOUN" in p.tag) or ("ADJF" in p.tag) or ("PRTF" in p.tag) for p in parsed):
                continue

            normalized = " ".join(p.normal_form for p in parsed)
            if len(normalized) < 4:
                continue
            phrases[normalized] += 1

    result = [name for name, _ in phrases.most_common(max_items)]
    if len(result) >= max_items:
        return result[:max_items]

    noun_counter = Counter()
    for word in words:
        if word in STOP_TOPIC_WORDS:
            continue
        parse = morph.parse(word)[0]
        if "NOUN" in parse.tag and len(parse.normal_form) > 3:
            noun_counter[parse.normal_form] += 1

    for noun, _ in noun_counter.most_common(max_items * 2):
        candidate = f"{noun} в контексте {chapter_title[:40]}" if chapter_title else noun
        if candidate not in result:
            result.append(candidate)
        if len(result) >= max_items:
            break

    if not result:
        result = [f"Основные идеи: {chapter_title or 'раздел'}"]
    return result[:max_items]


def _build_fallback_subtopics(
    chapter_title: str,
    text: str,
    start_block_number: int,
    end_block_number: int,
    start_paragraph: int,
    end_paragraph: int,
) -> list[ThemeSubtopicData]:
    names = _fallback_topic_phrases(text, chapter_title, max_items=4)
    score_step = 0.15
    subtopics: list[ThemeSubtopicData] = []
    for idx, name in enumerate(names, start=1):
        subtopics.append(
            ThemeSubtopicData(
                name=name[:255],
                summary="Ключевая подтема этого смыслового фрагмента.",
                source_quote="",
                importance_score=max(0.2, 0.95 - idx * score_step),
                start_block_number=start_block_number,
                end_block_number=end_block_number,
                start_paragraph=start_paragraph,
                end_paragraph=end_paragraph,
            )
        )
    return subtopics


def _fallback_chapter_themes(chapter_title: str, blocks: list[LogicalBlock]) -> list[dict[str, Any]]:
    desired = _theme_count_for_chapter(len(blocks))
    segments = _split_contiguous(len(blocks), desired)
    themes: list[dict[str, Any]] = []

    for idx, (start_local, end_local) in enumerate(segments, start=1):
        segment_blocks = blocks[start_local - 1 : end_local]
        summary_parts = [b.short_summary for b in segment_blocks[:2] if b.short_summary]
        summary = _normalize_text(" ".join(summary_parts))
        if not summary:
            summary = _normalize_text(" ".join(b.source_text[:350] for b in segment_blocks[:2]))

        text = "\n\n".join(b.source_text for b in segment_blocks)
        themes.append(
            {
                "title": f"{chapter_title or 'Тема'} — часть {idx}",
                "summary": summary or "Ключевая смысловая часть главы.",
                "start_block": start_local,
                "end_block": end_local,
                "subthemes": [
                    {
                        "name": item.name,
                        "summary": item.summary,
                        "source_quote": item.source_quote,
                        "importance_score": item.importance_score,
                        "start_block": start_local,
                        "end_block": end_local,
                    }
                    for item in _build_fallback_subtopics(
                        chapter_title=chapter_title,
                        text=text,
                        start_block_number=segment_blocks[0].order_number,
                        end_block_number=segment_blocks[-1].order_number,
                        start_paragraph=segment_blocks[0].start_paragraph,
                        end_paragraph=segment_blocks[-1].end_paragraph,
                    )
                ],
            }
        )

    return themes


def _normalize_llm_themes(raw_themes: list[dict[str, Any]], blocks_count: int) -> list[dict[str, Any]]:
    themes: list[dict[str, Any]] = []

    themes = [item for item in raw_themes if isinstance(item, dict)]

    normalized: list[dict[str, Any]] = []
    for item in themes:
        try:
            start_block = int(item.get("start_block", 1))
        except (TypeError, ValueError):
            start_block = 1
        try:
            end_block = int(item.get("end_block", start_block))
        except (TypeError, ValueError):
            end_block = start_block

        start_block = max(1, min(start_block, blocks_count))
        end_block = max(start_block, min(end_block, blocks_count))

        title = _normalize_text(str(item.get("title", "")))
        summary = _normalize_text(str(item.get("summary", "")))
        raw_subthemes = item.get("subthemes", [])
        subthemes = [sub for sub in raw_subthemes if isinstance(sub, dict)] if isinstance(raw_subthemes, list) else []

        normalized.append(
            {
                "title": title,
                "summary": summary,
                "start_block": start_block,
                "end_block": end_block,
                "subthemes": subthemes,
            }
        )

    normalized.sort(key=lambda item: (item["start_block"], item["end_block"]))
    return normalized


def _llm_chapter_themes(chapter_title: str, blocks: list[LogicalBlock]) -> list[dict[str, Any]]:
    desired = _theme_count_for_chapter(len(blocks))
    digests: list[dict[str, Any]] = []
    for local_idx, block in enumerate(blocks, start=1):
        digest = _normalize_text(block.short_summary or block.source_text[:240])
        digests.append(
            {
                "index": local_idx,
                "start_paragraph": block.start_paragraph,
                "end_paragraph": block.end_paragraph,
                "summary": digest[:260],
            }
        )

    raw_themes = extract_theme_hierarchy_for_chapter(
        chapter_title=chapter_title,
        block_digests=digests,
        desired_themes=desired,
    )
    if not raw_themes:
        return []
    return _normalize_llm_themes(raw_themes, len(blocks))


def build_theme_hierarchy(blocks: list[LogicalBlock]) -> list[BookThemeData]:
    if not blocks:
        return []

    chapters: list[tuple[str, list[LogicalBlock]]] = []
    current_title = None
    current_blocks: list[LogicalBlock] = []

    for block in sorted(blocks, key=lambda item: item.order_number):
        chapter_title = _normalize_text(block.chapter_title) or "Основной раздел"
        if current_title is None:
            current_title = chapter_title
        if chapter_title != current_title:
            chapters.append((current_title, current_blocks))
            current_title = chapter_title
            current_blocks = []
        current_blocks.append(block)

    if current_blocks:
        chapters.append((current_title or "Основной раздел", current_blocks))

    theme_order = 1
    result: list[BookThemeData] = []

    for chapter_title, chapter_blocks in chapters:
        llm_themes = _llm_chapter_themes(chapter_title, chapter_blocks)
        chapter_themes = llm_themes or _fallback_chapter_themes(chapter_title, chapter_blocks)

        if not chapter_themes:
            chapter_themes = _fallback_chapter_themes(chapter_title, chapter_blocks)

        for theme_idx, theme_raw in enumerate(chapter_themes, start=1):
            start_local = int(theme_raw.get("start_block", 1))
            end_local = int(theme_raw.get("end_block", start_local))
            start_local = max(1, min(start_local, len(chapter_blocks)))
            end_local = max(start_local, min(end_local, len(chapter_blocks)))
            themed_blocks = chapter_blocks[start_local - 1 : end_local]
            if not themed_blocks:
                continue

            theme_title = _normalize_text(str(theme_raw.get("title", "")))
            if not theme_title:
                theme_title = f"{chapter_title} — тема {theme_idx}"
            raw_theme_summary = _normalize_text(str(theme_raw.get("summary", "")))
            fallback_theme_summary = extractive_theme_summary_from_digests(
                [{"summary": b.short_summary or b.source_text[:240]} for b in themed_blocks],
                limit_sentences=3,
            )
            theme_evidence_text = "\n\n".join(
                _normalize_text(f"{b.short_summary or ''}\n{b.source_text[:1400]}") for b in themed_blocks
            )
            theme_summary = ensure_grounded_summary(
                raw_theme_summary,
                theme_evidence_text,
                fallback_theme_summary,
            )

            start_block_number = themed_blocks[0].order_number
            end_block_number = themed_blocks[-1].order_number
            start_paragraph = themed_blocks[0].start_paragraph
            end_paragraph = themed_blocks[-1].end_paragraph

            raw_subthemes = theme_raw.get("subthemes", [])
            if not isinstance(raw_subthemes, list):
                raw_subthemes = []

            subtopics: list[ThemeSubtopicData] = []
            for item in raw_subthemes:
                if not isinstance(item, dict):
                    continue
                name = _normalize_text(str(item.get("name", "")))
                if not name:
                    continue
                summary_raw = _normalize_text(str(item.get("summary", "")))
                quote = _normalize_text(str(item.get("source_quote", "")))
                try:
                    importance = float(item.get("importance_score", 0.6))
                except (TypeError, ValueError):
                    importance = 0.6
                importance = max(0.0, min(1.0, importance))

                sub_start_local = int(item.get("start_block", start_local)) if str(item.get("start_block", "")).isdigit() else start_local
                sub_end_local = int(item.get("end_block", end_local)) if str(item.get("end_block", "")).isdigit() else end_local
                sub_start_local = max(start_local, min(sub_start_local, end_local))
                sub_end_local = max(sub_start_local, min(sub_end_local, end_local))
                sub_blocks = chapter_blocks[sub_start_local - 1 : sub_end_local]
                if sub_blocks:
                    sub_start_par = sub_blocks[0].start_paragraph
                    sub_end_par = sub_blocks[-1].end_paragraph
                else:
                    sub_start_par = start_paragraph
                    sub_end_par = end_paragraph

                sub_evidence_text = "\n\n".join(
                    _normalize_text(f"{b.short_summary or ''}\n{b.source_text[:1200]}") for b in (sub_blocks or themed_blocks)
                )
                sub_fallback_summary = extractive_theme_summary_from_digests(
                    [{"summary": b.short_summary or b.source_text[:200]} for b in (sub_blocks or themed_blocks)],
                    limit_sentences=2,
                )
                summary = ensure_grounded_summary(summary_raw, sub_evidence_text, sub_fallback_summary)
                if not quote:
                    quote = _quote_from_text(sub_evidence_text, max_chars=220)

                subtopics.append(
                    ThemeSubtopicData(
                        name=name[:255],
                        summary=summary[:1000],
                        source_quote=quote[:500],
                        importance_score=importance,
                        start_block_number=sub_blocks[0].order_number if sub_blocks else start_block_number,
                        end_block_number=sub_blocks[-1].order_number if sub_blocks else end_block_number,
                        start_paragraph=sub_start_par,
                        end_paragraph=sub_end_par,
                    )
                )

            if len(subtopics) < 2:
                fallback_text = "\n\n".join(block.source_text for block in themed_blocks)
                subtopics = _build_fallback_subtopics(
                    chapter_title=chapter_title,
                    text=fallback_text,
                    start_block_number=start_block_number,
                    end_block_number=end_block_number,
                    start_paragraph=start_paragraph,
                    end_paragraph=end_paragraph,
                )

            deduped: list[ThemeSubtopicData] = []
            seen_sub = set()
            for subtopic in subtopics:
                normalized = normalize_concept_name(subtopic.name)
                if not normalized or normalized in seen_sub:
                    continue
                seen_sub.add(normalized)
                deduped.append(subtopic)
                if len(deduped) >= 4:
                    break

            result.append(
                BookThemeData(
                    chapter_title=chapter_title,
                    title=theme_title[:512],
                    order_number=theme_order,
                    start_block_number=start_block_number,
                    end_block_number=end_block_number,
                    start_paragraph=start_paragraph,
                    end_paragraph=end_paragraph,
                    summary=theme_summary[:2000],
                    subtopics=deduped,
                )
            )
            theme_order += 1

    return result
