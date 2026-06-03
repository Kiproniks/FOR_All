from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from apps.books.services.book_parser import chapter_paragraph_records
from apps.books.services.content_filter import filter_content_for_analysis, is_front_matter_title
from apps.books.services.llm_service import suggest_chapter_boundaries
from apps.books.services.rag_service import cosine_similarity, create_embedding

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
TITLE_HINT_RE = re.compile(r"^\s*(?:\d+[.)]\s+)?[А-ЯA-Z][^.!?]{2,140}$")
TABLE_CODE_RE = re.compile(r"^[A-ZА-Я0-9]{1,8}(?:-[A-ZА-Я0-9]{1,8})?$")
GENERIC_TABLE_HEADER_RE = re.compile(
    r"^(?:да|нет|yes|no|пример|аббревиатура|типичные приложения|полное название|example)$",
    re.IGNORECASE,
)
GENERIC_BAD_TITLE_RE = re.compile(
    r"^(?:да|нет|yes|no|пример|abbrev(?:iation)?|table|таблица|рис\.?|илл\.?)$",
    re.IGNORECASE,
)
TRANSITION_HINTS = (
    "итак",
    "таким образом",
    "следовательно",
    "в заключение",
    "подведем итог",
    "далее",
    "с другой стороны",
    "однако",
    "например",
)
DISALLOWED_ANALYSIS_TYPES = {
    "copyright",
    "dedication",
    "acknowledgements",
    "bibliography",
    "acronym_list",
    "index",
    "toc",
    "figure_caption",
    "table_caption",
    "exercise",
    "question",
    "empty_or_noise",
}


@dataclass
class LogicalBlockData:
    title: str
    order_number: int
    source_text: str
    chapter_title: str
    start_paragraph: int
    end_paragraph: int
    token_count: int
    clean_text_for_analysis: str = ""
    section_title: str = ""
    section_path: list[dict[str, Any]] | None = None
    content_types: dict[str, int] | None = None
    paragraph_records: list[dict[str, Any]] | None = None


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _normalize_paragraphs(paragraphs: list[str]) -> list[str]:
    normalized = []
    for raw in paragraphs:
        text = " ".join(str(raw).split()).strip()
        if len(text) < 20:
            continue
        normalized.append(text)
    return normalized


def _chapter_name(chapter_title: str, chapter_index: int) -> str:
    clean = " ".join((chapter_title or "").split()).strip()
    if clean:
        return clean
    return f"Глава {chapter_index}"


def _boundary_score(
    prev_paragraph: str,
    curr_paragraph: str,
    prev_vector: list[float],
    curr_vector: list[float],
) -> float:
    sim = cosine_similarity(prev_vector, curr_vector)
    score = (1.0 - sim) * 0.8

    current_lower = curr_paragraph.lower()
    if any(current_lower.startswith(hint) for hint in TRANSITION_HINTS):
        score += 0.2

    if TITLE_HINT_RE.match(curr_paragraph):
        score += 0.5

    if len(curr_paragraph) < 80:
        score += 0.05

    return score


def _split_chapter_paragraphs(
    paragraphs: list[str],
    *,
    min_words: int,
    target_words: int,
    max_words: int,
) -> list[list[str]]:
    if not paragraphs:
        return []

    vectors = [create_embedding(item[:2200]) for item in paragraphs]
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_words = 0

    for index, paragraph in enumerate(paragraphs):
        paragraph_words = count_words(paragraph)
        should_cut = False

        if current_chunk:
            previous_text = paragraphs[index - 1]
            previous_vector = vectors[index - 1]
            current_vector = vectors[index]
            score = _boundary_score(previous_text, paragraph, previous_vector, current_vector)

            if current_words >= min_words and score >= 0.34:
                should_cut = True
            if current_words >= target_words and score >= 0.25:
                should_cut = True
            if current_words + paragraph_words > max_words:
                should_cut = True

        if should_cut and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_words = 0

        current_chunk.append(paragraph)
        current_words += paragraph_words

    if current_chunk:
        chunks.append(current_chunk)

    merged: list[list[str]] = []
    for chunk in chunks:
        chunk_words = sum(count_words(item) for item in chunk)
        if merged and chunk_words < max(80, min_words // 2):
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)
    return merged


def _split_by_boundaries(paragraphs: list[str], boundaries: list[int]) -> list[list[str]]:
    if not paragraphs:
        return []
    clean_boundaries = sorted({item for item in boundaries if 1 <= item <= len(paragraphs)})
    if not clean_boundaries or clean_boundaries[-1] != len(paragraphs):
        clean_boundaries.append(len(paragraphs))

    chunks: list[list[str]] = []
    start = 0
    for end in clean_boundaries:
        if end <= start:
            continue
        chunk = paragraphs[start:end]
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _chunk_words(chunk: list[str]) -> int:
    return sum(count_words(item) for item in chunk)


def _merge_tiny_chunks(chunks: list[list[str]], min_words: int) -> list[list[str]]:
    merged: list[list[str]] = []
    tiny_threshold = max(40, min_words // 2)
    for chunk in chunks:
        words = _chunk_words(chunk)
        if merged and words < tiny_threshold:
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)
    return merged


def _llm_chunks_are_valid(chunks: list[list[str]], min_words: int, max_words: int) -> bool:
    if not chunks:
        return False
    too_large = 0
    too_small = 0
    for chunk in chunks:
        words = _chunk_words(chunk)
        if words > int(max_words * 1.4):
            too_large += 1
        if words < max(35, int(min_words * 0.4)):
            too_small += 1
    if too_large > 0:
        return False
    if too_small > max(1, len(chunks) // 2):
        return False
    return True


def split_into_logical_blocks(
    parsed_book,
    *,
    min_words: int = 260,
    target_words: int = 760,
    max_words: int = 1300,
) -> list[LogicalBlockData]:
    """Classic splitter preserved for backward compatibility."""
    chapters = parsed_book.chapters or []
    blocks: list[LogicalBlockData] = []
    if not chapters:
        return blocks

    order_number = 1
    global_paragraph_index = 0
    llm_boundary_calls = 0
    llm_boundary_enabled = os.getenv("BLOCK_LLM_BOUNDARY_ENABLED", "1").lower() in {"1", "true", "yes"}
    llm_boundary_max_calls = int(os.getenv("BLOCK_LLM_BOUNDARY_MAX_CALLS", "8"))

    for chapter_index, chapter in enumerate(chapters, start=1):
        chapter_title = _chapter_name(str(chapter.get("chapter_title", "")), chapter_index)
        raw_records = chapter_paragraph_records(chapter)
        paragraphs = _normalize_paragraphs([item.get("text", "") for item in raw_records])
        if not paragraphs:
            continue

        semantic_chunks = _split_chapter_paragraphs(
            paragraphs,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
        )

        can_call_llm = (
            llm_boundary_enabled
            and llm_boundary_calls < llm_boundary_max_calls
            and len(paragraphs) >= 6
            and len(paragraphs) <= 80
        )
        llm_boundaries: list[int] = []
        if can_call_llm:
            llm_boundary_calls += 1
            llm_boundaries = suggest_chapter_boundaries(
                chapter_title,
                paragraphs,
                min_words=min_words,
                target_words=target_words,
                max_words=max_words,
            )

        if llm_boundaries:
            llm_chunks = _split_by_boundaries(paragraphs, llm_boundaries)
            repaired_chunks: list[list[str]] = []
            for chunk in llm_chunks:
                if _chunk_words(chunk) > max_words:
                    repaired_chunks.extend(
                        _split_chapter_paragraphs(
                            chunk,
                            min_words=min_words,
                            target_words=target_words,
                            max_words=max_words,
                        )
                    )
                else:
                    repaired_chunks.append(chunk)
            repaired_chunks = _merge_tiny_chunks(repaired_chunks, min_words=min_words)
            if _llm_chunks_are_valid(repaired_chunks, min_words=min_words, max_words=max_words):
                semantic_chunks = repaired_chunks

        for local_index, chunk in enumerate(semantic_chunks, start=1):
            start_paragraph = global_paragraph_index + 1
            global_paragraph_index += len(chunk)
            end_paragraph = global_paragraph_index
            source_text = "\n\n".join(chunk)
            token_count = count_words(source_text)
            title = chapter_title
            if len(semantic_chunks) > 1:
                title = f"{chapter_title} - блок {local_index}"

            blocks.append(
                LogicalBlockData(
                    title=title,
                    order_number=order_number,
                    source_text=source_text,
                    chapter_title=chapter_title,
                    start_paragraph=start_paragraph,
                    end_paragraph=end_paragraph,
                    token_count=token_count,
                )
            )
            order_number += 1

    if not blocks:
        all_paragraphs: list[str] = []
        for chapter in chapters:
            all_paragraphs.extend(_normalize_paragraphs([item.get("text", "") for item in chapter_paragraph_records(chapter)]))

        semantic_chunks = _split_chapter_paragraphs(
            all_paragraphs,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
        )
        paragraph_cursor = 1
        for local_index, chunk in enumerate(semantic_chunks, start=1):
            source_text = "\n\n".join(chunk)
            token_count = count_words(source_text)
            start_paragraph = paragraph_cursor
            paragraph_cursor += len(chunk)
            end_paragraph = paragraph_cursor - 1
            blocks.append(
                LogicalBlockData(
                    title=f"Логический блок {local_index}",
                    order_number=local_index,
                    source_text=source_text,
                    chapter_title="Основной текст",
                    start_paragraph=start_paragraph,
                    end_paragraph=end_paragraph,
                    token_count=token_count,
                )
            )
    return blocks


def _choose_block_title(chapter_title: str, chunk_rows: list[dict[str, Any]], chunk_index: int) -> str:
    for row in chunk_rows:
        text = " ".join(str(row.get("text", "")).split()).strip()
        if row["content_type"] in {"title", "subtitle"} and len(text.split()) <= 16 and not _is_weak_heading_title(text):
            return text[:512]
    if chunk_index > 1:
        return f"{chapter_title} - блок {chunk_index}"[:512]
    return chapter_title[:512]


def _clean_text_from_rows(rows: list[dict[str, Any]], *, mode: str) -> str:
    filtered = filter_content_for_analysis(rows, mode=mode)
    return "\n".join(item["text"] for item in filtered["kept"] if item.get("text")).strip()


def _is_weak_heading_title(text: str) -> bool:
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return True
    if GENERIC_BAD_TITLE_RE.match(clean):
        return True
    if len(clean) <= 2:
        return True
    if TABLE_CODE_RE.match(clean):
        return True
    return False


def _is_table_like_row(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get("text", "")).split()).strip()
    if not text:
        return True
    row_type = str(row.get("content_type", ""))
    words = text.split()

    if row_type in {"figure_caption", "table_caption"}:
        return True
    if row_type == "subtitle" and (TABLE_CODE_RE.match(text) or GENERIC_TABLE_HEADER_RE.match(text)):
        return True
    if row_type == "subtitle" and len(words) <= 2 and not re.search(r"[.!?]$", text):
        return True
    if len(words) <= 3 and (TABLE_CODE_RE.match(text) or GENERIC_TABLE_HEADER_RE.match(text)):
        return True
    if row_type == "main_text" and len(words) <= 7 and not re.search(r"[.!?]$", text):
        # Typical table cells like "Business-to-consumer", "Ноутбук в номере отеля".
        if not any(marker in text.lower() for marker in TRANSITION_HINTS):
            return True
    return False


def _compute_table_zone_mask(rows: list[dict[str, Any]]) -> list[bool]:
    raw = [_is_table_like_row(row) for row in rows]
    if not raw:
        return raw
    result = [False] * len(raw)
    for idx, is_raw in enumerate(raw):
        if not is_raw:
            continue
        row_type = str(rows[idx].get("content_type", ""))
        if row_type in {"figure_caption", "table_caption"}:
            result[idx] = True
            continue
        left = max(0, idx - 1)
        right = min(len(raw), idx + 2)
        window = raw[left:right]
        # Keep only clustered table-like rows; suppress isolated short subtitles.
        if sum(1 for item in window if item) >= 2:
            result[idx] = True
    return result


def _chunk_clean_words(rows: list[dict[str, Any]]) -> int:
    return count_words(_clean_text_from_rows(rows, mode="themes"))


def _chunk_table_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    table_rows = sum(1 for row in rows if _is_table_like_row(row))
    return table_rows / len(rows)


def split_into_logical_blocks_improved(
    parsed_book,
    *,
    min_words: int = 220,
    target_words: int = 700,
    max_words: int = 1200,
) -> tuple[list[LogicalBlockData], dict[str, Any]]:
    """Improved universal splitter based on real sections + content filtering."""

    chapters = parsed_book.chapters or []
    blocks: list[LogicalBlockData] = []
    order_number = 1

    total_stats: Counter[str] = Counter()
    filtered_out_block_candidates = 0
    filtered_candidates: list[dict[str, Any]] = []

    for chapter_index, chapter in enumerate(chapters, start=1):
        chapter_title = _chapter_name(str(chapter.get("chapter_title", "")), chapter_index)
        section_title = chapter_title
        section_path = chapter.get("section_path") or []
        front_matter_section = is_front_matter_title(chapter_title) or is_front_matter_title(section_title)
        records = chapter_paragraph_records(chapter)
        if not records:
            continue

        block_filter = filter_content_for_analysis(
            records,
            mode="blocks",
            chapter_title=chapter_title,
            section_title=section_title,
            allow_dialogue=True,
        )
        classified = block_filter["classified"]
        total_stats.update(block_filter["stats"].get("by_type", {}))

        if not classified:
            continue

        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        clean_words = 0
        table_zone_mask = _compute_table_zone_mask(classified)
        current_chunk_table_rows = 0

        for row_index, row in enumerate(classified):
            row_words = count_words(row["text"])
            row_type = row["content_type"]
            row_is_table_zone = bool(table_zone_mask[row_index])
            prev_is_table_zone = bool(table_zone_mask[row_index - 1]) if row_index > 0 else False
            starts_new_section = (
                row_type in {"title", "subtitle"}
                and not row_is_table_zone
                and not _is_weak_heading_title(str(row.get("text", "")))
                and current_chunk
            )
            too_big = clean_words >= min_words and clean_words + row_words > max_words
            starts_table_zone = (
                row_is_table_zone
                and not prev_is_table_zone
                and current_chunk
                and clean_words >= max(90, min_words // 3)
            )
            ends_table_zone = (
                not row_is_table_zone
                and prev_is_table_zone
                and current_chunk
                and current_chunk_table_rows >= 2
                and clean_words >= max(60, min_words // 4)
            )

            if starts_new_section or too_big or starts_table_zone or ends_table_zone:
                chunks.append(current_chunk)
                current_chunk = []
                clean_words = 0
                current_chunk_table_rows = 0

            current_chunk.append(row)
            if row_is_table_zone:
                current_chunk_table_rows += 1
            if row_type not in DISALLOWED_ANALYSIS_TYPES:
                clean_words += row_words

        if current_chunk:
            chunks.append(current_chunk)

        # Merge tiny tails.
        merged_chunks: list[list[dict[str, Any]]] = []
        for chunk in chunks:
            chunk_clean_words = _chunk_clean_words(chunk)
            chunk_table_ratio = _chunk_table_ratio(chunk)
            if (
                merged_chunks
                and chunk_clean_words < max(40, min_words // 2)
                and chunk_table_ratio < 0.45
            ):
                merged_chunks[-1].extend(chunk)
            else:
                merged_chunks.append(chunk)

        # Merge tiny edge chunks with neighbors to avoid 1-2 sentence fragments.
        tiny_threshold = max(65, min_words // 3)
        if len(merged_chunks) > 1 and _chunk_clean_words(merged_chunks[0]) < tiny_threshold:
            merged_chunks[1] = merged_chunks[0] + merged_chunks[1]
            merged_chunks = merged_chunks[1:]

        compacted_chunks: list[list[dict[str, Any]]] = []
        for chunk in merged_chunks:
            if not compacted_chunks:
                compacted_chunks.append(chunk)
                continue
            if _chunk_clean_words(chunk) < tiny_threshold and _chunk_table_ratio(chunk) < 0.45:
                compacted_chunks[-1].extend(chunk)
            else:
                compacted_chunks.append(chunk)
        merged_chunks = compacted_chunks

        # Media-caption chunks should not become standalone block starts.
        normalized_chunks: list[list[dict[str, Any]]] = []
        for chunk in merged_chunks:
            if not chunk:
                continue
            first_type = str(chunk[0].get("content_type", ""))
            if first_type in {"figure_caption", "table_caption"} and normalized_chunks:
                normalized_chunks[-1].extend(chunk)
            else:
                normalized_chunks.append(chunk)
        merged_chunks = normalized_chunks

        # Avoid finishing a block with a pure media caption; move it to the next chunk.
        for idx in range(len(merged_chunks) - 1):
            current = merged_chunks[idx]
            nxt = merged_chunks[idx + 1]
            moved_tail: list[dict[str, Any]] = []
            while (
                len(current) > 1
                and str(current[-1].get("content_type", "")) in {"figure_caption", "table_caption"}
            ):
                moved_tail.insert(0, current.pop())
            if moved_tail:
                merged_chunks[idx + 1] = moved_tail + nxt

        for chunk_index, chunk_rows in enumerate(merged_chunks, start=1):
            source_text = "\n\n".join(row["text"] for row in chunk_rows if row.get("text")).strip()
            clean_text_for_analysis = _clean_text_from_rows(chunk_rows, mode="concepts")
            para_indexes = [int(row["paragraph_index"]) for row in chunk_rows if row.get("paragraph_index") is not None]
            start_para = min(para_indexes) if para_indexes else 0
            end_para = max(para_indexes) if para_indexes else 0

            if not source_text:
                filtered_candidates.append(
                    {
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "chunk_index": chunk_index,
                        "reason": "empty_source_text",
                        "start_paragraph": start_para,
                        "end_paragraph": end_para,
                    }
                )
                continue
            if not clean_text_for_analysis:
                filtered_out_block_candidates += 1
                filtered_candidates.append(
                    {
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "chunk_index": chunk_index,
                        "reason": "empty_clean_text_for_analysis",
                        "start_paragraph": start_para,
                        "end_paragraph": end_para,
                    }
                )
                continue
            if front_matter_section and count_words(clean_text_for_analysis) < 180:
                filtered_out_block_candidates += 1
                filtered_candidates.append(
                    {
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "chunk_index": chunk_index,
                        "reason": "front_matter_short_clean_text",
                        "start_paragraph": start_para,
                        "end_paragraph": end_para,
                    }
                )
                continue

            content_counts = Counter(row["content_type"] for row in chunk_rows)
            table_ratio = _chunk_table_ratio(chunk_rows)
            dominant_noise = sum(content_counts[item] for item in DISALLOWED_ANALYSIS_TYPES)
            if dominant_noise >= len(chunk_rows) and count_words(clean_text_for_analysis) < 30:
                filtered_out_block_candidates += 1
                filtered_candidates.append(
                    {
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "chunk_index": chunk_index,
                        "reason": "noise_dominated_chunk",
                        "start_paragraph": start_para,
                        "end_paragraph": end_para,
                        "content_types": dict(content_counts),
                    }
                )
                continue
            if table_ratio >= 0.62 and count_words(clean_text_for_analysis) < max(180, min_words):
                filtered_out_block_candidates += 1
                filtered_candidates.append(
                    {
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "chunk_index": chunk_index,
                        "reason": "table_like_chunk_for_analysis",
                        "start_paragraph": start_para,
                        "end_paragraph": end_para,
                        "content_types": dict(content_counts),
                    }
                )
                continue

            if not para_indexes:
                filtered_candidates.append(
                    {
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "chunk_index": chunk_index,
                        "reason": "no_paragraph_indexes",
                    }
                )
                continue

            title = _choose_block_title(chapter_title, chunk_rows, chunk_index)
            block = LogicalBlockData(
                title=title,
                order_number=order_number,
                source_text=source_text,
                chapter_title=chapter_title,
                start_paragraph=min(para_indexes),
                end_paragraph=max(para_indexes),
                token_count=count_words(source_text),
                clean_text_for_analysis=clean_text_for_analysis,
                section_title=section_title,
                section_path=section_path,
                content_types=dict(content_counts),
                paragraph_records=chunk_rows,
            )
            blocks.append(block)
            order_number += 1

    diagnostics = {
        "splitter": "classic_improved_v2",
        "total_blocks": len(blocks),
        "filtered_out_block_candidates": filtered_out_block_candidates,
        "content_types_total": dict(total_stats),
        "filtered_candidates": filtered_candidates[:3000],
    }

    if not blocks:
        fallback_blocks = split_into_logical_blocks(
            parsed_book,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
        )
        diagnostics["fallback_to_classic"] = True
        diagnostics["total_blocks"] = len(fallback_blocks)
        return fallback_blocks, diagnostics

    diagnostics["fallback_to_classic"] = False
    return blocks, diagnostics
