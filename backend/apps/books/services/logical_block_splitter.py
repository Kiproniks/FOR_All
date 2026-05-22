from __future__ import annotations

import os
import re
from dataclasses import dataclass

from apps.books.services.llm_service import suggest_chapter_boundaries
from apps.books.services.rag_service import cosine_similarity, create_embedding

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
TITLE_HINT_RE = re.compile(r"^\s*(?:\d+[.)]\s+)?[А-ЯA-Z][^.!?]{2,120}$")
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


@dataclass
class LogicalBlockData:
    title: str
    order_number: int
    source_text: str
    chapter_title: str
    start_paragraph: int
    end_paragraph: int
    token_count: int


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


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

    # Merge too small tail chunks with previous piece to keep complete thoughts.
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
        chapter_title = _chapter_name(chapter.get("chapter_title"), chapter_index)
        paragraphs = _normalize_paragraphs(chapter.get("paragraphs", []))
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
        llm_boundaries = []
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
                title = f"{chapter_title} — блок {local_index}"

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

    # Fallback if chapter structure was broken but parser still returned text.
    if not blocks:
        all_paragraphs = []
        for chapter in chapters:
            all_paragraphs.extend(_normalize_paragraphs(chapter.get("paragraphs", [])))
        semantic_chunks = _split_chapter_paragraphs(
            all_paragraphs,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
        )
        for local_index, chunk in enumerate(semantic_chunks, start=1):
            source_text = "\n\n".join(chunk)
            token_count = count_words(source_text)
            start_paragraph = sum(len(semantic_chunks[idx]) for idx in range(local_index - 1)) + 1
            end_paragraph = start_paragraph + len(chunk) - 1
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
