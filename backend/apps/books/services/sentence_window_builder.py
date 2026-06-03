from __future__ import annotations

import re
from dataclasses import dataclass

from apps.books.services.sentence_segmenter import SourceSentence

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")


@dataclass(slots=True)
class SentenceWindow:
    id: str
    chapter_title: str
    sentence_ids: list[str]
    text: str
    start_sentence_index: int
    end_sentence_index: int


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _group_by_chapter(sentences: list[SourceSentence]) -> dict[str, list[SourceSentence]]:
    grouped: dict[str, list[SourceSentence]] = {}
    for sentence in sentences:
        grouped.setdefault(sentence.chapter_title, []).append(sentence)
    return grouped


def build_sentence_windows(
    sentences: list[SourceSentence],
    *,
    min_sentences: int = 3,
    max_sentences: int = 7,
    overlap: int = 2,
    max_words: int = 240,
) -> list[SentenceWindow]:
    """Build overlapping windows and keep chapter boundaries intact."""

    if not sentences:
        return []

    min_sentences = max(1, min_sentences)
    max_sentences = max(min_sentences, max_sentences)
    overlap = max(0, min(overlap, max_sentences - 1))

    windows: list[SentenceWindow] = []
    window_counter = 1

    for chapter_title, chapter_sentences in _group_by_chapter(sentences).items():
        if not chapter_sentences:
            continue

        if len(chapter_sentences) <= max_sentences:
            text = " ".join(sentence.text for sentence in chapter_sentences).strip()
            windows.append(
                SentenceWindow(
                    id=f"w{window_counter}",
                    chapter_title=chapter_title,
                    sentence_ids=[sentence.id for sentence in chapter_sentences],
                    text=text,
                    start_sentence_index=1,
                    end_sentence_index=len(chapter_sentences),
                )
            )
            window_counter += 1
            continue

        position = 0
        while position < len(chapter_sentences):
            end_position = min(position + max_sentences, len(chapter_sentences))
            chunk = chapter_sentences[position:end_position]

            while len(chunk) > min_sentences and _word_count(" ".join(item.text for item in chunk)) > max_words:
                end_position -= 1
                chunk = chapter_sentences[position:end_position]

            if len(chunk) < min_sentences:
                if end_position >= len(chapter_sentences):
                    tail_start = max(0, len(chapter_sentences) - min_sentences)
                    chunk = chapter_sentences[tail_start:len(chapter_sentences)]
                    end_position = len(chapter_sentences)
                    position = tail_start
                elif windows and windows[-1].chapter_title == chapter_title:
                    break

            window_text = " ".join(sentence.text for sentence in chunk).strip()
            if window_text:
                windows.append(
                    SentenceWindow(
                        id=f"w{window_counter}",
                        chapter_title=chapter_title,
                        sentence_ids=[sentence.id for sentence in chunk],
                        text=window_text,
                        start_sentence_index=position + 1,
                        end_sentence_index=end_position,
                    )
                )
                window_counter += 1

            if end_position >= len(chapter_sentences):
                break

            step = max(1, len(chunk) - overlap)
            position += step

    deduped: list[SentenceWindow] = []
    seen = set()
    for window in windows:
        key = tuple(window.sentence_ids)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)
    return deduped
