from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from razdel import sentenize

from apps.books.services.book_parser import chapter_paragraph_records


@dataclass(slots=True)
class SourceSentence:
    id: str
    chapter_title: str
    paragraph_index: int
    sentence_index: int
    text: str
    start: int | None = None
    stop: int | None = None


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _chapter_title(raw: str, chapter_number: int) -> str:
    title = _normalize_text(raw)
    return title or f"Chapter {chapter_number}"


def segment_book_sentences(parsed_book: Any) -> list[SourceSentence]:
    """Split parsed-book chapters/paragraphs into source-aware sentences."""

    chapters = list(getattr(parsed_book, "chapters", []) or [])
    if not chapters:
        return []

    result: list[SourceSentence] = []
    sentence_counter = 1
    paragraph_counter = 0

    for chapter_number, chapter in enumerate(chapters, start=1):
        chapter_title = _chapter_title(str(chapter.get("chapter_title", "")), chapter_number)
        paragraph_records = chapter_paragraph_records(chapter)
        for local_paragraph_order, record in enumerate(paragraph_records, start=1):
            paragraph_text = _normalize_text(str(record.get("text", "")))
            if not paragraph_text:
                continue

            record_index = int(record.get("paragraph_index") or 0)
            if record_index > 0:
                paragraph_counter = max(paragraph_counter, record_index)
            else:
                paragraph_counter += 1

            sentence_index = 0
            found = False
            for sentence in sentenize(paragraph_text):
                text = _normalize_text(sentence.text)
                if len(text) < 2:
                    continue
                sentence_index += 1
                found = True
                result.append(
                    SourceSentence(
                        id=f"s{sentence_counter}",
                        chapter_title=chapter_title,
                        paragraph_index=paragraph_counter,
                        sentence_index=sentence_index,
                        text=text,
                        start=getattr(sentence, "start", None),
                        stop=getattr(sentence, "stop", None),
                    )
                )
                sentence_counter += 1

            if not found:
                result.append(
                    SourceSentence(
                        id=f"s{sentence_counter}",
                        chapter_title=chapter_title,
                        paragraph_index=paragraph_counter,
                        sentence_index=1,
                        text=paragraph_text,
                    )
                )
                sentence_counter += 1

    return result
