from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from razdel import sentenize

from apps.books.services.sentence_segmenter import segment_book_sentences
from apps.books.services.structure_detector import build_canonical_outline


@dataclass(slots=True)
class ThoughtChainSentence:
    index: int
    text: str
    chapter_title: str = ""
    section_title: str = ""
    paragraph_index: int = 0
    source_start: int | None = None
    source_end: int | None = None


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _append_section_sentences(
    result: list[ThoughtChainSentence],
    *,
    section: Any,
    max_sentences: int | None,
) -> bool:
    chapter_title = _normalize_text(
        getattr(section, "parent_chapter_title", "") or getattr(section, "chapter_title", "") or getattr(section, "section_title", "")
    )
    section_title = _normalize_text(getattr(section, "section_title", "") or chapter_title)
    records = list(getattr(section, "paragraphs", []) or [])
    for record in records:
        paragraph_text = _normalize_text(str(record.get("text", "")))
        if not paragraph_text:
            continue
        paragraph_index = max(0, int(record.get("paragraph_index") or 0))
        found = False
        for sentence in sentenize(paragraph_text):
            text = _normalize_text(sentence.text)
            if len(text) < 2:
                continue
            found = True
            result.append(
                ThoughtChainSentence(
                    index=len(result) + 1,
                    text=text,
                    chapter_title=chapter_title,
                    section_title=section_title,
                    paragraph_index=paragraph_index,
                    source_start=getattr(sentence, "start", None),
                    source_end=getattr(sentence, "stop", None),
                )
            )
            if max_sentences and len(result) >= max_sentences:
                return True
        if not found:
            result.append(
                ThoughtChainSentence(
                    index=len(result) + 1,
                    text=paragraph_text,
                    chapter_title=chapter_title,
                    section_title=section_title,
                    paragraph_index=paragraph_index,
                )
            )
            if max_sentences and len(result) >= max_sentences:
                return True
    return False


def split_book_into_sentences(parsed_book: Any, *, max_sentences: int | None = None) -> list[ThoughtChainSentence]:
    """Return source-aware main-content sentences for thought-chain analysis."""

    outline = build_canonical_outline(parsed_book)
    main_sections = [item for item in outline.get("sections", []) if getattr(item, "is_main_content", False)]
    result: list[ThoughtChainSentence] = []
    if main_sections:
        for section in main_sections:
            if _append_section_sentences(result, section=section, max_sentences=max_sentences):
                return result
    else:
        source_sentences = segment_book_sentences(parsed_book)
        for item in source_sentences:
            text = _normalize_text(item.text or "")
            if not text:
                continue
            result.append(
                ThoughtChainSentence(
                    index=len(result) + 1,
                    text=text,
                    chapter_title=item.chapter_title or "",
                    section_title=item.chapter_title or "",
                    paragraph_index=max(0, int(item.paragraph_index or 0)),
                    source_start=item.start,
                    source_end=item.stop,
                )
            )
            if max_sentences and len(result) >= max_sentences:
                break
    return result
