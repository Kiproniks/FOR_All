from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from apps.books.services.book_parser import chapter_paragraph_records

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")

CHAPTER_LIKE_RE = re.compile(
    r"^(?:глава\s+\d+|chapter\s+\d+|часть\s+[ivxlcdm\d]+|part\s+[ivxlcdm\d]+|раздел\s+\d+|section\s+\d+|лекция\s+\d+)\b",
    re.IGNORECASE,
)
SUBSECTION_RE = re.compile(r"^\d+(?:\.\d+){1,5}\.?\s*")
SECTION_NUMBER_RE = re.compile(r"^\d+\.\s+")
CHAPTER_NUMBER_RE = re.compile(
    r"^(?:глава|chapter|лекция|lecture|раздел|section)\s+(\d+)\b",
    re.IGNORECASE,
)
TOP_NUMBER_RE = re.compile(r"^(\d+)\.\s+")
DECIMAL_NUMBER_RE = re.compile(r"^(\d+)(?:\.\d+){1,5}\.?\s+")

FRONT_MATTER_RE = re.compile(
    r"(?:предисловие|foreword|acknowledg|благодарност|об авторах|about the author|от издательства|publisher|"
    r"переводчик|copyright|all rights reserved|isbn|посвящени|annotation|аннотац|список аббревиатур|"
    r"материалы для студентов|введение к изданию)",
    re.IGNORECASE,
)
TOC_RE = re.compile(r"(?:содержание|оглавление|table of contents)", re.IGNORECASE)
BIB_RE = re.compile(r"(?:библиограф|литератур|references|bibliography)", re.IGNORECASE)
INDEX_RE = re.compile(r"(?:предметный указатель|index)", re.IGNORECASE)
EXERCISE_RE = re.compile(r"(?:вопросы и задачи|задачи|упражнения|questions and exercises|exercises)", re.IGNORECASE)
SUMMARY_RE = re.compile(r"(?:резюме|summary|заключение|conclusion)", re.IGNORECASE)
ABBR_RE = re.compile(r"(?:аббревиатур|abbreviations|acronyms)", re.IGNORECASE)
PREFACE_RE = re.compile(r"(?:предисловие|foreword|preface)", re.IGNORECASE)


@dataclass(slots=True)
class CanonicalSection:
    section_index: int
    chapter_title: str
    section_title: str
    parent_chapter_title: str
    content_type: str
    is_main_content: bool
    start_paragraph: int
    end_paragraph: int
    word_count: int
    paragraphs: list[dict[str, Any]]
    section_path: list[dict[str, Any]]
    level: int


def _normalize(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _title_level(title: str) -> int:
    value = _normalize(title)
    if not value:
        return 1
    if CHAPTER_LIKE_RE.match(value):
        return 1
    if SUBSECTION_RE.match(value):
        dots = value.split(" ", 1)[0].count(".")
        return min(4, dots + 1)
    if SECTION_NUMBER_RE.match(value):
        return 2
    return 2


def _section_type_from_title(title: str, *, near_end: bool) -> str:
    value = _normalize(title)
    if not value:
        return "unknown"
    if TOC_RE.search(value):
        return "toc"
    if PREFACE_RE.search(value):
        return "preface"
    if ABBR_RE.search(value):
        return "abbreviation_list"
    if EXERCISE_RE.search(value):
        return "exercises"
    if SUMMARY_RE.search(value):
        return "summary"
    if BIB_RE.search(value):
        return "bibliography"
    if INDEX_RE.search(value):
        return "index"
    if FRONT_MATTER_RE.search(value):
        if "об авторах" in value.lower() or "about the author" in value.lower():
            return "author_bio"
        if "благодар" in value.lower() or "acknowledg" in value.lower():
            return "acknowledgements"
        if "издатель" in value.lower() or "publisher" in value.lower():
            return "publisher_note"
        if "copyright" in value.lower() or "isbn" in value.lower():
            return "copyright"
        return "front_matter"
    if near_end and (BIB_RE.search(value) or INDEX_RE.search(value)):
        return "back_matter"
    return "main_content"


def _parent_chapter_title(section_title: str, section_path: list[dict[str, Any]]) -> str:
    if section_path:
        top = section_path[0]
        if isinstance(top, dict):
            title = _normalize(str(top.get("title", "")))
            if title:
                return title
    return section_title


def _chapter_number_from_title(title: str) -> int | None:
    value = _normalize(title)
    match = CHAPTER_NUMBER_RE.match(value)
    if match:
        return int(match.group(1))
    match = TOP_NUMBER_RE.match(value)
    if match and not DECIMAL_NUMBER_RE.match(value):
        return int(match.group(1))
    return None


def _decimal_parent_number(title: str) -> int | None:
    match = DECIMAL_NUMBER_RE.match(_normalize(title))
    return int(match.group(1)) if match else None


def _repair_parent_chapters(sections: list[CanonicalSection]) -> None:
    """Recover chapter grouping when FB2/PDF flattening loses nested sections."""
    chapter_by_number: dict[int, str] = {}
    for section in sections:
        number = _chapter_number_from_title(section.section_title)
        if number is not None:
            chapter_by_number.setdefault(number, section.section_title)

    if not chapter_by_number:
        return

    for section in sections:
        parent_number = _decimal_parent_number(section.section_title)
        if parent_number is None:
            continue
        parent_title = chapter_by_number.get(parent_number)
        if parent_title:
            section.parent_chapter_title = parent_title


def _looks_like_short_trailing_note(section: CanonicalSection) -> bool:
    title = _normalize(section.section_title)
    if not title:
        return False
    if CHAPTER_LIKE_RE.match(title) or SUBSECTION_RE.match(title) or SECTION_NUMBER_RE.match(title):
        return False
    words = title.replace(",", " ").split()
    if len(words) > 6:
        return False
    capitalized = sum(1 for word in words if word[:1].isupper())
    return section.word_count <= 350 and capitalized >= max(1, len(words) - 1)


def _repair_trailing_back_matter(sections: list[CanonicalSection]) -> None:
    seen_back_matter = False
    trailing_start = max(1, len(sections) - 10)
    for section in sections:
        if section.content_type in {"bibliography", "index", "back_matter"}:
            seen_back_matter = True
            continue
        near_end = section.section_index >= trailing_start
        if near_end and seen_back_matter and section.content_type == "main_content":
            if _looks_like_short_trailing_note(section):
                section.content_type = "back_matter"
                section.is_main_content = False


def build_canonical_outline(parsed_book) -> dict[str, Any]:
    chapters = parsed_book.chapters or []
    sections: list[CanonicalSection] = []

    for index, raw_section in enumerate(chapters, start=1):
        title = _normalize(str(raw_section.get("chapter_title", ""))) or f"Section {index}"
        records = chapter_paragraph_records(raw_section)
        if not records:
            continue
        section_path = raw_section.get("section_path") or []
        para_indexes = [int(item.get("paragraph_index", 0) or 0) for item in records]
        start_paragraph = min((value for value in para_indexes if value > 0), default=0)
        end_paragraph = max(para_indexes, default=0)
        text = "\n".join(str(item.get("text", "")) for item in records)
        word_count = _words(text)

        near_end = index >= max(1, len(chapters) - 3)
        content_type = _section_type_from_title(title, near_end=near_end)
        is_main = content_type in {"main_content", "summary", "exercises", "questions"}
        if content_type in {
            "front_matter",
            "preface",
            "acknowledgements",
            "author_bio",
            "publisher_note",
            "copyright",
            "abbreviation_list",
            "toc",
            "bibliography",
            "index",
            "back_matter",
        }:
            is_main = False

        sections.append(
            CanonicalSection(
                section_index=index,
                chapter_title=title,
                section_title=title,
                parent_chapter_title=_parent_chapter_title(title, section_path),
                content_type=content_type,
                is_main_content=is_main,
                start_paragraph=start_paragraph,
                end_paragraph=end_paragraph,
                word_count=word_count,
                paragraphs=records,
                section_path=section_path,
                level=_title_level(title),
            )
        )

    # Extra pass: if first chapter-like sections exist, ignore earlier unknown/front matter.
    chapter_like_positions = [
        item.section_index
        for item in sections
        if CHAPTER_LIKE_RE.match(item.section_title) or SUBSECTION_RE.match(item.section_title)
    ]
    main_content_start = chapter_like_positions[0] if chapter_like_positions else 1
    if chapter_like_positions:
        for item in sections:
            if item.section_index < main_content_start and item.content_type == "main_content":
                item.is_main_content = False
                item.content_type = "front_matter"

    _repair_parent_chapters(sections)
    _repair_trailing_back_matter(sections)

    main_sections = [item for item in sections if item.is_main_content]
    filtered_sections = [item for item in sections if not item.is_main_content]

    chapters_map: dict[str, list[CanonicalSection]] = {}
    for section in main_sections:
        chapters_map.setdefault(section.parent_chapter_title, []).append(section)

    canonical_chapters = []
    for chapter_index, (chapter_title, section_items) in enumerate(chapters_map.items(), start=1):
        canonical_chapters.append(
            {
                "chapter_index": chapter_index,
                "chapter_title": chapter_title,
                "sections": [
                    {
                        "section_index": item.section_index,
                        "section_title": item.section_title,
                        "content_type": item.content_type,
                        "start_paragraph": item.start_paragraph,
                        "end_paragraph": item.end_paragraph,
                        "word_count": item.word_count,
                        "level": item.level,
                    }
                    for item in section_items
                ],
            }
        )

    return {
        "main_content_start": main_content_start,
        "sections_total": len(sections),
        "main_sections_count": len(main_sections),
        "filtered_sections_count": len(filtered_sections),
        "sections": sections,
        "canonical_chapters": canonical_chapters,
        "stats": {
            "by_type": _count_by_type(sections),
            "total_words_main": sum(item.word_count for item in main_sections),
            "total_words_filtered": sum(item.word_count for item in filtered_sections),
        },
    }


def _count_by_type(sections: list[CanonicalSection]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in sections:
        result[item.content_type] = result.get(item.content_type, 0) + 1
    return result
