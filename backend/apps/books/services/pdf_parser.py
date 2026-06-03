from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from pypdf import PdfReader

from .fb2_parser import ParsedBook, normalize_spaces

HEADING_NUMBER_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+){0,4}[.)]?"
    r"|\u0433\u043b\u0430\u0432\u0430\s+\d+"
    r"|\u0447\u0430\u0441\u0442\u044c\s+[ivxlcdm\d]+"
    r"|\u0440\u0430\u0437\u0434\u0435\u043b\s+\d+"
    r"|chapter\s+\d+"
    r"|part\s+[ivxlcdm\d]+"
    r"|section\s+\d+"
    r")\b",
    re.IGNORECASE,
)
ROMAN_HEADING_RE = re.compile(r"^[IVXLCDM]+(?:[.)]|\b)", re.IGNORECASE)
ALL_CAPS_RE = re.compile(r"^[A-ZА-ЯЁ\d\-\s]{4,}$")
LEADING_HEADING_RE = re.compile(
    r"^(?P<prefix>(?:\u0433\u043b\u0430\u0432\u0430|\u0447\u0430\u0441\u0442\u044c|\u0440\u0430\u0437\u0434\u0435\u043b|chapter|part|section)\s+[IVXLCDM\d]+(?:[.)]?))\s+(?P<rest>.+)$",
    re.IGNORECASE,
)


def _extract_page_paragraphs(page_text: str) -> list[str]:
    raw = (page_text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return []

    chunks = [chunk for chunk in re.split(r"\n{2,}", raw) if chunk.strip()]
    paragraphs: list[str] = []
    for chunk in chunks:
        merged = normalize_spaces(chunk.replace("\n", " "))
        if len(merged) >= 10:
            paragraphs.append(merged)

    if paragraphs:
        return paragraphs

    sentences = [normalize_spaces(item) for item in re.split(r"(?<=[.!?])\s+", raw) if item.strip()]
    rebuilt: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        if len(" ".join(current)) >= 220:
            rebuilt.append(normalize_spaces(" ".join(current)))
            current = []
    if current:
        rebuilt.append(normalize_spaces(" ".join(current)))
    return [item for item in rebuilt if len(item) >= 10]


def _looks_like_heading(text: str, previous_line: str = "") -> bool:
    value = normalize_spaces(text)
    if not value or len(value) > 160:
        return False
    if value.endswith(":") and len(value.split()) <= 12:
        return True
    if HEADING_NUMBER_RE.match(value):
        # PDF extraction often glues heading + paragraph into one line.
        if len(value.split()) > 9 and re.search(r"[.!?]", value):
            return False
        return True
    if ROMAN_HEADING_RE.match(value):
        return True
    if ALL_CAPS_RE.match(value) and len(value.split()) <= 12:
        return True

    words = value.split()
    if len(words) <= 8 and not re.search(r"[.!?]", value):
        # Likely heading if surrounded by short lines or after blank gap.
        if len(previous_line) < 70:
            return True
        if value.istitle():
            return True
    return False


def _title_from_first_lines(chapters: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for chapter in chapters[:3]:
        for item in chapter.get("paragraphs", [])[:4]:
            text = normalize_spaces(str(item.get("text", "")))
            if not text:
                continue
            if _looks_like_heading(text):
                candidates.append(text)
            elif 20 <= len(text) <= 120 and not any(symbol in text for symbol in ("@", "http", "ISBN", "©")):
                candidates.append(text)
            if len(candidates) >= 3:
                break
        if len(candidates) >= 3:
            break
    return candidates[0] if candidates else "PDF document"


def _split_heading_prefix(text: str) -> tuple[str, str]:
    """
    Split merged PDF line into heading prefix and remaining sentence content.
    Example:
    \"Chapter 2 Routing Routing algorithms...\" -> (\"Chapter 2 Routing\", \"Routing algorithms...\")
    """
    value = normalize_spaces(text)
    if not value:
        return "", ""

    match = LEADING_HEADING_RE.match(value)
    if not match:
        return "", value

    rest = match.group("rest").strip()
    if not rest:
        return "", value

    rest_words = rest.split()
    # Keep one additional token as part of heading title.
    heading_tail = rest_words[0]
    heading = f"{match.group('prefix').strip()} {heading_tail}".strip()
    remainder = " ".join(rest_words[1:]).strip()
    if not remainder:
        return heading, ""
    return heading, remainder


def parse_pdf(content: bytes) -> ParsedBook:
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Invalid PDF: {exc}") from exc

    metadata = reader.metadata or {}
    title = normalize_spaces(str(getattr(metadata, "title", "") or metadata.get("/Title", "")))
    authors = normalize_spaces(str(getattr(metadata, "author", "") or metadata.get("/Author", "")))

    chapters: list[dict[str, Any]] = []
    section_order = 0
    paragraph_index = 0

    current_title = "Section 1"
    current_level = 1
    current_paragraphs: list[dict[str, Any]] = []
    current_page_start = 1

    def flush_section(page_number: int) -> None:
        nonlocal current_paragraphs, current_title, current_level, section_order, current_page_start
        if not current_paragraphs:
            return
        section_order += 1
        chapters.append(
            {
                "chapter_title": current_title,
                "paragraphs": list(current_paragraphs),
                "section_path": [
                    {
                        "level": current_level,
                        "title": current_title,
                        "parent_title": "",
                        "order": section_order,
                        "paragraph_start": current_paragraphs[0]["paragraph_index"],
                        "paragraph_end": current_paragraphs[-1]["paragraph_index"],
                    }
                ],
                "section_level": current_level,
                "parent_title": "",
                "order": section_order,
                "page_start": current_page_start,
                "page_end": page_number,
            }
        )
        current_paragraphs = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        page_paragraphs = _extract_page_paragraphs(page_text)
        if not page_paragraphs:
            continue

        previous = ""
        for paragraph in page_paragraphs:
            text = normalize_spaces(paragraph)
            if not text:
                continue
            heading_prefix, remainder = _split_heading_prefix(text)
            if heading_prefix:
                flush_section(page_number - 1 if page_number > current_page_start else page_number)
                current_title = heading_prefix
                current_level = 1
                current_page_start = page_number
                text = remainder
                if not text:
                    previous = heading_prefix
                    continue

            if _looks_like_heading(text, previous_line=previous):
                flush_section(page_number - 1 if page_number > current_page_start else page_number)
                current_title = text
                current_level = 1 if HEADING_NUMBER_RE.match(text) or ROMAN_HEADING_RE.match(text) else 2
                current_page_start = page_number
                previous = text
                continue

            paragraph_index += 1
            current_paragraphs.append(
                {
                    "text": text,
                    "paragraph_index": paragraph_index,
                    "content_type_hint": "main_text",
                    "page": page_number,
                }
            )
            previous = text

    flush_section(len(reader.pages) if reader.pages else 1)

    if not chapters:
        raise ValueError("PDF has no extractable text.")

    if not title:
        title = _title_from_first_lines(chapters)

    return ParsedBook(
        title=title,
        authors=authors,
        chapters=chapters,
        metadata={
            "source_format": "pdf",
            "parser_version": "pdf_structured_v2",
            "pages_total": len(reader.pages),
            "sections_count": len(chapters),
            "paragraphs_count": paragraph_index,
        },
    )
