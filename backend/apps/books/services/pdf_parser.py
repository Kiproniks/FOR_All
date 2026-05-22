from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

from .fb2_parser import ParsedBook, normalize_spaces


def _extract_page_paragraphs(page_text: str) -> list[str]:
    raw = (page_text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return []

    # Keep page structure but avoid line-wrap noise from PDF text extraction.
    chunks = [chunk for chunk in re.split(r"\n{2,}", raw) if chunk.strip()]
    paragraphs: list[str] = []
    for chunk in chunks:
        merged = normalize_spaces(chunk.replace("\n", " "))
        if len(merged) >= 20:
            paragraphs.append(merged)

    if paragraphs:
        return paragraphs

    # Fallback: split by sentence punctuation if page has no clear paragraph breaks.
    sentences = [normalize_spaces(item) for item in re.split(r"(?<=[.!?])\s+", raw) if item.strip()]
    rebuilt: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        if len(" ".join(current)) >= 240:
            rebuilt.append(normalize_spaces(" ".join(current)))
            current = []
    if current:
        rebuilt.append(normalize_spaces(" ".join(current)))
    return [item for item in rebuilt if len(item) >= 20]


def parse_pdf(content: bytes) -> ParsedBook:
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Invalid PDF: {exc}") from exc

    metadata = reader.metadata or {}
    title = normalize_spaces(str(getattr(metadata, "title", "") or metadata.get("/Title", ""))) or "PDF document"
    authors = normalize_spaces(str(getattr(metadata, "author", "") or metadata.get("/Author", "")))

    chapters: list[dict[str, object]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        paragraphs = _extract_page_paragraphs(page_text)
        if not paragraphs:
            continue
        chapters.append(
            {
                "chapter_title": f"Page {index}",
                "paragraphs": paragraphs,
            }
        )

    if not chapters:
        raise ValueError("PDF has no extractable text.")

    return ParsedBook(
        title=title,
        authors=authors,
        chapters=chapters,
        metadata={
            "source_format": "pdf",
            "pages_total": len(reader.pages),
            "pages_with_text": len(chapters),
        },
    )

