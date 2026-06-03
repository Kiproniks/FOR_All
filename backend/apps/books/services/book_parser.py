from __future__ import annotations

import os
from typing import Any

from .fb2_parser import ParsedBook, parse_fb2
from .pdf_parser import parse_pdf

SUPPORTED_BOOK_EXTENSIONS = {".fb2", ".pdf"}


def get_file_extension(filename: str | None) -> str:
    return os.path.splitext((filename or "").strip().lower())[1]


def is_supported_book_extension(filename: str | None) -> bool:
    return get_file_extension(filename) in SUPPORTED_BOOK_EXTENSIONS


def supported_extensions_text() -> str:
    return ", ".join(sorted(SUPPORTED_BOOK_EXTENSIONS))


def parse_uploaded_book(content: bytes, filename: str | None) -> ParsedBook:
    ext = get_file_extension(filename)
    if ext == ".fb2":
        return parse_fb2(content)
    if ext == ".pdf":
        return parse_pdf(content)
    raise ValueError(f"Unsupported file extension: {ext or '<none>'}. Supported: {supported_extensions_text()}.")


def chapter_paragraph_records(chapter: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalize chapter paragraphs to dict records:
    {"text": str, "paragraph_index": int|None, ...}
    """
    result: list[dict[str, Any]] = []
    for item in chapter.get("paragraphs", []) or []:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            record = dict(item)
            record["text"] = text
            result.append(record)
        else:
            text = str(item).strip()
            if not text:
                continue
            result.append({"text": text})
    return result
