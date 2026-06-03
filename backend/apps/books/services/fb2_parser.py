from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from lxml import etree


HEADING_NUMBER_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+){0,4}[.)]?"
    r"|глава\s+\d+"
    r"|часть\s+[ivxlcdm\d]+"
    r"|раздел\s+\d+"
    r")\b",
    re.IGNORECASE,
)
ROMAN_RE = re.compile(r"^[IVXLCDM]+(?:[.)]|\b)", re.IGNORECASE)


@dataclass
class ParsedBook:
    title: str
    authors: str
    chapters: list[dict[str, Any]]
    metadata: dict[str, Any]


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def normalize_spaces(value: str) -> str:
    return " ".join((value or "").split()).strip()


def iter_by_tag(element: etree._Element, tag_name: str):
    for node in element.iter():
        if isinstance(node.tag, str) and strip_namespace(node.tag) == tag_name:
            yield node


def extract_text(node: etree._Element) -> str:
    return normalize_spaces(" ".join(node.itertext()))


def _title_from_section(section: etree._Element) -> str:
    title_node = next(
        (child for child in section if isinstance(child.tag, str) and strip_namespace(child.tag) == "title"),
        None,
    )
    if title_node is not None:
        parts = [
            extract_text(part)
            for part in title_node.iter()
            if isinstance(part.tag, str) and strip_namespace(part.tag) in {"p", "subtitle"}
        ]
        joined = normalize_spaces(" ".join(item for item in parts if item))
        if joined:
            return joined

    # Fallback: use first short paragraph looking like heading.
    for child in section:
        if not isinstance(child.tag, str):
            continue
        if strip_namespace(child.tag) != "p":
            continue
        text = extract_text(child)
        if _looks_like_heading(text):
            return text
    return ""


def _looks_like_heading(text: str) -> bool:
    text = normalize_spaces(text)
    if not text:
        return False
    if len(text) > 150:
        return False
    if HEADING_NUMBER_RE.match(text):
        return True
    if ROMAN_RE.match(text):
        return True
    if text.istitle() and len(text.split()) <= 8:
        return True
    if len(text.split()) <= 8 and not re.search(r"[.!?]", text):
        return True
    return False


def _paragraph_record(text: str, paragraph_index: int) -> dict[str, Any]:
    return {
        "text": text,
        "paragraph_index": paragraph_index,
        "content_type_hint": "main_text",
    }


def _iter_direct_section_paragraphs(section: etree._Element) -> list[str]:
    paragraphs: list[str] = []
    for child in section:
        if not isinstance(child.tag, str):
            continue
        tag = strip_namespace(child.tag)
        if tag == "title":
            continue
        if tag == "section":
            continue
        if tag == "p":
            text = extract_text(child)
            if text:
                paragraphs.append(text)
            continue

        # Some FB2 blocks (epigraph, poem, cite) may contain <p> deeper.
        for p_node in child.iter():
            if isinstance(p_node.tag, str) and strip_namespace(p_node.tag) == "p":
                if p_node.getparent() is not None and strip_namespace(p_node.getparent().tag) == "title":
                    continue
                text = extract_text(p_node)
                if text:
                    paragraphs.append(text)
    return paragraphs


def _section_path_to_payload(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in path:
        result.append(
            {
                "level": item["level"],
                "title": item["title"],
                "parent_title": item.get("parent_title", ""),
                "order": item["order"],
                "paragraph_start": item.get("paragraph_start", 0),
                "paragraph_end": item.get("paragraph_end", 0),
            }
        )
    return result


def _flatten_sections(
    section: etree._Element,
    *,
    parent_title: str,
    level: int,
    order_seed: list[int],
    paragraph_counter: list[int],
    path: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    section_title = _title_from_section(section) or f"Section {order_seed[0]}"

    order_seed[0] += 1
    current_path_item = {
        "level": level,
        "title": section_title,
        "parent_title": parent_title,
        "order": order_seed[0],
        "paragraph_start": paragraph_counter[0] + 1,
        "paragraph_end": paragraph_counter[0],
    }
    current_path = path + [current_path_item]

    paragraphs = _iter_direct_section_paragraphs(section)
    paragraph_records: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        paragraph_counter[0] += 1
        paragraph_records.append(_paragraph_record(paragraph, paragraph_counter[0]))

    if paragraph_records:
        current_path_item["paragraph_end"] = paragraph_records[-1]["paragraph_index"]
    else:
        current_path_item["paragraph_end"] = max(current_path_item["paragraph_start"] - 1, 0)

    flattened: list[dict[str, Any]] = []
    if paragraph_records:
        flattened.append(
            {
                "chapter_title": section_title,
                "paragraphs": paragraph_records,
                "section_path": _section_path_to_payload(current_path),
                "section_level": level,
                "parent_title": parent_title,
                "order": current_path_item["order"],
            }
        )

    nested_sections = [
        child
        for child in section
        if isinstance(child.tag, str) and strip_namespace(child.tag) == "section"
    ]
    for nested in nested_sections:
        flattened.extend(
            _flatten_sections(
                nested,
                parent_title=section_title,
                level=level + 1,
                order_seed=order_seed,
                paragraph_counter=paragraph_counter,
                path=current_path,
            )
        )
    return flattened


def _extract_authors(description: etree._Element) -> str:
    authors_list: list[str] = []
    for title_info in iter_by_tag(description, "title-info"):
        for author in iter_by_tag(title_info, "author"):
            parts = []
            for child_name in ("first-name", "middle-name", "last-name", "nickname"):
                child = next((c for c in author if isinstance(c.tag, str) and strip_namespace(c.tag) == child_name), None)
                if child is not None:
                    value = extract_text(child)
                    if value:
                        parts.append(value)
            if parts:
                authors_list.append(" ".join(parts))
    return ", ".join(dict.fromkeys(authors_list))


def parse_fb2(content: bytes) -> ParsedBook:
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"Invalid FB2 XML: {exc}") from exc

    title = "Без названия"
    authors = ""
    metadata: dict[str, Any] = {
        "source_format": "fb2",
        "parser_version": "fb2_structured_v2",
    }

    description = next(iter_by_tag(root, "description"), None)
    if description is not None:
        for title_info in iter_by_tag(description, "title-info"):
            book_title = next(iter_by_tag(title_info, "book-title"), None)
            if book_title is not None:
                title_text = extract_text(book_title)
                if title_text:
                    title = title_text
            annotation = next((node for node in title_info if isinstance(node.tag, str) and strip_namespace(node.tag) == "annotation"), None)
            if annotation is not None:
                metadata["annotation"] = extract_text(annotation)
            break
        authors = _extract_authors(description)

    chapters: list[dict[str, Any]] = []
    order_seed = [0]
    paragraph_counter = [0]

    for body in iter_by_tag(root, "body"):
        sections = [
            node for node in body
            if isinstance(node.tag, str) and strip_namespace(node.tag) == "section"
        ]
        for section in sections:
            chapters.extend(
                _flatten_sections(
                    section,
                    parent_title="",
                    level=1,
                    order_seed=order_seed,
                    paragraph_counter=paragraph_counter,
                    path=[],
                )
            )

    if not chapters:
        paragraphs: list[dict[str, Any]] = []
        for paragraph in iter_by_tag(root, "p"):
            parent = paragraph.getparent()
            if parent is not None and isinstance(parent.tag, str) and strip_namespace(parent.tag) == "title":
                continue
            text = extract_text(paragraph)
            if not text:
                continue
            paragraph_counter[0] += 1
            paragraphs.append(_paragraph_record(text, paragraph_counter[0]))

        if paragraphs:
            chapters.append(
                {
                    "chapter_title": "Основной текст",
                    "paragraphs": paragraphs,
                    "section_path": [
                        {
                            "level": 1,
                            "title": "Основной текст",
                            "parent_title": "",
                            "order": 1,
                            "paragraph_start": paragraphs[0]["paragraph_index"],
                            "paragraph_end": paragraphs[-1]["paragraph_index"],
                        }
                    ],
                    "section_level": 1,
                    "parent_title": "",
                    "order": 1,
                }
            )

    metadata["sections_count"] = len(chapters)
    metadata["paragraphs_count"] = paragraph_counter[0]
    metadata["section_paths"] = [chapter.get("section_path", []) for chapter in chapters[:3000]]

    return ParsedBook(
        title=title,
        authors=authors,
        chapters=chapters,
        metadata=metadata,
    )
