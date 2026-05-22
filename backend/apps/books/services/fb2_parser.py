from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxml import etree


@dataclass
class ParsedChapter:
    chapter_title: str
    paragraphs: list[str]


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
    return " ".join(value.split()).strip()


def iter_by_tag(element: etree._Element, tag_name: str):
    for node in element.iter():
        if isinstance(node.tag, str) and strip_namespace(node.tag) == tag_name:
            yield node


def extract_text(node: etree._Element) -> str:
    return normalize_spaces(" ".join(node.itertext()))


def parse_fb2(content: bytes) -> ParsedBook:
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"Invalid FB2 XML: {exc}") from exc

    title = "Без названия"
    authors_list: list[str] = []
    metadata: dict[str, Any] = {}

    for description in iter_by_tag(root, "description"):
        for title_info in iter_by_tag(description, "title-info"):
            for book_title in iter_by_tag(title_info, "book-title"):
                text = extract_text(book_title)
                if text:
                    title = text
                    break
            for author in iter_by_tag(title_info, "author"):
                parts = []
                for child_name in ("first-name", "middle-name", "last-name", "nickname"):
                    child = next((c for c in author if strip_namespace(c.tag) == child_name), None)
                    if child is not None:
                        value = extract_text(child)
                        if value:
                            parts.append(value)
                if parts:
                    authors_list.append(" ".join(parts))
            annotation = next((node for node in title_info if strip_namespace(node.tag) == "annotation"), None)
            if annotation is not None:
                metadata["annotation"] = extract_text(annotation)
        break

    chapters: list[dict[str, Any]] = []
    for body in iter_by_tag(root, "body"):
        for section in [n for n in body if isinstance(n.tag, str) and strip_namespace(n.tag) == "section"]:
            chapters.extend(_extract_section_tree(section))

    if not chapters:
        paragraphs = []
        for paragraph in iter_by_tag(root, "p"):
            parent = paragraph.getparent()
            if parent is not None and strip_namespace(parent.tag) == "title":
                continue
            text = extract_text(paragraph)
            if text:
                paragraphs.append(text)
        if paragraphs:
            chapters.append({"chapter_title": "Основной текст", "paragraphs": paragraphs})

    return ParsedBook(
        title=title,
        authors=", ".join(dict.fromkeys(authors_list)),
        chapters=chapters,
        metadata=metadata,
    )


def _extract_section_tree(section: etree._Element) -> list[dict[str, Any]]:
    chapter_title = "Без названия главы"
    direct_title = next((node for node in section if isinstance(node.tag, str) and strip_namespace(node.tag) == "title"), None)
    if direct_title is not None:
        title_parts = [extract_text(p) for p in direct_title if isinstance(p.tag, str) and strip_namespace(p.tag) == "p"]
        merged = normalize_spaces(" ".join(part for part in title_parts if part))
        if merged:
            chapter_title = merged

    paragraphs: list[str] = []
    for paragraph in section.iter():
        if not isinstance(paragraph.tag, str) or strip_namespace(paragraph.tag) != "p":
            continue
        parent = paragraph.getparent()
        if parent is not None and strip_namespace(parent.tag) == "title":
            continue
        text = extract_text(paragraph)
        if text:
            paragraphs.append(text)

    result = [{"chapter_title": chapter_title, "paragraphs": paragraphs}] if paragraphs else []

    for child in section:
        if isinstance(child.tag, str) and strip_namespace(child.tag) == "section":
            result.extend(_extract_section_tree(child))
    return result
