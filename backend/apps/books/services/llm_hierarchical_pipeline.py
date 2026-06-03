from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.books.services.llm_service import (
    analyze_section_with_llm,
    build_book_analysis_with_llm,
    get_llm_runtime_config,
    merge_chapter_analyses_with_llm,
)
from apps.books.services.structure_detector import CanonicalSection, build_canonical_outline


@dataclass(slots=True)
class SectionAnalysisResult:
    section: CanonicalSection
    payload: dict[str, Any]


def _split_records_into_chunks(
    records: list[dict[str, Any]],
    *,
    max_chars: int,
    max_chunks: int,
) -> list[list[dict[str, Any]]]:
    if not records:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for record in records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        item_chars = len(text) + 2
        if current and (current_chars + item_chars > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(record)
        current_chars += item_chars

    if current:
        chunks.append(current)

    if len(chunks) <= max_chunks:
        return chunks

    # Merge tail chunks to keep bounded call count.
    merged: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        if len(merged) < max_chunks - 1:
            merged.append(chunk)
        else:
            if not merged:
                merged.append(chunk)
            else:
                merged[-1].extend(chunk)
    return merged


def _chunk_text(chunk: list[dict[str, Any]]) -> str:
    return "\n\n".join(str(item.get("text", "")).strip() for item in chunk if str(item.get("text", "")).strip())


def _merge_chunk_payloads(section_title: str, chunk_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunk_payloads:
        return {
            "section_title": section_title,
            "section_type": "main_content",
            "summary": "",
            "key_terms": [],
            "subtopics": [],
            "important_facts": [],
            "formulas_or_protocols": [],
            "source_quotes": [],
            "difficulty_level": "unknown",
            "links_to_parent_theme": [],
            "quality_flags": ["empty_chunk_payloads"],
            "_meta": {"llm_used": False, "fallback_used": True, "llm_failure": "empty_chunk_payloads"},
        }

    summaries = [str(item.get("summary", "")).strip() for item in chunk_payloads if str(item.get("summary", "")).strip()]
    key_terms: list[dict[str, Any]] = []
    subtopics: list[dict[str, Any]] = []
    facts: list[str] = []
    formulas: list[str] = []
    quotes: list[str] = []
    links: list[str] = []
    quality_flags: list[str] = []
    llm_used = 0
    fallback_used = 0

    for payload in chunk_payloads:
        for item in payload.get("key_terms", []):
            if isinstance(item, dict):
                key_terms.append(item)
        for item in payload.get("subtopics", []):
            if isinstance(item, dict):
                subtopics.append(item)
        facts.extend([str(item) for item in payload.get("important_facts", []) if str(item).strip()])
        formulas.extend([str(item) for item in payload.get("formulas_or_protocols", []) if str(item).strip()])
        quotes.extend([str(item) for item in payload.get("source_quotes", []) if str(item).strip()])
        links.extend([str(item) for item in payload.get("links_to_parent_theme", []) if str(item).strip()])
        quality_flags.extend([str(item) for item in payload.get("quality_flags", []) if str(item).strip()])
        meta = payload.get("_meta", {})
        if isinstance(meta, dict) and meta.get("llm_used"):
            llm_used += 1
        if isinstance(meta, dict) and meta.get("fallback_used"):
            fallback_used += 1

    # Dedupe key terms.
    seen_terms: set[str] = set()
    clean_terms: list[dict[str, Any]] = []
    for item in sorted(key_terms, key=lambda x: float(x.get("importance", 0.0)), reverse=True):
        term = str(item.get("term", "")).strip().lower()
        if not term or term in seen_terms:
            continue
        seen_terms.add(term)
        clean_terms.append(item)
        if len(clean_terms) >= 14:
            break

    summary = " ".join(summaries[:4]).strip()
    return {
        "section_title": section_title,
        "section_type": "main_content",
        "summary": summary[:1800],
        "key_terms": clean_terms,
        "subtopics": subtopics[:12],
        "important_facts": list(dict.fromkeys(facts))[:20],
        "formulas_or_protocols": list(dict.fromkeys(formulas))[:20],
        "source_quotes": list(dict.fromkeys(quotes))[:12],
        "difficulty_level": "intermediate",
        "links_to_parent_theme": list(dict.fromkeys(links))[:10],
        "quality_flags": list(dict.fromkeys(quality_flags))[:12],
        "_meta": {
            "llm_used": llm_used > 0,
            "fallback_used": fallback_used > 0,
            "llm_failure": "",
            "chunk_count": len(chunk_payloads),
        },
    }


def run_hierarchical_llm_pipeline(
    parsed_book,
    *,
    mode: str = "llm_full",
) -> dict[str, Any]:
    """
    Hierarchical LLM pipeline:
    section -> chapter -> book
    """

    cfg = get_llm_runtime_config()
    outline = build_canonical_outline(parsed_book)
    all_sections: list[CanonicalSection] = list(outline.get("sections", []))
    main_sections = [item for item in all_sections if item.is_main_content]

    preview_limit = int(cfg.get("max_calls_per_chapter", 40))
    if mode == "llm_preview":
        main_sections = main_sections[: max(2, min(3, preview_limit))]

    max_calls_book = int(cfg.get("max_calls_per_book", 220))
    max_chunks_per_section = int(cfg.get("max_chunks_per_section", 4))
    max_input_chars = int(cfg.get("max_input_chars", 8000))

    section_results: list[SectionAnalysisResult] = []
    llm_calls_total = 0
    llm_failures_total = 0
    fallback_used_count = 0

    for section in main_sections:
        if llm_calls_total >= max_calls_book:
            break
        chunks = _split_records_into_chunks(
            section.paragraphs,
            max_chars=max_input_chars,
            max_chunks=max_chunks_per_section,
        )
        chunk_payloads: list[dict[str, Any]] = []
        for chunk in chunks:
            if llm_calls_total >= max_calls_book:
                break
            payload = analyze_section_with_llm(
                section_title=section.section_title,
                section_text=_chunk_text(chunk),
                chapter_title=section.parent_chapter_title,
                section_type=section.content_type,
            )
            chunk_payloads.append(payload)
            llm_calls_total += 1
            meta = payload.get("_meta", {})
            if isinstance(meta, dict) and meta.get("fallback_used"):
                fallback_used_count += 1
                llm_failures_total += 1

        merged_payload = _merge_chunk_payloads(section.section_title, chunk_payloads)
        section_results.append(SectionAnalysisResult(section=section, payload=merged_payload))

    # Group section payloads by parent chapter.
    by_chapter: dict[str, list[dict[str, Any]]] = {}
    for result in section_results:
        by_chapter.setdefault(result.section.parent_chapter_title, []).append(result.payload)

    chapter_payloads: list[dict[str, Any]] = []
    max_calls_chapter = int(cfg.get("max_calls_per_chapter", 40))
    for chapter_title, payloads in by_chapter.items():
        if llm_calls_total >= max_calls_book:
            break
        if len(chapter_payloads) >= max_calls_chapter:
            break
        chapter_payload = merge_chapter_analyses_with_llm(chapter_title, payloads)
        chapter_payloads.append(chapter_payload)
        llm_calls_total += 1
        meta = chapter_payload.get("_meta", {})
        if isinstance(meta, dict) and meta.get("fallback_used"):
            fallback_used_count += 1
            llm_failures_total += 1

    book_payload = build_book_analysis_with_llm(chapter_payloads)
    llm_calls_total += 1
    book_meta = book_payload.get("_meta", {})
    if isinstance(book_meta, dict) and book_meta.get("fallback_used"):
        fallback_used_count += 1
        llm_failures_total += 1

    return {
        "outline": outline,
        "section_results": section_results,
        "chapter_payloads": chapter_payloads,
        "book_payload": book_payload,
        "metrics": {
            "llm_calls_total": llm_calls_total,
            "llm_failures_total": llm_failures_total,
            "fallback_used_count": fallback_used_count,
            "sections_total": len(all_sections),
            "main_sections_total": len(main_sections),
            "chapters_analyzed": len(chapter_payloads),
        },
    }
