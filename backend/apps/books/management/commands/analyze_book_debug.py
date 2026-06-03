from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.books.services.analysis_quality import evaluate_analysis_quality
from apps.books.services.atomic_thought_extractor import extract_atomic_thoughts_from_windows
from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.content_filter import is_front_matter_title
from apps.books.services.llm_hierarchical_pipeline import run_hierarchical_llm_pipeline
from apps.books.services.logical_block_splitter import (
    split_into_logical_blocks,
    split_into_logical_blocks_improved,
)
from apps.books.services.llm_service import summarize_book_representative, summarize_logical_block
from apps.books.services.semantic_block_builder import build_semantic_logical_blocks
from apps.books.services.sentence_segmenter import segment_book_sentences
from apps.books.services.sentence_window_builder import build_sentence_windows
from apps.books.services.structure_detector import build_canonical_outline
from apps.books.services.theme_hierarchy import build_theme_hierarchy
from apps.books.services.thought_clusterer import cluster_atomic_thoughts
from apps.books.services.thought_quality import clean_and_validate_thoughts


@dataclass(slots=True)
class PreviewBlock:
    id: int
    order_number: int
    title: str
    chapter_title: str
    source_text: str
    short_summary: str
    start_paragraph: int
    end_paragraph: int
    semantic_data: dict[str, Any]


class Command(BaseCommand):
    help = "Debug analysis pipeline on limited sections/paragraphs without full task run"

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True, help="Path to FB2/PDF file")
        parser.add_argument("--limit-sections", type=int, default=10)
        parser.add_argument("--limit-paragraphs", type=int, default=0)
        parser.add_argument(
            "--mode",
            choices=["debug_structure", "llm_preview", "llm_full", "classic", "classic_improved", "semantic_fast", "hybrid"],
            default="debug_structure",
        )
        parser.add_argument("--show-filtered", action="store_true")
        parser.add_argument("--show-quality", action="store_true")
        parser.add_argument("--no-llm", action="store_true")

    def handle(self, *args, **options):
        path = Path(options["file"]).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise CommandError(f"File not found: {path}")

        no_llm = bool(options["no_llm"])
        previous_llm = os.getenv("LLM_PROVIDER")
        if no_llm:
            os.environ["LLM_PROVIDER"] = "none"

        try:
            content = path.read_bytes()
            parsed = parse_uploaded_book(content, path.name)
            outline = build_canonical_outline(parsed)
            parsed = self._trim_parsed(parsed, outline, options["limit_sections"], options["limit_paragraphs"])

            self.stdout.write(self.style.SUCCESS(f"Title: {parsed.title}"))
            self.stdout.write(f"Authors: {parsed.authors or '-'}")
            self.stdout.write(f"Sections detected: {len(parsed.chapters)}")
            self.stdout.write(f"Mode: {options['mode']}")
            self.stdout.write(
                f"Main-content start: {outline.get('main_content_start')} | "
                f"main sections total: {outline.get('main_sections_count')}"
            )

            self.stdout.write("\nDetected sections (first N):")
            for idx, chapter in enumerate(parsed.chapters[: options["limit_sections"]], start=1):
                title = chapter.get("chapter_title", "")
                paragraphs = chapter.get("paragraphs", [])
                self.stdout.write(f"  {idx}. {title} (paragraphs={len(paragraphs)})")

            preview_blocks, splitter_diagnostics = self._build_preview_blocks(parsed, options["mode"])

            self.stdout.write(f"\nLogical blocks created: {len(preview_blocks)}")
            for item in preview_blocks[:10]:
                self.stdout.write(
                    f"  - [{item.order_number}] {item.title} | chapter={item.chapter_title} | p{item.start_paragraph}-{item.end_paragraph}"
                )

            themes = build_theme_hierarchy(preview_blocks)
            self.stdout.write(f"\nThemes created: {len(themes)}")
            for theme in themes[:3]:
                self.stdout.write(f"  * {theme.title}")

            concept_preview: list[SimpleNamespace] = []
            for theme in themes:
                for sub in theme.subtopics:
                    block = next((b for b in preview_blocks if b.order_number == sub.start_block_number), preview_blocks[0])
                    concept_preview.append(
                        SimpleNamespace(
                            source_quote=sub.source_quote,
                            logical_block=block,
                        )
                    )

            section_titles: list[str] = []
            for block in preview_blocks:
                semantic_data = block.semantic_data or {}
                if isinstance(semantic_data, dict) and semantic_data.get("front_matter"):
                    continue
                section_title = semantic_data.get("section_title") if isinstance(semantic_data, dict) else None
                if section_title:
                    section_titles.append(str(section_title))
                elif block.chapter_title:
                    section_titles.append(block.chapter_title)
            if not section_titles:
                section_titles = [chapter.get("chapter_title", "") for chapter in parsed.chapters]
            summary = summarize_book_representative(
                section_titles=section_titles,
                block_summaries=[
                    item.short_summary
                    for item in preview_blocks
                    if not (isinstance(item.semantic_data, dict) and item.semantic_data.get("front_matter"))
                ] or [item.short_summary for item in preview_blocks],
                top_concepts=[sub.name for theme in themes for sub in theme.subtopics],
            )
            self.stdout.write("\nSummary preview:")
            self.stdout.write(summary[:1200])

            quality = evaluate_analysis_quality(
                summary_text=summary,
                blocks=preview_blocks,
                themes=themes,
                subtopics=[sub for theme in themes for sub in theme.subtopics],
                concept_mentions=concept_preview,
                parser_metadata=parsed.metadata,
                content_filter_stats={
                    "total_paragraphs": sum(splitter_diagnostics.get("content_types_total", {}).values()),
                    "removed_count": splitter_diagnostics.get("filtered_out_block_candidates", 0),
                    "kept_count": len(preview_blocks),
                },
            )

            if options["show_quality"]:
                self.stdout.write("\nQuality diagnostics:")
                self.stdout.write(str(quality))

            if options["show_filtered"] and splitter_diagnostics.get("content_types_total"):
                self.stdout.write("\nContent filter stats:")
                self.stdout.write(str(splitter_diagnostics.get("content_types_total")))

            self.stdout.write("\nTop 5 concepts with quotes:")
            concept_rows = []
            for theme in themes:
                for sub in theme.subtopics:
                    concept_rows.append((sub.name, sub.source_quote))
            for name, quote in concept_rows[:5]:
                self.stdout.write(f"  - {name}: {quote[:180]}")

            self.stdout.write(self.style.SUCCESS("\nDebug analysis completed."))
        finally:
            if no_llm:
                if previous_llm is None:
                    os.environ.pop("LLM_PROVIDER", None)
                else:
                    os.environ["LLM_PROVIDER"] = previous_llm

    def _trim_parsed(self, parsed, outline: dict[str, Any], limit_sections: int, limit_paragraphs: int):
        selected_indexes: list[int] = []
        sections = outline.get("sections", [])
        for section in sections:
            if getattr(section, "is_main_content", False):
                selected_indexes.append(max(0, int(getattr(section, "section_index", 1)) - 1))
        if not selected_indexes:
            selected_indexes = list(range(len(parsed.chapters)))

        if limit_sections > 0:
            selected_indexes = selected_indexes[:limit_sections]

        parsed.chapters = [parsed.chapters[idx] for idx in selected_indexes if 0 <= idx < len(parsed.chapters)]

        if limit_paragraphs > 0:
            for chapter in parsed.chapters:
                paragraphs = list(chapter.get("paragraphs", []))
                chapter["paragraphs"] = paragraphs[:limit_paragraphs]

        return parsed

    def _build_preview_blocks(self, parsed, mode: str) -> tuple[list[PreviewBlock], dict[str, Any]]:
        if mode == "debug_structure":
            blocks_data, diagnostics = split_into_logical_blocks_improved(parsed)
            return self._preview_from_classic_blocks(blocks_data), {**diagnostics, "mode": "debug_structure"}

        if mode in {"llm_preview", "llm_full"}:
            return self._preview_from_llm(parsed, mode=mode), {"mode": mode}

        if mode == "classic":
            blocks_data = split_into_logical_blocks(parsed)
            return self._preview_from_classic_blocks(blocks_data), {"mode": "classic"}

        if mode == "classic_improved":
            blocks_data, diagnostics = split_into_logical_blocks_improved(parsed)
            return self._preview_from_classic_blocks(blocks_data), {**diagnostics, "mode": "classic_improved"}

        if mode == "semantic_fast":
            return self._preview_from_semantic(parsed), {"mode": "semantic_fast"}

        # hybrid mode
        try:
            preview, diagnostics = self._preview_from_semantic(parsed), {"mode": "semantic_fast"}
            diagnostics["hybrid_used"] = "semantic_fast"
            return preview, diagnostics
        except Exception as exc:
            blocks_data, diagnostics = split_into_logical_blocks_improved(parsed)
            diagnostics = {
                **diagnostics,
                "mode": "hybrid",
                "hybrid_used": "classic_improved",
                "fallback_reason": str(exc),
            }
            return self._preview_from_classic_blocks(blocks_data), diagnostics

    def _preview_from_classic_blocks(self, blocks_data) -> list[PreviewBlock]:
        preview_blocks: list[PreviewBlock] = []
        for idx, block in enumerate(blocks_data, start=1):
            clean_text = block.clean_text_for_analysis or block.source_text
            summary = summarize_logical_block(clean_text)
            section_title = block.section_title or block.chapter_title
            front_matter = is_front_matter_title(section_title) or is_front_matter_title(block.chapter_title)
            preview_blocks.append(
                PreviewBlock(
                    id=idx,
                    order_number=block.order_number,
                    title=block.title,
                    chapter_title=block.chapter_title,
                    source_text=block.source_text,
                    short_summary=summary,
                    start_paragraph=block.start_paragraph,
                    end_paragraph=block.end_paragraph,
                    semantic_data={
                        "clean_text_for_analysis": clean_text,
                        "section_title": section_title,
                        "section_path": block.section_path or [],
                        "content_types": block.content_types or {},
                        "paragraph_records": block.paragraph_records or [],
                        "front_matter": front_matter,
                    },
                )
            )
        return preview_blocks

    def _preview_from_llm(self, parsed, *, mode: str) -> list[PreviewBlock]:
        result = run_hierarchical_llm_pipeline(parsed, mode=mode)
        preview_blocks: list[PreviewBlock] = []
        for idx, item in enumerate(result.get("section_results", []), start=1):
            section = item.section
            payload = item.payload
            source_text = "\n\n".join(
                str(row.get("text", "")).strip()
                for row in section.paragraphs
                if str(row.get("text", "")).strip()
            )
            front_matter = is_front_matter_title(section.section_title)
            preview_blocks.append(
                PreviewBlock(
                    id=idx,
                    order_number=idx,
                    title=section.section_title,
                    chapter_title=section.parent_chapter_title or section.chapter_title,
                    source_text=source_text,
                    short_summary=str(payload.get("summary", "")).strip(),
                    start_paragraph=section.start_paragraph,
                    end_paragraph=section.end_paragraph,
                    semantic_data={
                        "clean_text_for_analysis": source_text,
                        "section_title": section.section_title,
                        "section_path": section.section_path,
                        "content_types": {},
                        "paragraph_records": section.paragraphs,
                        "source_sentence_ids": [],
                        "concept_candidates": [term.get("term", "") for term in payload.get("key_terms", []) if isinstance(term, dict)],
                        "atomic_thoughts": [],
                        "front_matter": front_matter,
                    },
                )
            )
        return preview_blocks

    def _preview_from_semantic(self, parsed) -> list[PreviewBlock]:
        sentences = segment_book_sentences(parsed)
        windows = build_sentence_windows(
            sentences,
            min_sentences=3,
            max_sentences=7,
            overlap=2,
            max_words=240,
        )
        thoughts, _extraction_stats = extract_atomic_thoughts_from_windows(windows, sentences)
        clean_thoughts, _quality_stats = clean_and_validate_thoughts(thoughts, sentences)
        clusters, _cluster_stats = cluster_atomic_thoughts(clean_thoughts, sentences)
        semantic_blocks = build_semantic_logical_blocks(clusters, sentences, min_words=180, max_words=1300)
        if not semantic_blocks:
            raise CommandError("semantic_fast produced no blocks")

        preview_blocks: list[PreviewBlock] = []
        for idx, block in enumerate(semantic_blocks, start=1):
            front_matter = is_front_matter_title(block.chapter_title)
            preview_blocks.append(
                PreviewBlock(
                    id=idx,
                    order_number=block.order_number,
                    title=block.title,
                    chapter_title=block.chapter_title,
                    source_text=block.source_text,
                    short_summary=block.main_meaning or summarize_logical_block(block.source_text),
                    start_paragraph=block.start_paragraph,
                    end_paragraph=block.end_paragraph,
                    semantic_data={
                        "clean_text_for_analysis": block.source_text,
                        "section_title": block.chapter_title,
                        "section_path": [],
                        "content_types": {},
                        "paragraph_records": [],
                        "source_sentence_ids": block.source_sentence_ids,
                        "concept_candidates": block.concept_candidates,
                        "atomic_thoughts": block.atomic_thoughts,
                        "front_matter": front_matter,
                    },
                )
            )
        return preview_blocks
