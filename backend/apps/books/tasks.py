from __future__ import annotations

import logging
import os
from datetime import timedelta
from dataclasses import dataclass
from typing import Any

from celery import shared_task
from django.core.management import call_command
from django.conf import settings
from django.utils import timezone

from apps.books.models import (
    BookSummary,
    BookTheme,
    Concept,
    ConceptMention,
    GlobalBookCache,
    LogicalBlock,
    ThemeSubtopic,
    UserBook,
)
from apps.books.services.analysis_quality import evaluate_analysis_quality
from apps.books.services.atomic_thought_extractor import extract_atomic_thoughts_from_windows
from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.concept_normalizer import (
    find_existing_similar_concept,
    is_bad_concept,
    normalize_concept_name,
)
from apps.books.services.llm_service import (
    ensure_llm_ready,
    get_llm_runtime_config,
    llm_provider_enabled,
    select_ollama_model,
    summarize_book,
    summarize_book_representative,
    summarize_logical_block,
)
from apps.books.services.llm_hierarchical_pipeline import run_hierarchical_llm_pipeline
from apps.books.services.content_filter import is_front_matter_title
from apps.books.services.logical_block_splitter import (
    LogicalBlockData,
    split_into_logical_blocks,
    split_into_logical_blocks_improved,
)
from apps.books.services.rag_service import delete_embeddings, save_logical_block_embedding
from apps.books.services.semantic_block_builder import build_semantic_logical_blocks
from apps.books.services.semantic_map_builder import build_semantic_map
from apps.books.services.sentence_segmenter import segment_book_sentences
from apps.books.services.sentence_window_builder import build_sentence_windows
from apps.books.services.structure_detector import build_canonical_outline
from apps.books.services.theme_hierarchy import build_theme_hierarchy
from apps.books.services.thought_clusterer import cluster_atomic_thoughts
from apps.books.services.thought_quality import clean_and_validate_thoughts

logger = logging.getLogger(__name__)


DISALLOWED_QUOTE_TYPES = {
    "code",
    "copyright",
    "exercise",
    "question",
    "figure_caption",
    "table_caption",
    "dedication",
    "acknowledgements",
    "acronym_list",
    "toc",
    "index",
    "bibliography",
}


@dataclass(slots=True)
class PreparedBlock:
    title: str
    order_number: int
    source_text: str
    short_summary: str
    chapter_title: str
    start_paragraph: int
    end_paragraph: int
    token_count: int
    source_sentence_ids: list[str]
    concept_candidates: list[str]
    thought_cluster_ids: list[str]
    semantic_data: dict[str, Any]


def _resolve_analysis_mode(requested_mode: str | None) -> str:
    mode = (requested_mode or os.getenv("BOOK_ANALYSIS_MODE", "llm_fast_batched")).strip().lower()
    aliases = {
        "semantic": "semantic_fast",
        "full": "llm_full",
    }
    mode = aliases.get(mode, mode)
    if mode not in {
        "classic",
        "classic_improved",
        "semantic_fast",
        "hybrid",
        "debug_structure",
        "llm_preview",
        "llm_full",
        "llm_fast_batched",
        "repair_stuck",
    }:
        return "llm_fast_batched"
    return mode


def _load_book_bytes(user_book: UserBook) -> bytes | None:
    if user_book.file:
        user_book.file.open("rb")
        data = user_book.file.read()
        user_book.file.close()
        return data

    donor = (
        UserBook.objects.exclude(id=user_book.id)
        .exclude(file="")
        .filter(file_hash=user_book.file_hash, file__isnull=False)
        .order_by("-uploaded_at")
        .first()
    )
    if donor and donor.file:
        donor.file.open("rb")
        data = donor.file.read()
        donor.file.close()
        return data
    return None


def _analysis_is_ready(cache: GlobalBookCache, *, required_mode: str | None = None) -> bool:
    base_ready = cache.logical_blocks.exists() and cache.themes.exists()
    if not base_ready:
        return False
    metadata = cache.metadata if isinstance(cache.metadata, dict) else {}
    pipeline = str(metadata.get("pipeline_used", "")).strip()
    if required_mode == "llm_full":
        return pipeline == "llm_full"
    if required_mode == "llm_fast_batched":
        return pipeline == "llm_fast_batched"
    if required_mode == "llm_preview":
        return pipeline in {"llm_preview", "llm_full", "llm_fast_batched"}
    if required_mode == "debug_structure":
        return pipeline == "debug_structure"
    return True


def _set_book_stage(
    user_book: UserBook,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: int | None = None,
    error_message: str | None = None,
    llm_provider_used: str | None = None,
    llm_model_used: str | None = None,
    analysis_mode: str | None = None,
    llm_calls_delta: int = 0,
    llm_failures_delta: int = 0,
    fallback_delta: int = 0,
) -> None:
    update_fields: list[str] = []
    now = timezone.now()

    if status is not None and user_book.status != status:
        user_book.status = status
        update_fields.append("status")
    if stage is not None and user_book.current_stage != stage:
        user_book.current_stage = stage
        update_fields.append("current_stage")
    if progress is not None:
        bounded = max(0, min(100, int(progress)))
        if user_book.progress_percent != bounded:
            user_book.progress_percent = bounded
            update_fields.append("progress_percent")
    if error_message is not None and user_book.error_message != error_message:
        user_book.error_message = error_message[:2000]
        update_fields.append("error_message")
    if llm_provider_used is not None and user_book.llm_provider_used != llm_provider_used:
        user_book.llm_provider_used = llm_provider_used[:64]
        update_fields.append("llm_provider_used")
    if llm_model_used is not None and user_book.llm_model_used != llm_model_used:
        user_book.llm_model_used = llm_model_used[:128]
        update_fields.append("llm_model_used")
    if analysis_mode is not None and user_book.analysis_mode != analysis_mode:
        user_book.analysis_mode = analysis_mode[:32]
        update_fields.append("analysis_mode")

    if llm_calls_delta:
        user_book.llm_calls_total = max(0, int(user_book.llm_calls_total) + int(llm_calls_delta))
        update_fields.append("llm_calls_total")
    if llm_failures_delta:
        user_book.llm_failures_total = max(0, int(user_book.llm_failures_total) + int(llm_failures_delta))
        update_fields.append("llm_failures_total")
    if fallback_delta:
        user_book.fallback_used_count = max(0, int(user_book.fallback_used_count) + int(fallback_delta))
        update_fields.append("fallback_used_count")

    user_book.last_heartbeat_at = now
    update_fields.append("last_heartbeat_at")

    if user_book.started_at is None and status in {
        UserBook.Status.QUEUED,
        UserBook.Status.PROCESSING,
        UserBook.Status.PARSING,
        UserBook.Status.STRUCTURE_DETECTION,
        UserBook.Status.FILTERING,
        UserBook.Status.CHUNKING,
        UserBook.Status.LLM_SECTION_ANALYSIS,
        UserBook.Status.LLM_CHAPTER_ANALYSIS,
        UserBook.Status.LLM_BOOK_ANALYSIS,
        UserBook.Status.LLM_FAST_BATCHED_SECTION_ANALYSIS,
        UserBook.Status.LLM_FAST_BATCHED_CHAPTER_ANALYSIS,
        UserBook.Status.LLM_FAST_BATCHED_BOOK_ANALYSIS,
    }:
        user_book.started_at = now
        update_fields.append("started_at")

    if status in {
        UserBook.Status.READY,
        UserBook.Status.PARTIAL_READY,
        UserBook.Status.FAILED,
        UserBook.Status.FAILED_TIMEOUT,
        UserBook.Status.DEBUG_PREVIEW,
        UserBook.Status.HEURISTIC_PREVIEW,
        UserBook.Status.CANCELLED,
    }:
        user_book.finished_at = now
        update_fields.append("finished_at")
        user_book.processed_at = now
        update_fields.append("processed_at")

    if update_fields:
        user_book.save(update_fields=list(dict.fromkeys(update_fields)))


def _mark_user_book_ready(user_book: UserBook, cache: GlobalBookCache) -> None:
    user_book.global_cache = cache
    user_book.title = cache.title
    user_book.authors = cache.authors
    user_book.status = UserBook.Status.READY
    user_book.current_stage = "ready"
    user_book.progress_percent = 100
    user_book.finished_at = timezone.now()
    user_book.last_heartbeat_at = timezone.now()
    user_book.processed_at = timezone.now()
    user_book.error_message = ""
    user_book.save(
        update_fields=[
            "global_cache",
            "title",
            "authors",
            "status",
            "current_stage",
            "progress_percent",
            "finished_at",
            "last_heartbeat_at",
            "processed_at",
            "error_message",
        ]
    )


def _cleanup_cache_analysis(cache: GlobalBookCache) -> None:
    existing_embedding_ids = list(
        LogicalBlock.objects.filter(global_book=cache)
        .exclude(embedding_id="")
        .values_list("embedding_id", flat=True)
    )
    delete_embeddings(existing_embedding_ids)

    ConceptMention.objects.filter(global_book=cache).delete()
    ThemeSubtopic.objects.filter(theme__global_book=cache).delete()
    BookTheme.objects.filter(global_book=cache).delete()
    LogicalBlock.objects.filter(global_book=cache).delete()
    BookSummary.objects.filter(global_book=cache).delete()


def _from_block_data(block_data: LogicalBlockData, *, summary_source: str, pipeline: str) -> PreparedBlock:
    summary_text = summarize_logical_block(summary_source or block_data.source_text)

    section_title = block_data.section_title or block_data.chapter_title
    front_matter = is_front_matter_title(section_title) or is_front_matter_title(block_data.chapter_title)

    semantic_data = {
        "pipeline": pipeline,
        "clean_text_for_analysis": block_data.clean_text_for_analysis or summary_source,
        "section_title": section_title,
        "section_path": block_data.section_path or [],
        "content_types": block_data.content_types or {},
        "paragraph_records": block_data.paragraph_records or [],
        "front_matter": front_matter,
    }

    return PreparedBlock(
        title=block_data.title,
        order_number=block_data.order_number,
        source_text=block_data.source_text,
        short_summary=summary_text,
        chapter_title=block_data.chapter_title,
        start_paragraph=block_data.start_paragraph,
        end_paragraph=block_data.end_paragraph,
        token_count=block_data.token_count,
        source_sentence_ids=[],
        concept_candidates=[],
        thought_cluster_ids=[],
        semantic_data=semantic_data,
    )


def _prepare_classic_blocks(parsed) -> tuple[list[PreparedBlock], dict[str, Any], dict[str, Any]]:
    blocks_data = split_into_logical_blocks(
        parsed,
        min_words=getattr(settings, "BLOCK_MIN_WORDS", 260),
        target_words=getattr(settings, "BLOCK_TARGET_WORDS", 760),
        max_words=getattr(settings, "BLOCK_MAX_WORDS", 1300),
    )
    if not blocks_data:
        raise ValueError("No logical blocks were extracted from the uploaded book")

    prepared = [_from_block_data(item, summary_source=item.source_text, pipeline="classic") for item in blocks_data]
    metrics = {
        "pipeline": "classic",
        "logical_blocks_count": len(prepared),
    }
    return prepared, {}, metrics


def _prepare_classic_improved_blocks(parsed) -> tuple[list[PreparedBlock], dict[str, Any], dict[str, Any]]:
    blocks_data, splitter_diagnostics = split_into_logical_blocks_improved(
        parsed,
        min_words=max(160, getattr(settings, "BLOCK_MIN_WORDS", 260) - 40),
        target_words=getattr(settings, "BLOCK_TARGET_WORDS", 760),
        max_words=getattr(settings, "BLOCK_MAX_WORDS", 1300),
    )
    if not blocks_data:
        raise ValueError("classic_improved splitter returned no blocks")

    prepared: list[PreparedBlock] = []
    for item in blocks_data:
        summary_source = item.clean_text_for_analysis or item.source_text
        prepared.append(_from_block_data(item, summary_source=summary_source, pipeline="classic_improved"))

    content_filter_stats = {
        "by_type": splitter_diagnostics.get("content_types_total", {}),
        "total_paragraphs": sum(splitter_diagnostics.get("content_types_total", {}).values()),
        "kept_count": sum(
            count
            for ctype, count in splitter_diagnostics.get("content_types_total", {}).items()
            if ctype not in DISALLOWED_QUOTE_TYPES and ctype != "empty_or_noise"
        ),
        "removed_count": sum(
            count
            for ctype, count in splitter_diagnostics.get("content_types_total", {}).items()
            if ctype in DISALLOWED_QUOTE_TYPES or ctype == "empty_or_noise"
        ),
    }

    metrics = {
        "pipeline": "classic_improved",
        "logical_blocks_count": len(prepared),
        "splitter_diagnostics": splitter_diagnostics,
        "content_filter_stats": content_filter_stats,
    }
    return prepared, {}, metrics


def _limit_windows_for_semantic(windows: list[Any]) -> list[Any]:
    max_windows_per_section = int(os.getenv("SEMANTIC_MAX_WINDOWS_PER_SECTION", "40"))
    max_windows_per_book = int(os.getenv("SEMANTIC_MAX_WINDOWS_PER_BOOK", "350"))

    grouped: dict[str, list[Any]] = {}
    for window in windows:
        grouped.setdefault(window.chapter_title, []).append(window)

    limited: list[Any] = []
    for chapter_title, items in grouped.items():
        limited.extend(items[:max_windows_per_section])

    return limited[:max_windows_per_book]


def _prepare_semantic_fast_blocks(parsed) -> tuple[list[PreparedBlock], dict[str, Any], dict[str, Any]]:
    sentences = segment_book_sentences(parsed)
    windows = build_sentence_windows(
        sentences,
        min_sentences=3,
        max_sentences=7,
        overlap=2,
        max_words=240,
    )
    windows = _limit_windows_for_semantic(windows)

    thoughts, extraction_stats = extract_atomic_thoughts_from_windows(windows, sentences)
    clean_thoughts, quality_stats = clean_and_validate_thoughts(thoughts, sentences)
    clusters, cluster_stats = cluster_atomic_thoughts(clean_thoughts, sentences)

    semantic_blocks = build_semantic_logical_blocks(
        clusters,
        sentences,
        min_words=max(180, getattr(settings, "BLOCK_MIN_WORDS", 260) - 20),
        max_words=getattr(settings, "BLOCK_MAX_WORDS", 1300),
    )

    if not semantic_blocks:
        raise ValueError("Semantic pipeline produced no logical blocks")

    prepared: list[PreparedBlock] = []
    for block in semantic_blocks:
        if not block.source_text.strip():
            continue
        summary_text = (block.main_meaning or "").strip()
        if len(summary_text) < 40:
            summary_text = summarize_logical_block(block.source_text)

        concept_candidates = [item for item in block.concept_candidates if item.strip()]

        prepared.append(
            PreparedBlock(
                title=block.title,
                order_number=block.order_number,
                source_text=block.source_text,
                short_summary=summary_text,
                chapter_title=block.chapter_title,
                start_paragraph=block.start_paragraph,
                end_paragraph=block.end_paragraph,
                token_count=block.token_count,
                source_sentence_ids=block.source_sentence_ids,
                concept_candidates=concept_candidates,
                thought_cluster_ids=block.thought_cluster_ids,
                semantic_data={
                    "pipeline": "semantic_fast",
                    "main_meaning": summary_text,
                    "atomic_thoughts": block.atomic_thoughts,
                    "concept_candidates": concept_candidates,
                    "clean_text_for_analysis": block.source_text,
                    "section_title": block.chapter_title,
                    "section_path": [],
                },
            )
        )

    if not prepared:
        raise ValueError("Semantic pipeline produced empty block payload")

    if len(parsed.chapters) >= 8 and len(prepared) < 3:
        raise ValueError("Semantic pipeline quality check failed: too few blocks")

    semantic_map = build_semantic_map(parsed.title, semantic_blocks)
    metrics = {
        "pipeline": "semantic_fast",
        "sentences_count": len(sentences),
        "windows_count": len(windows),
        "atomic_thoughts_count": extraction_stats.get("thoughts_count", 0),
        "clean_thoughts_count": len(clean_thoughts),
        "clusters_count": cluster_stats.get("clusters_count", 0),
        "logical_blocks_count": len(prepared),
        "llm_calls": extraction_stats.get("llm_calls", 0) + cluster_stats.get("llm_merge_calls", 0),
        "fallback_calls": extraction_stats.get("fallback_calls", 0),
        "cache_hits": extraction_stats.get("cache_hits", 0),
        "average_confidence": quality_stats.get("average_confidence", 0.0),
        "quality": quality_stats,
    }

    logger.info(
        "Semantic fast metrics: sentences=%s windows=%s thoughts=%s clusters=%s blocks=%s llm_calls=%s",
        metrics["sentences_count"],
        metrics["windows_count"],
        metrics["atomic_thoughts_count"],
        metrics["clusters_count"],
        metrics["logical_blocks_count"],
        metrics["llm_calls"],
    )

    return prepared, semantic_map, metrics


def _prepare_debug_structure_blocks(parsed) -> tuple[list[PreparedBlock], dict[str, Any], dict[str, Any]]:
    outline = build_canonical_outline(parsed)
    main_sections = [item for item in outline.get("sections", []) if item.is_main_content]
    # Fast debug mode: first real main-content sections only.
    main_sections = main_sections[:10]

    prepared: list[PreparedBlock] = []
    for order, section in enumerate(main_sections, start=1):
        source_text = "\n\n".join(str(item.get("text", "")).strip() for item in section.paragraphs if str(item.get("text", "")).strip())
        if not source_text:
            continue
        summary_text = source_text[:500]
        concept_candidates = []
        prepared.append(
            PreparedBlock(
                title=section.section_title,
                order_number=order,
                source_text=source_text,
                short_summary=summary_text,
                chapter_title=section.parent_chapter_title or section.chapter_title,
                start_paragraph=section.start_paragraph,
                end_paragraph=section.end_paragraph,
                token_count=len(source_text.split()),
                source_sentence_ids=[],
                concept_candidates=concept_candidates,
                thought_cluster_ids=[],
                semantic_data={
                    "pipeline": "debug_structure",
                    "section_type": section.content_type,
                    "clean_text_for_analysis": source_text,
                    "section_title": section.section_title,
                    "section_path": section.section_path,
                    "content_types": {},
                    "paragraph_records": section.paragraphs,
                    "front_matter": False,
                    "debug_structure_only": True,
                },
            )
        )

    if not prepared:
        raise ValueError("debug_structure found no main-content sections")

    semantic_map = {
        "book_title": parsed.title,
        "blocks": [
            {
                "order": item.order_number,
                "title": item.title,
                "main_meaning": item.short_summary[:300],
                "source_range": f"p{item.start_paragraph}-{item.end_paragraph}",
                "concepts": [],
                "children": [],
            }
            for item in prepared
        ],
        "links": [],
    }
    metrics = {
        "pipeline": "debug_structure",
        "logical_blocks_count": len(prepared),
        "outline_stats": outline.get("stats", {}),
        "main_content_start": outline.get("main_content_start"),
        "sections_total": outline.get("sections_total"),
        "main_sections_count": outline.get("main_sections_count"),
        "filtered_sections_count": outline.get("filtered_sections_count"),
    }
    return prepared, semantic_map, metrics


def _prepare_llm_hierarchical_blocks(parsed, *, mode: str) -> tuple[list[PreparedBlock], dict[str, Any], dict[str, Any]]:
    result = run_hierarchical_llm_pipeline(parsed, mode=mode)
    section_results = result.get("section_results", [])
    if not section_results:
        raise ValueError(f"{mode} produced no section analysis results")

    prepared: list[PreparedBlock] = []
    for order, entry in enumerate(section_results, start=1):
        section = entry.section
        payload = entry.payload
        source_text = "\n\n".join(
            str(item.get("text", "")).strip()
            for item in section.paragraphs
            if str(item.get("text", "")).strip()
        )
        if not source_text:
            continue
        key_terms = [str(item.get("term", "")).strip() for item in payload.get("key_terms", []) if str(item.get("term", "")).strip()]
        prepared.append(
            PreparedBlock(
                title=section.section_title[:512],
                order_number=order,
                source_text=source_text,
                short_summary=str(payload.get("summary", "")).strip()[:2000] or summarize_logical_block(source_text),
                chapter_title=(section.parent_chapter_title or section.chapter_title)[:512],
                start_paragraph=section.start_paragraph,
                end_paragraph=section.end_paragraph,
                token_count=section.word_count,
                source_sentence_ids=[],
                concept_candidates=list(dict.fromkeys(key_terms))[:16],
                thought_cluster_ids=[],
                semantic_data={
                    "pipeline": mode,
                    "section_type": section.content_type,
                    "section_analysis": payload,
                    "clean_text_for_analysis": source_text,
                    "section_title": section.section_title,
                    "section_path": section.section_path,
                    "content_types": {},
                    "paragraph_records": section.paragraphs,
                    "front_matter": False,
                },
            )
        )

    if not prepared:
        raise ValueError(f"{mode} produced no prepared blocks")

    chapter_payloads = result.get("chapter_payloads", [])
    book_payload = result.get("book_payload", {})

    semantic_map_blocks = []
    for item in prepared:
        section_analysis = (item.semantic_data or {}).get("section_analysis", {})
        subtopics = section_analysis.get("subtopics", []) if isinstance(section_analysis, dict) else []
        semantic_map_blocks.append(
            {
                "order": item.order_number,
                "title": item.title,
                "main_meaning": item.short_summary,
                "source_range": f"p{item.start_paragraph}-{item.end_paragraph}",
                "concepts": item.concept_candidates[:10],
                "children": [
                    {
                        "thought": str(sub.get("summary", ""))[:280],
                        "quote": str(sub.get("source_quote", ""))[:280],
                        "source_sentence_ids": [],
                    }
                    for sub in subtopics[:8]
                    if isinstance(sub, dict)
                ],
            }
        )

    semantic_map = {
        "book_title": parsed.title,
        "book_summary": book_payload.get("book_summary", ""),
        "global_themes": book_payload.get("global_themes", []),
        "blocks": semantic_map_blocks,
        "links": [],
    }

    metrics = {
        "pipeline": mode,
        "logical_blocks_count": len(prepared),
        "outline_stats": result.get("outline", {}).get("stats", {}),
        "main_content_start": result.get("outline", {}).get("main_content_start"),
        "sections_total": result.get("metrics", {}).get("sections_total"),
        "main_sections_total": result.get("metrics", {}).get("main_sections_total"),
        "chapters_analyzed": result.get("metrics", {}).get("chapters_analyzed"),
        "llm_calls": result.get("metrics", {}).get("llm_calls_total", 0),
        "llm_failures": result.get("metrics", {}).get("llm_failures_total", 0),
        "fallback_calls": result.get("metrics", {}).get("fallback_used_count", 0),
        "chapter_payloads": chapter_payloads,
        "book_payload": book_payload,
    }
    return prepared, semantic_map, metrics


def _save_prepared_blocks(
    cache: GlobalBookCache,
    prepared_blocks: list[PreparedBlock],
) -> tuple[list[LogicalBlock], dict[int, LogicalBlock], list[str]]:
    created_blocks: list[LogicalBlock] = []
    block_by_order: dict[int, LogicalBlock] = {}
    block_summaries: list[str] = []

    for item in sorted(prepared_blocks, key=lambda block: block.order_number):
        if not item.source_text.strip():
            continue

        block = LogicalBlock.objects.create(
            global_book=cache,
            title=item.title[:512],
            order_number=item.order_number,
            source_text=item.source_text,
            short_summary=item.short_summary,
            start_paragraph=max(0, item.start_paragraph),
            end_paragraph=max(max(0, item.start_paragraph), max(0, item.end_paragraph)),
            chapter_title=(item.chapter_title or "")[:512],
            token_count=max(0, item.token_count),
            semantic_data=item.semantic_data,
            source_sentence_ids=item.source_sentence_ids,
            concept_candidates=item.concept_candidates,
            thought_cluster_ids=item.thought_cluster_ids,
        )
        save_logical_block_embedding(
            block.id,
            item.source_text,
            metadata={
                "block_id": block.id,
                "global_book_id": cache.id,
                "title": block.title,
                "chapter_title": block.chapter_title,
                "order_number": block.order_number,
            },
        )
        created_blocks.append(block)
        block_by_order[block.order_number] = block
        if item.short_summary:
            block_summaries.append(item.short_summary)

    if not created_blocks:
        raise ValueError("No logical blocks were persisted")
    return created_blocks, block_by_order, block_summaries


def _valid_quote_for_block(block: LogicalBlock, quote: str) -> bool:
    quote = (quote or "").strip()
    if not quote:
        return False
    semantic_data = block.semantic_data or {}
    rows = semantic_data.get("paragraph_records") if isinstance(semantic_data, dict) else None
    if not isinstance(rows, list):
        return True

    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        ctype = str(row.get("content_type", ""))
        if not text:
            continue
        if quote.lower() in text.lower() or text.lower() in quote.lower():
            return ctype not in DISALLOWED_QUOTE_TYPES
    return True


def _repair_quote_from_block(block: LogicalBlock) -> str:
    semantic_data = block.semantic_data or {}
    rows = semantic_data.get("paragraph_records") if isinstance(semantic_data, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            ctype = str(row.get("content_type", ""))
            text = str(row.get("text", "")).strip()
            if text and ctype not in DISALLOWED_QUOTE_TYPES:
                return text[:320]
    return (block.source_text or "")[:320]


def _build_themes_and_concepts(
    cache: GlobalBookCache,
    created_blocks: list[LogicalBlock],
    block_by_order: dict[int, LogicalBlock],
    pipeline_metrics: dict[str, Any] | None = None,
) -> tuple[int, int, list[str]]:
    pipeline_metrics = pipeline_metrics or {}
    chapter_payloads = pipeline_metrics.get("chapter_payloads", []) if isinstance(pipeline_metrics, dict) else []
    themes_data = []

    if isinstance(chapter_payloads, list) and chapter_payloads:
        for order, chapter_payload in enumerate(chapter_payloads, start=1):
            if not isinstance(chapter_payload, dict):
                continue
            chapter_title = str(chapter_payload.get("chapter_title", "")).strip() or f"Chapter {order}"
            chapter_blocks = [item for item in created_blocks if item.chapter_title == chapter_title]
            if not chapter_blocks:
                chapter_blocks = created_blocks[max(0, order - 1) : order] or [created_blocks[0]]
            start_block_number = chapter_blocks[0].order_number
            end_block_number = chapter_blocks[-1].order_number
            subtopics = []
            for idx, term in enumerate(chapter_payload.get("important_terms", [])[:10], start=1):
                value = str(term).strip()
                if not value:
                    continue
                quote = ""
                for block in chapter_blocks:
                    if value.lower() in block.source_text.lower():
                        quote = block.source_text[:320]
                        break
                if not quote:
                    quote = chapter_blocks[0].source_text[:320]
                subtopics.append(
                    {
                        "name": value,
                        "summary": f"Concept in chapter context: {value}.",
                        "source_quote": quote,
                        "importance_score": max(0.2, 1.0 - idx * 0.06),
                        "source_block_order": start_block_number,
                    }
                )

            themes_data.append(
                {
                    "chapter_title": chapter_title,
                    "title": chapter_title,
                    "order_number": order,
                    "start_block_number": start_block_number,
                    "end_block_number": end_block_number,
                    "start_paragraph": chapter_blocks[0].start_paragraph,
                    "end_paragraph": chapter_blocks[-1].end_paragraph,
                    "summary": str(chapter_payload.get("chapter_summary", "")).strip() or chapter_blocks[0].short_summary,
                    "subtopics": subtopics,
                }
            )

    if not themes_data:
        themes_data = build_theme_hierarchy(created_blocks)
    if not themes_data:
        only_block = created_blocks[0]
        themes_data = [
            {
                "chapter_title": only_block.chapter_title or "Основной раздел",
                "title": only_block.chapter_title or "Основная тема",
                "order_number": 1,
                "start_block_number": only_block.order_number,
                "end_block_number": created_blocks[-1].order_number,
                "start_paragraph": only_block.start_paragraph,
                "end_paragraph": created_blocks[-1].end_paragraph,
                "summary": only_block.short_summary or "Ключевая тема книги",
                "subtopics": [],
            }
        ]

    subtopics_count = 0
    concept_mentions_count = 0
    concept_names: list[str] = []

    for theme_data in themes_data:
        if isinstance(theme_data, dict):
            chapter_title = str(theme_data.get("chapter_title", ""))
            title = str(theme_data.get("title", ""))
            order_number = int(theme_data.get("order_number", 1))
            start_block_number = int(theme_data.get("start_block_number", 1))
            end_block_number = int(theme_data.get("end_block_number", start_block_number))
            start_paragraph = int(theme_data.get("start_paragraph", 0))
            end_paragraph = int(theme_data.get("end_paragraph", 0))
            summary = str(theme_data.get("summary", ""))
            subtopics_raw = list(theme_data.get("subtopics", []))
        else:
            chapter_title = theme_data.chapter_title
            title = theme_data.title
            order_number = theme_data.order_number
            start_block_number = theme_data.start_block_number
            end_block_number = theme_data.end_block_number
            start_paragraph = theme_data.start_paragraph
            end_paragraph = theme_data.end_paragraph
            summary = theme_data.summary
            subtopics_raw = list(theme_data.subtopics)

        theme = BookTheme.objects.create(
            global_book=cache,
            chapter_title=(chapter_title or "")[:512],
            title=(title or "Основная тема")[:512],
            order_number=order_number,
            start_block_number=max(1, start_block_number),
            end_block_number=max(max(1, start_block_number), end_block_number),
            start_paragraph=max(0, start_paragraph),
            end_paragraph=max(max(0, start_paragraph), max(0, end_paragraph)),
            summary=(summary or "Ключевая тема главы")[:2000],
        )

        anchor_block = block_by_order.get(theme.start_block_number) or created_blocks[0]

        for raw_subtopic in subtopics_raw[:8]:
            if isinstance(raw_subtopic, dict):
                sub_name = str(raw_subtopic.get("name", "")).strip()
                sub_summary = str(raw_subtopic.get("summary", "")).strip()
                sub_quote = str(raw_subtopic.get("source_quote", "")).strip()
                sub_score = raw_subtopic.get("importance_score", 0.6)
                sub_start_block = int(raw_subtopic.get("source_block_order", theme.start_block_number)) if str(raw_subtopic.get("source_block_order", "")).isdigit() else theme.start_block_number
            else:
                sub_name = str(getattr(raw_subtopic, "name", "")).strip()
                sub_summary = str(getattr(raw_subtopic, "summary", "")).strip()
                sub_quote = str(getattr(raw_subtopic, "source_quote", "")).strip()
                sub_score = getattr(raw_subtopic, "importance_score", 0.6)
                sub_start_block = int(getattr(raw_subtopic, "start_block_number", theme.start_block_number))

            if not sub_name:
                continue
            normalized_name = normalize_concept_name(sub_name)
            if not normalized_name or is_bad_concept(sub_name):
                continue

            try:
                score = float(sub_score)
            except (TypeError, ValueError):
                score = 0.6
            score = max(0.0, min(1.0, score))

            mention_anchor = block_by_order.get(sub_start_block) or anchor_block
            if not _valid_quote_for_block(mention_anchor, sub_quote):
                sub_quote = _repair_quote_from_block(mention_anchor)

            subtopic = ThemeSubtopic.objects.create(
                theme=theme,
                name=sub_name[:255],
                normalized_name=normalized_name[:255],
                summary=(sub_summary or "Ключевая подтема")[:1200],
                source_quote=(sub_quote or "")[:1000],
                importance_score=score,
                start_paragraph=max(0, mention_anchor.start_paragraph),
                end_paragraph=max(mention_anchor.start_paragraph, mention_anchor.end_paragraph),
            )
            subtopics_count += 1

            concept = find_existing_similar_concept(normalized_name)
            if concept is None:
                concept = Concept.objects.create(
                    name=subtopic.name,
                    normalized_name=normalized_name[:255],
                    description=(subtopic.summary or theme.summary)[:2000],
                )
            elif not concept.description and (subtopic.summary or theme.summary):
                concept.description = (subtopic.summary or theme.summary)[:2000]
                concept.save(update_fields=["description", "updated_at"])

            ConceptMention.objects.update_or_create(
                concept=concept,
                logical_block=mention_anchor,
                defaults={
                    "global_book": cache,
                    "short_explanation": subtopic.summary or theme.summary or "Ключевая мысль темы",
                    "source_quote": subtopic.source_quote,
                    "importance_score": subtopic.importance_score,
                },
            )
            concept_mentions_count += 1
            concept_names.append(concept.name)

    return subtopics_count, concept_mentions_count, concept_names


def _save_cache_summary(
    cache: GlobalBookCache,
    parsed,
    created_blocks: list[LogicalBlock],
    *,
    analysis_mode: str,
    pipeline_used: str,
    semantic_map: dict[str, Any],
    pipeline_metrics: dict[str, Any],
    subtopics_count: int,
    concept_mentions_count: int,
    concept_names: list[str],
) -> dict[str, Any]:
    summary_blocks = []
    for block in created_blocks:
        semantic_data = block.semantic_data or {}
        if isinstance(semantic_data, dict) and semantic_data.get("front_matter"):
            continue
        summary_blocks.append(block)

    if not summary_blocks:
        summary_blocks = created_blocks

    section_titles = []
    for block in summary_blocks:
        semantic_data = block.semantic_data or {}
        section_title = semantic_data.get("section_title") if isinstance(semantic_data, dict) else None
        if section_title:
            section_titles.append(str(section_title))
        elif block.chapter_title:
            section_titles.append(block.chapter_title)

    block_summaries = [block.short_summary for block in summary_blocks if block.short_summary]

    book_payload = pipeline_metrics.get("book_payload", {}) if isinstance(pipeline_metrics, dict) else {}
    full_summary = ""
    if isinstance(book_payload, dict):
        full_summary = str(book_payload.get("book_summary", "")).strip()
    if not full_summary:
        full_summary = summarize_book_representative(
            section_titles=list(dict.fromkeys(section_titles))[:30],
            block_summaries=block_summaries[:80],
            top_concepts=list(dict.fromkeys(concept_names))[:40],
        )
    if not full_summary:
        full_summary = summarize_book(block_summaries)

    cache.title = parsed.title
    cache.authors = parsed.authors
    cache.full_summary = full_summary
    cache.analysis_version = {
        "classic": "concept_rag_classic_v1",
        "classic_improved": "concept_rag_classic_improved_v1",
        "semantic_fast": "concept_rag_semantic_fast_v1",
        "debug_structure": "concept_rag_debug_structure_v1",
        "llm_preview": "concept_rag_llm_preview_v1",
        "llm_full": "concept_rag_llm_full_v1",
        "llm_fast_batched": "concept_rag_llm_fast_batched_v1",
    }.get(pipeline_used, "concept_rag_classic_improved_v1")

    content_filter_stats = pipeline_metrics.get("content_filter_stats", {}) if isinstance(pipeline_metrics, dict) else {}

    quality_diagnostics = evaluate_analysis_quality(
        summary_text=full_summary,
        blocks=created_blocks,
        themes=list(BookTheme.objects.filter(global_book=cache).order_by("order_number")),
        subtopics=list(ThemeSubtopic.objects.filter(theme__global_book=cache)),
        concept_mentions=list(ConceptMention.objects.filter(global_book=cache).select_related("logical_block")),
        parser_metadata=parsed.metadata,
        content_filter_stats=content_filter_stats,
    )

    cache.metadata = {
        **parsed.metadata,
        "chapters_count": len(parsed.chapters),
        "logical_blocks_count": len(created_blocks),
        "themes_count": BookTheme.objects.filter(global_book=cache).count(),
        "subtopics_count": subtopics_count,
        "concept_mentions_count": concept_mentions_count,
        "analysis_mode": analysis_mode,
        "pipeline_used": pipeline_used,
        "pipeline_metrics": pipeline_metrics,
        "semantic_map": semantic_map,
        "quality_diagnostics": quality_diagnostics,
    }
    cache.save(
        update_fields=[
            "title",
            "authors",
            "full_summary",
            "analysis_version",
            "metadata",
            "updated_at",
        ]
    )

    BookSummary.objects.create(
        global_book=cache,
        short_summary=(full_summary[:1000] if full_summary else ""),
        detailed_summary=full_summary or "",
    )

    return quality_diagnostics


@shared_task(bind=True)
def analyze_book_task(
    self,
    user_book_id: int,
    force_reanalyze: bool = False,
    analysis_mode: str | None = None,
):
    task_id = getattr(getattr(self, "request", None), "id", None)
    try:
        user_book = UserBook.objects.select_related("global_cache").get(id=user_book_id)
    except UserBook.DoesNotExist:
        logger.warning("UserBook %s not found for analysis", user_book_id)
        return

    try:
        mode = _resolve_analysis_mode(analysis_mode)
        runtime = get_llm_runtime_config()
        logger.info(
            "analyze_book start task_id=%s book_id=%s mode=%s provider=%s",
            task_id,
            user_book_id,
            mode,
            runtime.get("provider"),
        )
        _set_book_stage(
            user_book,
            status=UserBook.Status.QUEUED,
            stage="queued",
            progress=2,
            error_message="",
            llm_provider_used=runtime.get("provider", ""),
            analysis_mode=mode,
        )

        llm_required = mode in {"llm_preview", "llm_full", "llm_fast_batched"}
        if llm_required:
            ready = ensure_llm_ready(require_enabled=True)
            if not ready.get("ok"):
                _set_book_stage(
                    user_book,
                    status=UserBook.Status.FAILED,
                    stage="failed",
                    progress=100,
                    error_message=str(ready.get("error", "LLM is unavailable")),
                )
                return
            selected_model = select_ollama_model("fast", available_models=ready.get("models", [])) or ""
            _set_book_stage(user_book, llm_model_used=selected_model)

        _set_book_stage(user_book, status=UserBook.Status.PARSING, stage="parsing", progress=8)
        content = _load_book_bytes(user_book)
        if not content:
            raise ValueError("File for analysis is missing")
        parsed = parse_uploaded_book(content, user_book.original_filename)

        _set_book_stage(user_book, status=UserBook.Status.STRUCTURE_DETECTION, stage="structure_detection", progress=16)
        cache, _ = GlobalBookCache.objects.get_or_create(
            file_hash=user_book.file_hash,
            defaults={
                "title": parsed.title,
                "authors": parsed.authors,
                "metadata": {},
                "analysis_version": "concept_rag_classic_improved_v1",
            },
        )

        if not force_reanalyze and _analysis_is_ready(cache, required_mode=mode):
            _mark_user_book_ready(user_book, cache)
            return

        if mode == "llm_fast_batched":
            batch_size = int(os.getenv("LLM_FAST_BATCH_SIZE", "10"))
            output_dir = os.getenv("LLM_FAST_BATCH_OUTPUT_DIR", "")
            command_kwargs = {
                "book_id": user_book.id,
                "batch_size": batch_size,
                "max_input_chars": int(os.getenv("SECTION_LLM_MAX_INPUT_CHARS", "1200")),
                "batch_timeout_seconds": int(os.getenv("LLM_FAST_BATCH_TIMEOUT_SECONDS", "600")),
                "stop_on_error": True,
            }
            if output_dir:
                command_kwargs["output_dir"] = output_dir
            call_command("run_llm_fast_batched_analysis", **command_kwargs)
            return

        _set_book_stage(user_book, status=UserBook.Status.FILTERING, stage="filtering", progress=24)
        _cleanup_cache_analysis(cache)

        prepared_blocks: list[PreparedBlock] = []
        semantic_map: dict[str, Any] = {}
        pipeline_metrics: dict[str, Any] = {}
        pipeline_used = "classic"
        fallback_reason = ""

        if mode == "repair_stuck":
            _set_book_stage(
                user_book,
                status=UserBook.Status.CANCELLED,
                stage="cancelled",
                progress=100,
                error_message="repair_stuck mode is not supported for analyze_book_task",
            )
            return
        elif mode == "debug_structure":
            prepared_blocks, semantic_map, pipeline_metrics = _prepare_debug_structure_blocks(parsed)
            pipeline_used = "debug_structure"
        elif mode in {"llm_preview", "llm_full"}:
            _set_book_stage(
                user_book,
                status=UserBook.Status.CHUNKING,
                stage="chunking",
                progress=32,
            )
            prepared_blocks, semantic_map, pipeline_metrics = _prepare_llm_hierarchical_blocks(parsed, mode=mode)
            pipeline_used = mode
            _set_book_stage(
                user_book,
                status=UserBook.Status.LLM_BOOK_ANALYSIS,
                stage="llm_book_analysis",
                progress=70,
                llm_calls_delta=int(pipeline_metrics.get("llm_calls", 0)),
                llm_failures_delta=int(pipeline_metrics.get("llm_failures", 0)),
                fallback_delta=int(pipeline_metrics.get("fallback_calls", 0)),
            )
        elif mode == "classic":
            prepared_blocks, semantic_map, pipeline_metrics = _prepare_classic_blocks(parsed)
            pipeline_used = "classic"
        elif mode == "classic_improved":
            try:
                prepared_blocks, semantic_map, pipeline_metrics = _prepare_classic_improved_blocks(parsed)
                pipeline_used = "classic_improved"
            except Exception as exc:
                fallback_reason = str(exc)
                logger.exception("classic_improved failed for user_book=%s, fallback=classic", user_book_id)
                prepared_blocks, semantic_map, pipeline_metrics = _prepare_classic_blocks(parsed)
                pipeline_used = "classic"
        elif mode == "semantic_fast":
            try:
                prepared_blocks, semantic_map, pipeline_metrics = _prepare_semantic_fast_blocks(parsed)
                pipeline_used = "semantic_fast"
            except Exception as exc:
                fallback_reason = str(exc)
                logger.exception("semantic_fast failed for user_book=%s, fallback=classic_improved", user_book_id)
                try:
                    prepared_blocks, semantic_map, pipeline_metrics = _prepare_classic_improved_blocks(parsed)
                    pipeline_used = "classic_improved"
                except Exception:
                    prepared_blocks, semantic_map, pipeline_metrics = _prepare_classic_blocks(parsed)
                    pipeline_used = "classic"
        else:  # hybrid
            try:
                prepared_blocks, semantic_map, pipeline_metrics = _prepare_semantic_fast_blocks(parsed)
                pipeline_used = "semantic_fast"
            except Exception as exc:
                fallback_reason = str(exc)
                logger.exception("hybrid semantic stage failed for user_book=%s", user_book_id)
                try:
                    prepared_blocks, semantic_map, pipeline_metrics = _prepare_classic_improved_blocks(parsed)
                    pipeline_used = "classic_improved"
                except Exception as exc2:
                    fallback_reason += f" | classic_improved: {exc2}"
                    prepared_blocks, semantic_map, pipeline_metrics = _prepare_classic_blocks(parsed)
                    pipeline_used = "classic"

        if fallback_reason:
            pipeline_metrics = {
                **pipeline_metrics,
                "fallback_reason": fallback_reason[:1200],
            }

        _set_book_stage(
            user_book,
            status=UserBook.Status.SAVING_RESULTS,
            stage="saving_results",
            progress=82,
        )
        created_blocks, block_by_order, _block_summaries = _save_prepared_blocks(cache, prepared_blocks)
        subtopics_count, concept_mentions_count, concept_names = _build_themes_and_concepts(
            cache,
            created_blocks,
            block_by_order,
            pipeline_metrics=pipeline_metrics,
        )
        quality_diagnostics = _save_cache_summary(
            cache,
            parsed,
            created_blocks,
            analysis_mode=mode,
            pipeline_used=pipeline_used,
            semantic_map=semantic_map,
            pipeline_metrics=pipeline_metrics,
            subtopics_count=subtopics_count,
            concept_mentions_count=concept_mentions_count,
            concept_names=concept_names,
        )

        if quality_diagnostics.get("quality_score", 0.0) < 0.35:
            logger.warning(
                "Low quality analysis for user_book=%s score=%s problems=%s",
                user_book_id,
                quality_diagnostics.get("quality_score"),
                quality_diagnostics.get("problems", []),
            )

        # Final status resolution.
        final_status = UserBook.Status.READY
        if mode == "debug_structure":
            final_status = UserBook.Status.DEBUG_PREVIEW
        elif not llm_provider_enabled() and pipeline_used not in {"llm_preview", "llm_full"}:
            final_status = UserBook.Status.HEURISTIC_PREVIEW
        elif pipeline_metrics.get("fallback_calls", 0) and pipeline_metrics.get("llm_calls", 0):
            fallback_ratio = float(pipeline_metrics.get("fallback_calls", 0)) / max(
                1.0, float(pipeline_metrics.get("llm_calls", 0))
            )
            if fallback_ratio >= 0.35:
                final_status = UserBook.Status.PARTIAL_READY
        elif mode in {"llm_preview"}:
            final_status = UserBook.Status.PARTIAL_READY

        user_book.global_cache = cache
        user_book.title = cache.title
        user_book.authors = cache.authors
        user_book.save(update_fields=["global_cache", "title", "authors"])
        _set_book_stage(
            user_book,
            status=final_status,
            stage=final_status,
            progress=100,
            error_message="",
        )
        logger.info(
            "analyze_book done task_id=%s book_id=%s mode=%s final_status=%s llm_calls=%s llm_failures=%s fallback=%s",
            task_id,
            user_book_id,
            mode,
            final_status,
            user_book.llm_calls_total,
            user_book.llm_failures_total,
            user_book.fallback_used_count,
        )
    except Exception as exc:
        logger.exception("Failed to analyze user_book=%s", user_book_id)
        _set_book_stage(
            user_book,
            status=UserBook.Status.FAILED,
            stage="failed",
            progress=100,
            error_message=str(exc),
        )


@shared_task(bind=True)
def watchdog_stuck_books_task(self, timeout_minutes: int | None = None) -> dict[str, Any]:
    timeout = max(10, int(timeout_minutes or os.getenv("BOOK_STUCK_TIMEOUT_MINUTES", "90")))
    cutoff = timezone.now() - timedelta(minutes=timeout)
    active_statuses = {
        UserBook.Status.PROCESSING,
        UserBook.Status.PARSING,
        UserBook.Status.STRUCTURE_DETECTION,
        UserBook.Status.FILTERING,
        UserBook.Status.CHUNKING,
        UserBook.Status.LLM_SECTION_ANALYSIS,
        UserBook.Status.LLM_CHAPTER_ANALYSIS,
        UserBook.Status.LLM_BOOK_ANALYSIS,
        UserBook.Status.LLM_FAST_BATCHED_SECTION_ANALYSIS,
        UserBook.Status.LLM_FAST_BATCHED_CHAPTER_ANALYSIS,
        UserBook.Status.LLM_FAST_BATCHED_BOOK_ANALYSIS,
        UserBook.Status.BUILDING_MAP,
        UserBook.Status.SAVING_RESULTS,
    }

    qs = UserBook.objects.filter(status__in=active_statuses).filter(
        last_heartbeat_at__lt=cutoff
    ) | UserBook.objects.filter(status__in=active_statuses, last_heartbeat_at__isnull=True, updated_at__lt=cutoff)
    qs = qs.distinct()

    updated_ids: list[int] = []
    for item in qs:
        _set_book_stage(
            item,
            status=UserBook.Status.FAILED_TIMEOUT,
            stage="failed_timeout",
            progress=min(item.progress_percent, 99),
            error_message="Analysis timeout watchdog: stale heartbeat detected.",
        )
        updated_ids.append(item.id)

    if updated_ids:
        logger.warning("watchdog_stuck_books_task marked failed_timeout ids=%s", updated_ids)
    return {"updated_count": len(updated_ids), "updated_ids": updated_ids, "timeout_minutes": timeout}


@shared_task(bind=True)
def generate_book_study_notes_task(self, user_book_id: int, force: bool = False) -> dict[str, Any]:
    """Build cached study notes from already saved analysis results."""
    from apps.books.services.study_notes import generate_book_study_notes

    notes = generate_book_study_notes(user_book_id, force=force)
    return {
        "book_id": user_book_id,
        "notes_id": notes.id,
        "status": notes.status,
        "model_name": notes.model_name,
    }
