from __future__ import annotations

import logging

from celery import shared_task
from django.conf import settings
from django.db import transaction
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
from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.concept_normalizer import (
    find_existing_similar_concept,
    is_bad_concept,
    normalize_concept_name,
)
from apps.books.services.llm_service import summarize_book, summarize_logical_block
from apps.books.services.logical_block_splitter import split_into_logical_blocks
from apps.books.services.rag_service import delete_embeddings, save_logical_block_embedding
from apps.books.services.theme_hierarchy import build_theme_hierarchy

logger = logging.getLogger(__name__)


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


def _analysis_is_ready(cache: GlobalBookCache) -> bool:
    return cache.logical_blocks.exists() and cache.themes.exists()


def _mark_user_book_ready(user_book: UserBook, cache: GlobalBookCache) -> None:
    user_book.global_cache = cache
    user_book.title = cache.title
    user_book.authors = cache.authors
    user_book.status = UserBook.Status.READY
    user_book.processed_at = timezone.now()
    user_book.error_message = ""
    user_book.save(
        update_fields=[
            "global_cache",
            "title",
            "authors",
            "status",
            "processed_at",
            "error_message",
        ]
    )


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2})
def analyze_book_task(self, user_book_id: int, force_reanalyze: bool = False):
    try:
        user_book = UserBook.objects.select_related("global_cache").get(id=user_book_id)
    except UserBook.DoesNotExist:
        logger.warning("UserBook %s not found for analysis", user_book_id)
        return

    try:
        user_book.status = UserBook.Status.PROCESSING
        user_book.error_message = ""
        user_book.save(update_fields=["status", "error_message"])

        content = _load_book_bytes(user_book)
        if not content:
            raise ValueError("File for analysis is missing")

        parsed = parse_uploaded_book(content, user_book.original_filename)
        blocks_data = split_into_logical_blocks(
            parsed,
            min_words=getattr(settings, "BLOCK_MIN_WORDS", 260),
            target_words=getattr(settings, "BLOCK_TARGET_WORDS", 760),
            max_words=getattr(settings, "BLOCK_MAX_WORDS", 1300),
        )
        if not blocks_data:
            raise ValueError("No logical blocks were extracted from the uploaded book")

        with transaction.atomic():
            cache, _ = GlobalBookCache.objects.get_or_create(
                file_hash=user_book.file_hash,
                defaults={
                    "title": parsed.title,
                    "authors": parsed.authors,
                    "metadata": {},
                    "analysis_version": "concept_rag_v2_theme_map",
                },
            )

            if not force_reanalyze and _analysis_is_ready(cache):
                _mark_user_book_ready(user_book, cache)
                return

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

            block_summaries: list[str] = []
            created_blocks: list[LogicalBlock] = []
            block_by_order: dict[int, LogicalBlock] = {}

            for block_data in blocks_data:
                short_summary = summarize_logical_block(block_data.source_text)
                block = LogicalBlock.objects.create(
                    global_book=cache,
                    title=block_data.title,
                    order_number=block_data.order_number,
                    source_text=block_data.source_text,
                    short_summary=short_summary,
                    start_paragraph=block_data.start_paragraph,
                    end_paragraph=block_data.end_paragraph,
                    chapter_title=block_data.chapter_title,
                    token_count=block_data.token_count,
                )
                save_logical_block_embedding(
                    block.id,
                    block_data.source_text,
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
                if short_summary:
                    block_summaries.append(short_summary)

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
                        "summary": " ".join(block_summaries[:2])[:2000],
                        "subtopics": [],
                    }
                ]

            concept_mentions_count = 0
            subtopics_count = 0

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
                    start_block_number=start_block_number,
                    end_block_number=end_block_number,
                    start_paragraph=max(0, start_paragraph),
                    end_paragraph=max(0, end_paragraph),
                    summary=(summary or "Ключевая тема главы.")[:2000],
                )

                anchor_block = block_by_order.get(theme.start_block_number)
                if anchor_block is None:
                    anchor_block = created_blocks[0]

                for raw_subtopic in subtopics_raw[:4]:
                    if isinstance(raw_subtopic, dict):
                        sub_name = str(raw_subtopic.get("name", "")).strip()
                        sub_summary = str(raw_subtopic.get("summary", "")).strip()
                        sub_quote = str(raw_subtopic.get("source_quote", "")).strip()
                        sub_score = raw_subtopic.get("importance_score", 0.6)
                        sub_start_par = int(raw_subtopic.get("start_paragraph", theme.start_paragraph)) if str(raw_subtopic.get("start_paragraph", "")).isdigit() else theme.start_paragraph
                        sub_end_par = int(raw_subtopic.get("end_paragraph", theme.end_paragraph)) if str(raw_subtopic.get("end_paragraph", "")).isdigit() else theme.end_paragraph
                        sub_start_block = int(raw_subtopic.get("start_block_number", theme.start_block_number)) if str(raw_subtopic.get("start_block_number", "")).isdigit() else theme.start_block_number
                    else:
                        sub_name = str(getattr(raw_subtopic, "name", "")).strip()
                        sub_summary = str(getattr(raw_subtopic, "summary", "")).strip()
                        sub_quote = str(getattr(raw_subtopic, "source_quote", "")).strip()
                        sub_score = getattr(raw_subtopic, "importance_score", 0.6)
                        sub_start_par = int(getattr(raw_subtopic, "start_paragraph", theme.start_paragraph))
                        sub_end_par = int(getattr(raw_subtopic, "end_paragraph", theme.end_paragraph))
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

                    subtopic = ThemeSubtopic.objects.create(
                        theme=theme,
                        name=sub_name[:255],
                        normalized_name=normalized_name[:255],
                        summary=(sub_summary or "Ключевая подтема темы.")[:1200],
                        source_quote=sub_quote[:1000],
                        importance_score=score,
                        start_paragraph=max(0, sub_start_par),
                        end_paragraph=max(max(0, sub_start_par), max(0, sub_end_par)),
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

                    mention_anchor = block_by_order.get(sub_start_block) or anchor_block
                    ConceptMention.objects.update_or_create(
                        concept=concept,
                        logical_block=mention_anchor,
                        defaults={
                            "global_book": cache,
                            "short_explanation": subtopic.summary or theme.summary or "Ключевая мысль темы.",
                            "source_quote": subtopic.source_quote,
                            "importance_score": subtopic.importance_score,
                        },
                    )
                    concept_mentions_count += 1

            full_summary = summarize_book(block_summaries)
            cache.title = parsed.title
            cache.authors = parsed.authors
            cache.full_summary = full_summary
            cache.analysis_version = "concept_rag_v2_theme_map"
            cache.metadata = {
                **parsed.metadata,
                "chapters_count": len(parsed.chapters),
                "logical_blocks_count": len(created_blocks),
                "themes_count": BookTheme.objects.filter(global_book=cache).count(),
                "subtopics_count": subtopics_count,
                "concept_mentions_count": concept_mentions_count,
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

        _mark_user_book_ready(user_book, cache)
    except Exception as exc:
        logger.exception("Failed to analyze user_book=%s", user_book_id)
        user_book.mark_failed(str(exc))
