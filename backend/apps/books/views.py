from __future__ import annotations

import os

from django.conf import settings
from django.core.files.base import ContentFile
from django.db.models import Count, F, Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import pagination, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.books.models import (
    Concept,
    ConceptMention,
    GlobalBookCache,
    LogicalBlock,
    UserBook,
    UserConceptEdit,
)
from apps.books.serializers import (
    BookSummarySerializer,
    ConceptMentionSerializer,
    ConceptSerializer,
    LogicalBlockDetailSerializer,
    LogicalBlockSerializer,
    UserBookSerializer,
    UserConceptEditSerializer,
)
from apps.books.services.book_parser import (
    is_supported_book_extension,
    parse_uploaded_book,
    supported_extensions_text,
)
from apps.books.services.concept_normalizer import normalize_concept_name
from apps.books.services.concept_map import build_user_concept_map
from apps.books.services.glossary_export import export_csv, export_json, export_pdf, export_txt
from apps.books.services.hashing import sha256_bytes
from apps.books.services.llm_service import compare_concept_mentions, ensure_llm_ready
from apps.books.services.rag_service import search_similar_blocks, search_similar_concepts
from apps.books.services.rotation import (
    MAX_BOOKS_PER_USER,
    get_oldest_unprotected_book,
    get_user_books_count,
    rotate_books_if_needed,
)
from apps.books.tasks import analyze_book_task


class BookPagination(pagination.PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class BlockPagination(pagination.PageNumberPagination):
    page_size = 30
    page_size_query_param = "page_size"
    max_page_size = 200


class MentionPagination(pagination.PageNumberPagination):
    page_size = 50
    page_size_query_param = "page_size"
    max_page_size = 200


def _bool_from_request(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _book_to_delete_payload(book: UserBook | None):
    if not book:
        return None
    return {
        "id": book.id,
        "title": book.title or book.original_filename,
        "uploaded_at": book.uploaded_at,
        "is_protected": book.is_protected,
    }


def _analysis_mode_from_request(value) -> str:
    mode = str(value or "").strip().lower()
    if mode in {
        "classic",
        "classic_improved",
        "semantic_fast",
        "semantic",
        "hybrid",
        "debug_structure",
        "llm_preview",
        "llm_full",
        "llm_fast_batched",
        "llm_thought_chain",
    }:
        if mode == "semantic":
            return "semantic_fast"
        return mode
    return "llm_thought_chain"


def _has_cached_analysis(global_cache, *, required_mode: str = "llm_full") -> bool:
    if not global_cache:
        return False
    if not (global_cache.logical_blocks.exists() and global_cache.themes.exists()):
        return False
    metadata = global_cache.metadata if isinstance(global_cache.metadata, dict) else {}
    pipeline = str(metadata.get("pipeline_used", "")).strip()
    if required_mode == "llm_full":
        return pipeline == "llm_full"
    if required_mode == "llm_fast_batched":
        return pipeline == "llm_fast_batched"
    if required_mode == "llm_thought_chain":
        return pipeline == "llm_thought_chain"
    if required_mode == "llm_preview":
        return pipeline in {"llm_preview", "llm_full", "llm_fast_batched"}
    if required_mode == "debug_structure":
        return pipeline == "debug_structure"
    return True


def _get_user_book_or_404(user, book_id: int) -> UserBook:
    return get_object_or_404(UserBook.objects.select_related("global_cache"), id=book_id, user=user)


def _get_user_edits_map(user, mention_queryset) -> dict[int, UserConceptEdit]:
    mention_ids = list(mention_queryset.values_list("id", flat=True))
    edits = UserConceptEdit.objects.filter(user=user, concept_mention_id__in=mention_ids)
    return {edit.concept_mention_id: edit for edit in edits}


def _build_user_book_map(user, mention_queryset) -> dict[int, int]:
    global_book_ids = list(mention_queryset.values_list("global_book_id", flat=True).distinct())
    mapping: dict[int, int] = {}
    for row in UserBook.objects.filter(user=user, global_cache_id__in=global_book_ids).values("id", "global_cache_id"):
        global_id = row["global_cache_id"]
        if global_id not in mapping:
            mapping[global_id] = row["id"]
    return mapping


class UserBooksView(APIView):
    def get(self, request):
        queryset = UserBook.objects.filter(user=request.user).order_by("-uploaded_at")
        paginator = BookPagination()
        page = paginator.paginate_queryset(queryset, request)
        serializer = UserBookSerializer(page, many=True)
        return paginator.get_paginated_response(
            {
                "books": serializer.data,
                "books_used": get_user_books_count(request.user),
                "books_limit": MAX_BOOKS_PER_USER,
            }
        )


class UploadBooksView(APIView):
    def post(self, request):
        files = request.FILES.getlist("files")
        if not files and "file" in request.FILES:
            files = [request.FILES["file"]]
        if not files:
            return Response({"detail": "No files uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        confirm_rotation = _bool_from_request(request.data.get("confirm_rotation"))
        current_count = get_user_books_count(request.user)
        if current_count + len(files) > MAX_BOOKS_PER_USER and not confirm_rotation:
            candidate = get_oldest_unprotected_book(request.user)
            if not candidate:
                return Response(
                    {"detail": "Upload limit reached (50), and all books are protected."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            return Response(
                {
                    "need_confirmation": True,
                    "book_to_delete": _book_to_delete_payload(candidate),
                    "detail": "Rotation confirmation required.",
                },
                status=status.HTTP_409_CONFLICT,
            )

        analysis_mode = _analysis_mode_from_request(request.data.get("analysis_mode"))
        if analysis_mode in {"llm_full", "llm_preview", "llm_fast_batched", "llm_thought_chain"}:
            llm_state = ensure_llm_ready(require_enabled=True)
            if not llm_state.get("ok"):
                return Response(
                    {"detail": f"LLM is required for {analysis_mode}: {llm_state.get('error')}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        created_items = []
        for uploaded_file in files:
            if uploaded_file.size > settings.MAX_FB2_FILE_SIZE:
                return Response(
                    {"detail": f"File {uploaded_file.name} exceeds 50MB limit."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if not is_supported_book_extension(uploaded_file.name):
                return Response(
                    {
                        "detail": (
                            f"File {uploaded_file.name} has unsupported extension. "
                            f"Supported: {supported_extensions_text()}."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if get_user_books_count(request.user) >= MAX_BOOKS_PER_USER:
                rotation = rotate_books_if_needed(request.user, confirmed=confirm_rotation)
                if not rotation.can_upload:
                    return Response(
                        {
                            "need_confirmation": rotation.need_confirmation,
                            "book_to_delete": _book_to_delete_payload(rotation.book_to_delete),
                            "detail": rotation.reason,
                        },
                        status=status.HTTP_409_CONFLICT if rotation.need_confirmation else status.HTTP_400_BAD_REQUEST,
                    )

            content = uploaded_file.read()
            if not content:
                return Response({"detail": f"File {uploaded_file.name} is empty."}, status=status.HTTP_400_BAD_REQUEST)
            try:
                parse_uploaded_book(content, uploaded_file.name)
            except ValueError as exc:
                return Response(
                    {"detail": f"Invalid book file in {uploaded_file.name}: {exc}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            file_hash = sha256_bytes(content)
            global_cache = GlobalBookCache.objects.filter(file_hash=file_hash).first()
            original_filename = os.path.basename(uploaded_file.name)

            if global_cache and _has_cached_analysis(global_cache, required_mode=analysis_mode):
                user_book = UserBook.objects.create(
                    user=request.user,
                    global_cache=global_cache,
                    title=global_cache.title,
                    authors=global_cache.authors,
                    original_filename=original_filename,
                    file_hash=file_hash,
                    status=UserBook.Status.READY,
                    current_stage="ready",
                    progress_percent=100,
                )
                created_items.append({"id": user_book.id, "status": user_book.status, "used_cache": True})
                continue

            user_book = UserBook.objects.create(
                user=request.user,
                global_cache=global_cache if global_cache else None,
                original_filename=original_filename,
                file_hash=file_hash,
                status=UserBook.Status.QUEUED,
                current_stage="queued",
                progress_percent=1,
            )
            user_book.file.save(original_filename, ContentFile(content), save=True)
            analyze_book_task.delay(
                user_book.id,
                force_reanalyze=bool(global_cache),
                analysis_mode=analysis_mode,
            )
            created_items.append(
                {
                    "id": user_book.id,
                    "status": user_book.status,
                    "used_cache": False,
                    "analysis_mode": analysis_mode,
                }
            )

        return Response({"results": created_items}, status=status.HTTP_201_CREATED)


class ConfirmRotationView(APIView):
    def post(self, request):
        rotation = rotate_books_if_needed(request.user, confirmed=True)
        if not rotation.can_upload:
            return Response(
                {
                    "detail": rotation.reason,
                    "need_confirmation": rotation.need_confirmation,
                    "book_to_delete": _book_to_delete_payload(rotation.book_to_delete),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response({"detail": rotation.reason or "Rotation completed."})


class UserBookDetailView(APIView):
    def get(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        return Response(UserBookSerializer(user_book).data)

    def delete(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        if user_book.is_protected:
            return Response({"detail": "Disable protection before deleting this book."}, status=status.HTTP_400_BAD_REQUEST)
        user_book.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ProtectBookView(APIView):
    def post(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        user_book.is_protected = not user_book.is_protected
        user_book.save(update_fields=["is_protected"])
        return Response({"id": user_book.id, "is_protected": user_book.is_protected})


class ReanalyzeBookView(APIView):
    def post(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        analysis_mode = _analysis_mode_from_request(request.data.get("analysis_mode"))
        if analysis_mode in {"llm_full", "llm_preview", "llm_fast_batched", "llm_thought_chain"}:
            llm_state = ensure_llm_ready(require_enabled=True)
            if not llm_state.get("ok"):
                return Response(
                    {"detail": f"LLM is required for {analysis_mode}: {llm_state.get('error')}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        user_book.status = UserBook.Status.QUEUED
        user_book.current_stage = "queued"
        user_book.progress_percent = 1
        user_book.error_message = ""
        user_book.save(update_fields=["status", "current_stage", "progress_percent", "error_message"])
        analyze_book_task.delay(user_book.id, force_reanalyze=True, analysis_mode=analysis_mode)
        return Response({"detail": "Reanalysis started.", "status": user_book.status, "analysis_mode": analysis_mode})


class BookSummaryView(APIView):
    def get(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        if not user_book.global_cache:
            return Response({"detail": "Book has no analysis yet."}, status=status.HTTP_400_BAD_REQUEST)

        UserBook.objects.filter(id=user_book.id).update(views_count=F("views_count") + 1)
        user_book.refresh_from_db(fields=["views_count"])
        data = BookSummarySerializer(
            {
                "book_id": user_book.id,
                "title": user_book.title,
                "authors": user_book.authors,
                "full_summary": user_book.global_cache.full_summary or "",
                "blocks_count": user_book.global_cache.logical_blocks.count(),
                "concepts_count": user_book.global_cache.concept_mentions.count(),
            }
        ).data
        data["views_count"] = user_book.views_count
        return Response(data)


class BookSemanticMapView(APIView):
    def get(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        if not user_book.global_cache:
            return Response({"book_title": user_book.title or user_book.original_filename, "blocks": [], "links": []})

        metadata = user_book.global_cache.metadata or {}
        semantic_map = metadata.get("semantic_map")
        if isinstance(semantic_map, dict):
            return Response(semantic_map)

        return Response({"book_title": user_book.title, "blocks": [], "links": []})


class BookBlocksView(APIView):
    def get(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        if not user_book.global_cache:
            return Response({"count": 0, "results": []})
        blocks = LogicalBlock.objects.filter(global_book=user_book.global_cache).order_by("order_number")
        paginator = BlockPagination()
        page = paginator.paginate_queryset(blocks, request)
        serializer = LogicalBlockSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)


class BookBlockDetailView(APIView):
    def get(self, request, book_id, block_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        if not user_book.global_cache:
            return Response({"detail": "Book has no analysis yet."}, status=status.HTTP_400_BAD_REQUEST)
        block = get_object_or_404(
            LogicalBlock.objects.filter(global_book=user_book.global_cache).prefetch_related("concept_mentions__concept"),
            id=block_id,
        )
        mentions = ConceptMention.objects.filter(logical_block=block)
        edits_map = _get_user_edits_map(request.user, mentions)
        book_map = _build_user_book_map(request.user, mentions)
        serializer = LogicalBlockDetailSerializer(block, context={"edits_map": edits_map, "book_map": book_map})
        return Response(serializer.data)


class BookConceptsView(APIView):
    def get(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        if not user_book.global_cache:
            return Response({"count": 0, "results": []})
        mentions = ConceptMention.objects.filter(global_book=user_book.global_cache).select_related("concept", "logical_block", "global_book")
        q = request.query_params.get("q", "").strip()
        if q:
            normalized_q = normalize_concept_name(q)
            mentions = mentions.filter(Q(concept__name__icontains=q) | Q(concept__normalized_name__icontains=normalized_q))

        edits_map = _get_user_edits_map(request.user, mentions)
        book_map = _build_user_book_map(request.user, mentions)
        paginator = MentionPagination()
        page = paginator.paginate_queryset(mentions.order_by("-importance_score"), request)
        serializer = ConceptMentionSerializer(page, many=True, context={"edits_map": edits_map, "book_map": book_map})
        return paginator.get_paginated_response(serializer.data)


class AllConceptsView(APIView):
    def get(self, request):
        queryset = Concept.objects.filter(mentions__global_book__user_books__user=request.user).distinct()
        book_id = request.query_params.get("book_id")
        if book_id:
            queryset = queryset.filter(mentions__global_book__user_books__id=book_id)
        concepts = queryset.annotate(mentions_count=Count("mentions", distinct=True))
        results = []
        for concept in concepts:
            books = (
                concept.mentions.filter(global_book__user_books__user=request.user)
                .values_list("global_book__title", flat=True)
                .distinct()
            )
            results.append(
                {
                    **ConceptSerializer(concept).data,
                    "mentions_count": concept.mentions_count,
                    "books": list(books),
                }
            )
        return Response(results)


class ConceptDetailView(APIView):
    def get(self, request, concept_id):
        concept = get_object_or_404(Concept, id=concept_id)
        mentions = ConceptMention.objects.filter(
            concept=concept,
            global_book__user_books__user=request.user,
        ).select_related("concept", "logical_block", "global_book")
        if not mentions.exists():
            return Response({"detail": "Concept not found in your library."}, status=status.HTTP_404_NOT_FOUND)

        edits_map = _get_user_edits_map(request.user, mentions)
        book_map = _build_user_book_map(request.user, mentions)
        serializer = ConceptMentionSerializer(mentions, many=True, context={"edits_map": edits_map, "book_map": book_map})
        return Response(
            {
                "concept": ConceptSerializer(concept).data,
                "mentions": serializer.data,
                "similar_blocks": [
                    {
                        "block_id": item["block"].id,
                        "block_title": item["block"].title,
                        "book_title": item["block"].global_book.title,
                        "score": round(item["score"], 4),
                    }
                    for item in search_similar_blocks(concept.name, request.user.id, limit=5)
                ],
            }
        )


class ConceptSearchView(APIView):
    def get(self, request):
        q = request.query_params.get("q", "").strip()
        if not q:
            return Response([])
        normalized_q = normalize_concept_name(q)
        direct = Concept.objects.filter(
            mentions__global_book__user_books__user=request.user
        ).filter(Q(name__icontains=q) | Q(normalized_name__icontains=normalized_q)).distinct()

        rag_candidates = search_similar_concepts(q, request.user.id, limit=10)
        rag_ids = [item["concept"].id for item in rag_candidates]
        rag_map = {item["concept"].id: item["score"] for item in rag_candidates}
        merged = list(direct)
        for concept_id in rag_ids:
            concept = next((item["concept"] for item in rag_candidates if item["concept"].id == concept_id), None)
            if concept and concept not in merged:
                merged.append(concept)

        response_items = []
        for concept in merged:
            response_items.append(
                {
                    **ConceptSerializer(concept).data,
                    "similarity_score": round(float(rag_map.get(concept.id, 0.0)), 4),
                }
            )
        response_items.sort(key=lambda item: item["similarity_score"], reverse=True)
        return Response(response_items)


class ConceptMapView(APIView):
    def get(self, request):
        map_payload = build_user_concept_map(request.user.id)
        return Response(map_payload)


class ConceptCompareView(APIView):
    def get(self, request, concept_id):
        concept = get_object_or_404(Concept, id=concept_id)
        mentions = ConceptMention.objects.filter(
            concept=concept,
            global_book__user_books__user=request.user,
        ).select_related("global_book", "logical_block")
        if not mentions.exists():
            return Response({"detail": "Concept not found in your library."}, status=status.HTTP_404_NOT_FOUND)

        mention_payload = [
            {
                "book_title": mention.global_book.title,
                "block_title": mention.logical_block.title,
                "short_explanation": mention.short_explanation,
                "source_quote": mention.source_quote,
                "chapter_title": mention.logical_block.chapter_title,
            }
            for mention in mentions
        ]
        comparison = compare_concept_mentions(concept.name, mention_payload)
        return Response(
            {
                "concept": ConceptSerializer(concept).data,
                "mentions": mention_payload,
                "comparison": comparison,
            }
        )


class ConceptMentionEditView(APIView):
    def patch(self, request, mention_id):
        mention = get_object_or_404(
            ConceptMention.objects.filter(global_book__user_books__user=request.user).distinct(),
            id=mention_id,
        )
        serializer = UserConceptEditSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        edit, _ = UserConceptEdit.objects.update_or_create(
            user=request.user,
            concept_mention=mention,
            defaults={"custom_explanation": serializer.validated_data["custom_explanation"]},
        )
        response = ConceptMentionSerializer(
            mention,
            context={"edits_map": {mention.id: edit}, "book_map": {mention.global_book_id: mention.global_book.user_books.filter(user=request.user).first().id}},
        )
        return Response(response.data)


class ConceptMentionResetView(APIView):
    def post(self, request, mention_id):
        mention = get_object_or_404(
            ConceptMention.objects.filter(global_book__user_books__user=request.user).distinct(),
            id=mention_id,
        )
        UserConceptEdit.objects.filter(user=request.user, concept_mention=mention).delete()
        book = mention.global_book.user_books.filter(user=request.user).first()
        response = ConceptMentionSerializer(
            mention,
            context={"edits_map": {}, "book_map": {mention.global_book_id: book.id if book else None}},
        )
        return Response(response.data)


class StatsView(APIView):
    def get(self, request):
        books = UserBook.objects.filter(user=request.user)
        global_books = [item for item in books.values_list("global_cache_id", flat=True) if item]
        concepts_count = (
            Concept.objects.filter(mentions__global_book_id__in=global_books).distinct().count()
            if global_books
            else 0
        )
        blocks_count = LogicalBlock.objects.filter(global_book_id__in=global_books).count() if global_books else 0
        return Response(
            {
                "books_count": books.count(),
                "protected_books_count": books.filter(is_protected=True).count(),
                "max_books": MAX_BOOKS_PER_USER,
                "concepts_count": concepts_count,
                "logical_blocks_count": blocks_count,
            }
        )


class ExportBookView(APIView):
    def get(self, request, book_id):
        user_book = _get_user_book_or_404(request.user, book_id)
        if not user_book.global_cache:
            return Response({"detail": "Book has no analysis to export."}, status=status.HTTP_400_BAD_REQUEST)

        export_format = request.query_params.get("format", "pdf").lower()
        if export_format == "csv":
            content = export_csv(user_book)
            content_type = "text/csv; charset=utf-8"
            ext = "csv"
        elif export_format == "txt":
            content = export_txt(user_book)
            content_type = "text/plain; charset=utf-8"
            ext = "txt"
        elif export_format == "pdf":
            content = export_pdf(user_book)
            content_type = "application/pdf"
            ext = "pdf"
        elif export_format == "json":
            content = export_json(user_book)
            content_type = "application/json; charset=utf-8"
            ext = "json"
        else:
            return Response({"detail": "Supported formats: pdf, txt, csv, json."}, status=status.HTTP_400_BAD_REQUEST)

        response = HttpResponse(content, content_type=content_type)
        response["Content-Disposition"] = f'attachment; filename="book_{user_book.id}_analysis.{ext}"'
        return response
