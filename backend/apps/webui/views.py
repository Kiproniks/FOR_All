from __future__ import annotations

import os

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.db.models import Count, F, Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from apps.accounts.models import User
from apps.books.models import BookTheme, Concept, ConceptMention, LogicalBlock, UserBook, UserConceptEdit
from apps.books.services.book_parser import (
    is_supported_book_extension,
    parse_uploaded_book,
    supported_extensions_text,
)
from apps.books.services.concept_map import build_user_concept_map
from apps.books.services.concept_normalizer import normalize_concept_name
from apps.books.services.glossary_export import export_csv, export_json, export_pdf, export_txt
from apps.books.services.hashing import sha256_bytes
from apps.books.services.llm_service import compare_concept_mentions
from apps.books.services.rotation import (
    MAX_BOOKS_PER_USER,
    get_oldest_unprotected_book,
    get_user_books_count,
    rotate_books_if_needed,
)
from apps.books.tasks import analyze_book_task
from .forms import ConceptEditForm, LoginForm, RegisterForm


def _redirect_for_authenticated_user(request):
    if request.user.is_authenticated:
        return redirect("webui:library")
    return None


def _get_user_book_or_404(request, book_id: int) -> UserBook:
    return get_object_or_404(UserBook.objects.select_related("global_cache"), id=book_id, user=request.user)


def _get_owned_mention_or_404(request, mention_id: int) -> ConceptMention:
    return get_object_or_404(
        ConceptMention.objects.filter(global_book__user_books__user=request.user).distinct(),
        id=mention_id,
    )


@require_GET
def root_redirect(request):
    if request.user.is_authenticated:
        return redirect("webui:library")
    return redirect("webui:login")


def login_view(request):
    redirect_response = _redirect_for_authenticated_user(request)
    if redirect_response:
        return redirect_response

    form = LoginForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = authenticate(
            request=request,
            email=form.cleaned_data["email"].strip().lower(),
            password=form.cleaned_data["password"],
        )
        if user is None:
            form.add_error(None, "Invalid credentials.")
        else:
            login(request, user)
            return redirect("webui:library")
    return render(request, "webui/login.html", {"form": form})


def register_view(request):
    redirect_response = _redirect_for_authenticated_user(request)
    if redirect_response:
        return redirect_response

    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = User.objects.create_user(
            email=form.cleaned_data["email"],
            password=form.cleaned_data["password1"],
        )
        login(request, user)
        messages.success(request, "Account created.")
        return redirect("webui:library")
    return render(request, "webui/register.html", {"form": form})


@require_POST
def logout_view(request):
    logout(request)
    return redirect("webui:login")


@login_required(login_url="webui:login")
def library_view(request):
    books = (
        UserBook.objects.filter(user=request.user)
        .select_related("global_cache")
        .order_by("-uploaded_at")
    )
    books_count = books.count()
    protected_count = books.filter(is_protected=True).count()
    oldest_unprotected = get_oldest_unprotected_book(request.user) if books_count >= MAX_BOOKS_PER_USER else None

    for book in books:
        if book.global_cache_id:
            book.logical_blocks_count = book.global_cache.logical_blocks.count()
            book.concepts_count = book.global_cache.concept_mentions.count()
        else:
            book.logical_blocks_count = 0
            book.concepts_count = 0

    return render(
        request,
        "webui/library.html",
        {
            "books": books,
            "books_count": books_count,
            "protected_count": protected_count,
            "max_books": MAX_BOOKS_PER_USER,
            "oldest_unprotected": oldest_unprotected,
        },
    )


@login_required(login_url="webui:login")
@require_POST
def upload_books_view(request):
    files = request.FILES.getlist("files")
    if not files:
        messages.error(request, "Select at least one file.")
        return redirect("webui:library")

    confirm_rotation = str(request.POST.get("confirm_rotation", "")).strip().lower() in {"1", "true", "yes", "on"}
    current_count = get_user_books_count(request.user)
    if current_count + len(files) > MAX_BOOKS_PER_USER and not confirm_rotation:
        candidate = get_oldest_unprotected_book(request.user)
        if candidate:
            messages.warning(
                request,
                f"Limit is {MAX_BOOKS_PER_USER}. Confirm rotation to delete oldest unprotected book: "
                f"{candidate.title or candidate.original_filename}.",
            )
        else:
            messages.error(request, "Limit reached and all books are protected.")
        return redirect("webui:library")

    uploaded_ready = 0
    uploaded_processing = 0
    failed = 0

    for uploaded_file in files:
        if uploaded_file.size > settings.MAX_FB2_FILE_SIZE:
            messages.error(request, f"{uploaded_file.name}: file exceeds 50 MB limit.")
            failed += 1
            continue

        if not is_supported_book_extension(uploaded_file.name):
            messages.error(
                request,
                f"{uploaded_file.name}: unsupported extension. Supported: {supported_extensions_text()}."
            )
            failed += 1
            continue

        if get_user_books_count(request.user) >= MAX_BOOKS_PER_USER:
            rotation = rotate_books_if_needed(request.user, confirmed=confirm_rotation)
            if not rotation.can_upload:
                messages.error(request, rotation.reason or "Unable to rotate books.")
                failed += 1
                continue

        content = uploaded_file.read()
        if not content:
            messages.error(request, f"{uploaded_file.name}: empty file.")
            failed += 1
            continue

        try:
            parse_uploaded_book(content, uploaded_file.name)
        except ValueError as exc:
            messages.error(request, f"{uploaded_file.name}: invalid file ({exc}).")
            failed += 1
            continue

        file_hash = sha256_bytes(content)
        original_filename = os.path.basename(uploaded_file.name)
        cached = (
            UserBook.objects.filter(file_hash=file_hash)
            .exclude(global_cache=None)
            .select_related("global_cache")
            .first()
        )
        global_cache = cached.global_cache if cached else None
        has_cache_result = bool(global_cache and global_cache.logical_blocks.exists() and global_cache.themes.exists())

        if has_cache_result:
            UserBook.objects.create(
                user=request.user,
                global_cache=global_cache,
                title=global_cache.title,
                authors=global_cache.authors,
                original_filename=original_filename,
                file_hash=file_hash,
                status=UserBook.Status.READY,
            )
            uploaded_ready += 1
            continue

        user_book = UserBook.objects.create(
            user=request.user,
            global_cache=global_cache,
            title="",
            authors="",
            original_filename=original_filename,
            file_hash=file_hash,
            status=UserBook.Status.PROCESSING,
        )
        user_book.file.save(original_filename, ContentFile(content), save=True)
        analyze_book_task.delay(user_book.id, force_reanalyze=bool(global_cache))
        uploaded_processing += 1

    if uploaded_ready:
        messages.success(request, f"Added {uploaded_ready} file(s) from global cache.")
    if uploaded_processing:
        messages.success(request, f"Started background analysis for {uploaded_processing} file(s).")
    if failed and not (uploaded_ready or uploaded_processing):
        messages.error(request, "No files were uploaded.")
    elif failed:
        messages.warning(request, f"{failed} file(s) failed validation.")
    return redirect("webui:library")


@login_required(login_url="webui:login")
@require_POST
def delete_book_view(request, book_id: int):
    book = _get_user_book_or_404(request, book_id)
    if book.is_protected:
        messages.error(request, "Disable protection before deleting this book.")
        return redirect("webui:library")
    book.delete()
    messages.success(request, "Book deleted.")
    return redirect("webui:library")


@login_required(login_url="webui:login")
@require_POST
def protect_book_view(request, book_id: int):
    book = _get_user_book_or_404(request, book_id)
    book.is_protected = not book.is_protected
    book.save(update_fields=["is_protected"])
    messages.success(request, f"Protection set to {book.is_protected}.")
    return redirect("webui:library")


@login_required(login_url="webui:login")
@require_POST
def reanalyze_book_view(request, book_id: int):
    book = _get_user_book_or_404(request, book_id)
    book.status = UserBook.Status.PROCESSING
    book.error_message = ""
    book.save(update_fields=["status", "error_message"])
    analyze_book_task.delay(book.id, force_reanalyze=True)
    messages.success(request, "Reanalysis started.")
    return redirect("webui:library")


@login_required(login_url="webui:login")
def book_summary_view(request, book_id: int):
    book = _get_user_book_or_404(request, book_id)
    UserBook.objects.filter(id=book.id).update(views_count=F("views_count") + 1)
    book.refresh_from_db(fields=["views_count"])

    blocks = []
    mentions = []
    themes = []
    if book.global_cache_id:
        blocks = (
            LogicalBlock.objects.filter(global_book_id=book.global_cache_id)
            .prefetch_related("concept_mentions__concept")
            .order_by("order_number")
        )
        themes = (
            BookTheme.objects.filter(global_book_id=book.global_cache_id)
            .prefetch_related("subtopics")
            .order_by("order_number")
        )
        mentions = (
            ConceptMention.objects.filter(global_book_id=book.global_cache_id)
            .select_related("concept", "logical_block")
            .order_by("-importance_score", "concept__name")
        )

    return render(
        request,
        "webui/book_summary.html",
        {
            "book": book,
            "summary": book.global_cache.full_summary if book.global_cache_id else "",
            "blocks": blocks,
            "themes": themes,
            "mentions": mentions,
        },
    )


@login_required(login_url="webui:login")
def block_detail_view(request, book_id: int, block_id: int):
    book = _get_user_book_or_404(request, book_id)
    if not book.global_cache_id:
        messages.error(request, "Book is not analyzed yet.")
        return redirect("webui:book-summary", book_id=book.id)

    block = get_object_or_404(
        LogicalBlock.objects.filter(global_book_id=book.global_cache_id),
        id=block_id,
    )
    mentions = (
        ConceptMention.objects.filter(logical_block=block)
        .select_related("concept")
        .order_by("-importance_score", "concept__name")
    )
    edits_map = {
        edit.concept_mention_id: edit
        for edit in UserConceptEdit.objects.filter(user=request.user, concept_mention_id__in=mentions.values_list("id", flat=True))
    }
    for mention in mentions:
        mention.custom_explanation = edits_map.get(mention.id).custom_explanation if mention.id in edits_map else ""
    return render(
        request,
        "webui/block_detail.html",
        {"book": book, "block": block, "mentions": mentions},
    )


@login_required(login_url="webui:login")
def all_concepts_view(request):
    q = request.GET.get("q", "").strip()
    concepts = Concept.objects.filter(mentions__global_book__user_books__user=request.user).distinct()
    if q:
        normalized_q = normalize_concept_name(q)
        concepts = concepts.filter(Q(name__icontains=q) | Q(normalized_name__icontains=normalized_q))

    concepts = concepts.annotate(mentions_count=Count("mentions", distinct=True)).order_by("name")
    concepts_with_books = []
    for concept in concepts:
        books = (
            concept.mentions.filter(global_book__user_books__user=request.user)
            .values_list("global_book__title", flat=True)
            .distinct()
        )
        concepts_with_books.append((concept, list(books)))

    return render(
        request,
        "webui/concepts.html",
        {"concepts_with_books": concepts_with_books, "query": q},
    )


@login_required(login_url="webui:login")
def concept_map_view(request):
    map_payload = build_user_concept_map(request.user.id)
    return render(request, "webui/concept_map.html", {"map_payload": map_payload})


@login_required(login_url="webui:login")
def concept_detail_view(request, concept_id: int):
    concept = get_object_or_404(Concept, id=concept_id)
    mentions = (
        ConceptMention.objects.filter(concept=concept, global_book__user_books__user=request.user)
        .select_related("global_book", "logical_block")
        .order_by("-importance_score", "id")
    )
    if not mentions.exists():
        messages.error(request, "Concept not found in your library.")
        return redirect("webui:concepts")

    edits_map = {
        edit.concept_mention_id: edit
        for edit in UserConceptEdit.objects.filter(user=request.user, concept_mention_id__in=mentions.values_list("id", flat=True))
    }
    book_map = {}
    for row in UserBook.objects.filter(
        user=request.user,
        global_cache_id__in=mentions.values_list("global_book_id", flat=True).distinct(),
    ).values("id", "global_cache_id"):
        if row["global_cache_id"] not in book_map:
            book_map[row["global_cache_id"]] = row["id"]

    for mention in mentions:
        mention.custom_explanation = edits_map.get(mention.id).custom_explanation if mention.id in edits_map else ""
        mention.user_book_id = book_map.get(mention.global_book_id)

    return render(
        request,
        "webui/concept_detail.html",
        {
            "concept": concept,
            "mentions": mentions,
        },
    )


@login_required(login_url="webui:login")
def concept_compare_view(request, concept_id: int):
    concept = get_object_or_404(Concept, id=concept_id)
    mentions = (
        ConceptMention.objects.filter(concept=concept, global_book__user_books__user=request.user)
        .select_related("global_book", "logical_block")
        .order_by("-importance_score", "id")
    )
    if not mentions.exists():
        messages.error(request, "Concept not found in your library.")
        return redirect("webui:concepts")

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
    return render(
        request,
        "webui/concept_compare.html",
        {"concept": concept, "mentions": mentions, "comparison": comparison},
    )


@login_required(login_url="webui:login")
@require_POST
def edit_mention_view(request, mention_id: int):
    mention = _get_owned_mention_or_404(request, mention_id)
    form = ConceptEditForm(request.POST)
    if form.is_valid():
        UserConceptEdit.objects.update_or_create(
            user=request.user,
            concept_mention=mention,
            defaults={"custom_explanation": form.cleaned_data["custom_explanation"]},
        )
        messages.success(request, "Custom explanation saved.")
    else:
        messages.error(request, "Invalid custom explanation.")
    return redirect("webui:concept-detail", concept_id=mention.concept_id)


@login_required(login_url="webui:login")
@require_POST
def reset_mention_view(request, mention_id: int):
    mention = _get_owned_mention_or_404(request, mention_id)
    UserConceptEdit.objects.filter(user=request.user, concept_mention=mention).delete()
    messages.success(request, "Custom explanation reset.")
    return redirect("webui:concept-detail", concept_id=mention.concept_id)


@login_required(login_url="webui:login")
def export_book_web_view(request, book_id: int, fmt: str):
    user_book = _get_user_book_or_404(request, book_id)
    if not user_book.global_cache_id:
        messages.error(request, "Book has no analysis yet.")
        return redirect("webui:book-summary", book_id=book_id)

    export_format = fmt.lower()
    if export_format == "csv":
        content = export_csv(user_book)
        content_type = "text/csv; charset=utf-8"
    elif export_format == "txt":
        content = export_txt(user_book)
        content_type = "text/plain; charset=utf-8"
    elif export_format == "pdf":
        content = export_pdf(user_book)
        content_type = "application/pdf"
    elif export_format == "json":
        content = export_json(user_book)
        content_type = "application/json; charset=utf-8"
    else:
        messages.error(request, "Supported formats: pdf, txt, csv, json.")
        return redirect("webui:book-summary", book_id=book_id)

    response = HttpResponse(content, content_type=content_type)
    response["Content-Disposition"] = f'attachment; filename="book_{user_book.id}_analysis.{export_format}"'
    return response
