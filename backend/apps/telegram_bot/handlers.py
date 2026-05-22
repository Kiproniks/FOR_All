from __future__ import annotations

from io import BytesIO
from secrets import token_urlsafe

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message
from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile

from apps.books.models import Concept, ConceptMention, LogicalBlock, UserBook
from apps.books.services.book_parser import (
    is_supported_book_extension,
    parse_uploaded_book,
    supported_extensions_text,
)
from apps.books.services.glossary_export import export_csv, export_pdf
from apps.books.services.hashing import sha256_bytes
from apps.books.services.llm_service import compare_concept_mentions
from apps.books.services.rotation import MAX_BOOKS_PER_USER, rotate_books_if_needed
from apps.books.tasks import analyze_book_task
from apps.telegram_bot.models import TelegramProfile

router = Router()


def _book_label(book: UserBook) -> str:
    lock = "🔒 " if book.is_protected else ""
    return f"{lock}{book.title or book.original_filename} [{book.status}]"


@sync_to_async
def get_or_create_user(message: Message):
    User = get_user_model()
    telegram_id = message.from_user.id
    username = message.from_user.username or ""
    profile = TelegramProfile.objects.filter(telegram_id=telegram_id).select_related("user").first()
    if profile:
        if username and profile.username != username:
            profile.username = username
            profile.save(update_fields=["username", "updated_at"])
        return profile.user

    email = f"tg_{telegram_id}@local.local"
    user, created = User.objects.get_or_create(email=email, defaults={"is_active": True})
    if created:
        user.set_password(token_urlsafe(16))
        user.save(update_fields=["password"])
    TelegramProfile.objects.create(user=user, telegram_id=telegram_id, username=username)
    return user


def _find_book_sync(user, name: str) -> UserBook | None:
    return (
        UserBook.objects.filter(user=user, title__icontains=name)
        .select_related("global_cache")
        .order_by("-uploaded_at")
        .first()
    )


@sync_to_async
def upload_book_for_user(user, filename: str, content: bytes):
    if len(content) > 50 * 1024 * 1024:
        return {"status": "error", "message": "Файл больше 50 МБ."}
    if not is_supported_book_extension(filename):
        return {"status": "error", "message": f"Поддерживаются файлы: {supported_extensions_text()}"}
    try:
        parse_uploaded_book(content, filename)
    except ValueError as exc:
        return {"status": "error", "message": f"Невалидный файл: {exc}"}

    rotation = rotate_books_if_needed(user, confirmed=True)
    if not rotation.can_upload:
        return {"status": "error", "message": rotation.reason}

    file_hash = sha256_bytes(content)
    cached_book = UserBook.objects.filter(file_hash=file_hash, global_cache__isnull=False).select_related("global_cache").first()
    if cached_book and cached_book.global_cache and cached_book.global_cache.logical_blocks.exists():
        user_book = UserBook.objects.create(
            user=user,
            global_cache=cached_book.global_cache,
            title=cached_book.global_cache.title,
            authors=cached_book.global_cache.authors,
            original_filename=filename,
            file_hash=file_hash,
            status=UserBook.Status.READY,
        )
        return {"status": "cached", "book_id": user_book.id}

    user_book = UserBook.objects.create(
        user=user,
        original_filename=filename,
        file_hash=file_hash,
        status=UserBook.Status.PROCESSING,
    )
    user_book.file.save(filename, ContentFile(content), save=True)
    analyze_book_task.delay(user_book.id)
    return {"status": "processing", "book_id": user_book.id}


@sync_to_async
def list_books(user):
    return list(UserBook.objects.filter(user=user).order_by("-uploaded_at")[:20])


@sync_to_async
def get_book_summary(user, book_name: str):
    book = _find_book_sync(user, book_name)
    if not book or not book.global_cache:
        return None
    return {
        "title": book.title,
        "summary": book.global_cache.full_summary or "Конспект пока пуст.",
        "concepts_count": book.global_cache.concept_mentions.count(),
    }


@sync_to_async
def get_book_blocks(user, book_name: str):
    book = _find_book_sync(user, book_name)
    if not book or not book.global_cache:
        return None
    blocks = list(
        LogicalBlock.objects.filter(global_book=book.global_cache)
        .order_by("order_number")
        .values("id", "title", "short_summary")
    )
    return book, blocks[:20]


@sync_to_async
def get_book_concepts(user, book_name: str):
    book = _find_book_sync(user, book_name)
    if not book or not book.global_cache:
        return None
    mentions = list(
        ConceptMention.objects.filter(global_book=book.global_cache)
        .select_related("concept")
        .order_by("-importance_score")[:20]
    )
    return book, mentions


@sync_to_async
def get_concept_card(user, concept_name: str):
    concept = Concept.objects.filter(name__icontains=concept_name).first()
    if not concept:
        return None
    mentions = list(
        ConceptMention.objects.filter(concept=concept, global_book__user_books__user=user)
        .select_related("global_book", "logical_block")
        .order_by("-importance_score")[:10]
    )
    if not mentions:
        return None
    return concept, mentions


@sync_to_async
def search_concepts(user, query: str):
    return list(
        Concept.objects.filter(mentions__global_book__user_books__user=user, name__icontains=query)
        .distinct()[:20]
    )


@sync_to_async
def compare_concept_for_user(user, concept_name: str):
    concept = Concept.objects.filter(name__icontains=concept_name).first()
    if not concept:
        return None
    mentions = list(
        ConceptMention.objects.filter(concept=concept, global_book__user_books__user=user)
        .select_related("global_book", "logical_block")
        .order_by("-importance_score")
    )
    if len(mentions) < 2:
        return concept, mentions, "Недостаточно источников для сравнения."
    payload = [
        {
            "book_title": mention.global_book.title,
            "block_title": mention.logical_block.title,
            "short_explanation": mention.short_explanation,
            "source_quote": mention.source_quote,
        }
        for mention in mentions
    ]
    return concept, mentions, compare_concept_mentions(concept.name, payload)


@sync_to_async
def user_stats(user):
    books = UserBook.objects.filter(user=user)
    global_ids = [item for item in books.values_list("global_cache_id", flat=True) if item]
    concepts_count = (
        Concept.objects.filter(mentions__global_book_id__in=global_ids).distinct().count()
        if global_ids
        else 0
    )
    blocks_count = LogicalBlock.objects.filter(global_book_id__in=global_ids).count() if global_ids else 0
    return {
        "books": books.count(),
        "protected": books.filter(is_protected=True).count(),
        "remaining": max(0, MAX_BOOKS_PER_USER - books.count()),
        "concepts": concepts_count,
        "blocks": blocks_count,
    }


@sync_to_async
def toggle_protect(user, book_name: str):
    book = _find_book_sync(user, book_name)
    if not book:
        return None
    book.is_protected = not book.is_protected
    book.save(update_fields=["is_protected"])
    return book


@sync_to_async
def export_book(user, book_name: str):
    book = _find_book_sync(user, book_name)
    if not book or not book.global_cache:
        return None
    return book, export_pdf(book), export_csv(book)


@router.message(Command("start"))
async def start_command(message: Message):
    await get_or_create_user(message)
    await message.answer(
        "Я помогу загрузить FB2/PDF-книгу, сделать конспект, выделить концепты и сравнить их между книгами."
    )


@router.message(Command("upload"))
async def upload_command(message: Message):
    await get_or_create_user(message)
    await message.answer("Отправьте FB2 или PDF-файл сообщением.")


@router.message(F.document)
async def file_handler(message: Message):
    user = await get_or_create_user(message)
    document = message.document
    if not document.file_name or not is_supported_book_extension(document.file_name):
        await message.answer(f"Ошибка: поддерживаются файлы {supported_extensions_text()}")
        return
    if document.file_size and document.file_size > 50 * 1024 * 1024:
        await message.answer("Ошибка: файл превышает 50 МБ.")
        return

    file = await message.bot.get_file(document.file_id)
    bio = BytesIO()
    await message.bot.download(file, destination=bio)
    result = await upload_book_for_user(user, document.file_name, bio.getvalue())
    if result["status"] == "processing":
        await message.answer("Книга загружена, анализ начался.")
    elif result["status"] == "cached":
        await message.answer("Книга уже была проанализирована, конспект готов.")
    else:
        await message.answer(f"Ошибка обработки: {result['message']}")


@router.message(Command("my_books"))
async def my_books_command(message: Message):
    user = await get_or_create_user(message)
    books = await list_books(user)
    if not books:
        await message.answer("Книг пока нет.")
        return
    lines = ["Ваши книги:"]
    for book in books:
        concepts_count = book.global_cache.concept_mentions.count() if book.global_cache else 0
        lines.append(f"- {_book_label(book)} | concepts: {concepts_count}")
    await message.answer("\n".join(lines))


@router.message(Command("summary"))
async def summary_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /summary <название книги>")
        return
    data = await get_book_summary(user, command.args)
    if not data:
        await message.answer("Книга не найдена или еще не проанализирована.")
        return
    text = data["summary"][:3500]
    await message.answer(f"Конспект книги '{data['title']}':\n\n{text}")


@router.message(Command("blocks"))
async def blocks_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /blocks <название книги>")
        return
    payload = await get_book_blocks(user, command.args)
    if not payload:
        await message.answer("Книга не найдена.")
        return
    book, blocks = payload
    if not blocks:
        await message.answer(f"В книге '{book.title}' блоки пока не найдены.")
        return
    lines = [f"Логические блоки книги: {book.title}"]
    for index, block in enumerate(blocks[:20], start=1):
        lines.append(f"{index}. {block['title']} — {str(block['short_summary'])[:120]}")
    await message.answer("\n".join(lines))


@router.message(Command("concepts"))
async def concepts_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /concepts <название книги>")
        return
    payload = await get_book_concepts(user, command.args)
    if not payload:
        await message.answer("Книга не найдена.")
        return
    book, mentions = payload
    if not mentions:
        await message.answer(f"В книге '{book.title}' концепты пока не найдены.")
        return
    lines = [f"Концепты книги: {book.title}"]
    for index, mention in enumerate(mentions[:20], start=1):
        lines.append(f"{index}. {mention.concept.name}: {mention.short_explanation[:100]}")
    await message.answer("\n".join(lines))


@router.message(Command("concept"))
async def concept_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /concept <название концепта>")
        return
    payload = await get_concept_card(user, command.args)
    if not payload:
        await message.answer("Концепт не найден.")
        return
    concept, mentions = payload
    lines = [f"Концепт: {concept.name}", f"Описание: {concept.description or '-'}", "Упоминания:"]
    for item in mentions[:10]:
        lines.append(f"- {item.global_book.title} / {item.logical_block.title}: {item.short_explanation[:120]}")
    await message.answer("\n".join(lines))


@router.message(Command("search"))
async def search_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /search <запрос>")
        return
    concepts = await search_concepts(user, command.args)
    if not concepts:
        await message.answer("Совпадений не найдено.")
        return
    await message.answer("Результаты:\n" + "\n".join([f"- {concept.name}" for concept in concepts]))


@router.message(Command("compare"))
async def compare_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /compare <концепт>")
        return
    payload = await compare_concept_for_user(user, command.args)
    if not payload:
        await message.answer("Концепт не найден.")
        return
    concept, mentions, comparison = payload
    short_sources = "\n".join([f"- {item.global_book.title}: {item.logical_block.title}" for item in mentions[:10]])
    await message.answer(f"Сравнение концепта '{concept.name}'\n\nИсточники:\n{short_sources}\n\n{comparison[:3000]}")


@router.message(Command("stats"))
async def stats_command(message: Message):
    user = await get_or_create_user(message)
    stats = await user_stats(user)
    await message.answer(
        f"Книг: {stats['books']}/{MAX_BOOKS_PER_USER}\n"
        f"Защищенных: {stats['protected']}\n"
        f"Концептов: {stats['concepts']}\n"
        f"Логических блоков: {stats['blocks']}\n"
        f"Свободных мест: {stats['remaining']}"
    )


@router.message(Command("protect"))
async def protect_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /protect <название книги>")
        return
    book = await toggle_protect(user, command.args)
    if not book:
        await message.answer("Книга не найдена.")
        return
    state = "включена" if book.is_protected else "выключена"
    await message.answer(f"Защита книги '{book.title or book.original_filename}' {state}.")


@router.message(Command("export"))
async def export_command(message: Message, command: CommandObject):
    user = await get_or_create_user(message)
    if not command.args:
        await message.answer("Использование: /export <название книги>")
        return
    payload = await export_book(user, command.args)
    if not payload:
        await message.answer("Книга не найдена или анализ еще не готов.")
        return
    book, pdf_data, csv_data = payload
    base = (book.title or f"book_{book.id}").replace(" ", "_")
    await message.answer_document(BufferedInputFile(pdf_data, filename=f"{base}.pdf"))
    await message.answer_document(BufferedInputFile(csv_data, filename=f"{base}.csv"))
