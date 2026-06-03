from __future__ import annotations

import re
from collections import OrderedDict
from html import escape
from typing import Iterable

from django.db import transaction
from django.utils.safestring import mark_safe

from apps.books.models import (
    BookStudyNotes,
    BookTheme,
    ConceptMention,
    LLMAnalysisRun,
    LLMChapterAnalysis,
    LogicalBlock,
    ThemeSubtopic,
    UserBook,
)


NOTES_MODEL_NAME = "cached-analysis-study-notes-v1"


def _clean_line(value: str, *, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def _unique_items(values: Iterable[str], *, limit: int = 8, item_limit: int = 220) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _clean_line(value, limit=item_limit)
        key = item.lower()
        if not item or key in seen:
            continue
        result.append(item)
        seen.add(key)
        if len(result) >= limit:
            break
    return result


def _bullet_items(values: Iterable[str], *, limit: int = 8) -> str:
    items = _unique_items(values, limit=limit)
    return "\n".join(f"- {item}" for item in items) or "- Нет данных."


def _latest_source_run(book: UserBook) -> LLMAnalysisRun | None:
    if not book.global_cache_id:
        return None
    return (
        LLMAnalysisRun.objects.filter(global_book_id=book.global_cache_id, mode="llm_fast_batched")
        .order_by("-created_at")
        .first()
    )


def _chapter_notes(
    theme: BookTheme,
    blocks: list[LogicalBlock],
    subtopics: list[ThemeSubtopic],
    chapter_analyses_by_title: dict[str, LLMChapterAnalysis],
) -> str:
    related_blocks = [
        block
        for block in blocks
        if theme.start_block_number <= block.order_number <= theme.end_block_number
    ]
    chapter_analysis = chapter_analyses_by_title.get((theme.chapter_title or theme.title).strip().lower())
    ideas = []
    if chapter_analysis:
        ideas.append(chapter_analysis.chapter_summary)
        ideas.extend(chapter_analysis.main_topics or [])
    ideas.extend([theme.summary] + [block.short_summary for block in related_blocks[:4]])

    concepts = []
    if chapter_analysis:
        concepts.extend(chapter_analysis.key_terms or [])
    concepts.extend(item.name for item in subtopics[:10])

    remember = []
    for block in related_blocks[:4]:
        if block.title:
            remember.append(f"{block.order_number}. {block.title}: {_clean_line(block.short_summary, limit=260)}")

    return "\n".join(
        [
            f"### {theme.chapter_title or theme.title}",
            "",
            "**Основные идеи:**",
            _bullet_items(ideas, limit=6),
            "",
            "**Ключевые понятия:**",
            _bullet_items(concepts, limit=10),
            "",
            "**Что важно запомнить:**",
            _bullet_items(remember, limit=5),
            "",
        ]
    )


def build_book_study_notes_markdown(book: UserBook) -> str:
    cache = book.global_cache
    if not cache:
        raise ValueError("Book has no analysis cache.")

    themes = list(
        BookTheme.objects.filter(global_book=cache)
        .prefetch_related("subtopics")
        .order_by("order_number", "id")
    )
    blocks = list(LogicalBlock.objects.filter(global_book=cache).order_by("order_number"))
    chapter_analyses = list(
        LLMChapterAnalysis.objects.filter(global_book=cache, mode="llm_fast_batched")
        .order_by("chapter_index", "id")
    )
    chapter_analyses_by_title = {
        item.chapter_title.strip().lower(): item
        for item in chapter_analyses
        if item.chapter_title
    }
    mentions = list(
        ConceptMention.objects.filter(global_book=cache)
        .select_related("concept", "logical_block")
        .order_by("-importance_score", "concept__name")[:50]
    )
    subtopics_by_theme = {
        theme.id: list(theme.subtopics.all().order_by("-importance_score", "id"))
        for theme in themes
    }

    if not themes and not blocks:
        raise ValueError("Book has no ready themes or logical blocks.")

    title = book.title or cache.title or book.original_filename
    book_analysis = getattr(cache, "llm_book_analysis", None)
    book_summary = _clean_line(getattr(book_analysis, "book_summary", ""), limit=1400)
    if not book_summary:
        book_summary = _clean_line(cache.full_summary, limit=1400)
    if not book_summary:
        book_summary = (
            "Книга уже проанализирована, но общий итоговый summary не найден. "
            "Ниже конспект собран из тем, логических блоков и понятий."
        )

    chapter_map = [
        _chapter_notes(theme, blocks, subtopics_by_theme.get(theme.id, []), chapter_analyses_by_title)
        for theme in themes
    ]

    key_themes = [
        f"{theme.order_number}. {theme.title}: {_clean_line(theme.summary, limit=280)}"
        for theme in themes
    ]
    if book_analysis and getattr(book_analysis, "global_themes", None):
        key_themes = list(book_analysis.global_themes) + key_themes

    concept_rows = []
    seen_concepts: OrderedDict[str, str] = OrderedDict()
    for mention in mentions:
        name = _clean_line(mention.concept.name, limit=120)
        explanation = _clean_line(mention.short_explanation or mention.logical_block.short_summary, limit=300)
        if name and name not in seen_concepts:
            seen_concepts[name] = explanation
    for name, explanation in list(seen_concepts.items())[:28]:
        concept_rows.append(f"| {name} | {explanation or 'Связано с ключевыми блоками книги.'} |")

    theme_links = []
    for left, right in zip(themes, themes[1:]):
        theme_links.append(
            f"{left.title} -> {right.title}: материал последовательно переходит к следующему уровню модели, технологии или способу работы сети."
        )

    learning_path = []
    if book_analysis and getattr(book_analysis, "learning_path", None):
        learning_path = list(book_analysis.learning_path)
    if not learning_path:
        learning_path = [
            "Сначала прочитать общий обзор книги и понять, зачем нужны описанные системы или методы.",
            "Затем изучать главы по порядку, потому что каждая следующая тема опирается на предыдущую.",
            "После каждой главы выписывать ключевые понятия и проверять их по исходным logical blocks.",
            "В конце сопоставить темы между собой через карту и повторить основные концепты.",
        ]

    cheat_sheet = []
    for theme in themes[:14]:
        subtopics = ", ".join(item.name for item in subtopics_by_theme.get(theme.id, [])[:4])
        cheat_sheet.append(f"{theme.title}: {subtopics or _clean_line(theme.summary, limit=180)}")

    return "\n".join(
        [
            f"# Конспект книги: {title}",
            "",
            "## 1. О чём книга",
            book_summary,
            "",
            "## 2. Карта глав",
            "",
            "\n".join(chapter_map) if chapter_map else "Нет данных по главам.",
            "## 3. Ключевые темы",
            _bullet_items(key_themes, limit=18),
            "",
            "## 4. Основные понятия",
            "| Понятие | Краткое объяснение |",
            "|---|---|",
            "\n".join(concept_rows) if concept_rows else "| Нет данных | Нет данных |",
            "",
            "## 5. Связи между темами",
            _bullet_items(theme_links, limit=14),
            "",
            "## 6. Что учить в первую очередь",
            _bullet_items(learning_path, limit=8),
            "",
            "## 7. Краткая шпаргалка",
            _bullet_items(cheat_sheet, limit=14),
            "",
        ]
    )


@transaction.atomic
def generate_book_study_notes(book_id: int, *, force: bool = False) -> BookStudyNotes:
    book = UserBook.objects.select_related("global_cache").get(id=book_id)
    notes, _ = BookStudyNotes.objects.select_for_update().get_or_create(book=book)

    if notes.status == BookStudyNotes.Status.READY and notes.content_markdown and not force:
        return notes

    notes.status = BookStudyNotes.Status.GENERATING
    notes.error_message = ""
    notes.model_name = NOTES_MODEL_NAME
    notes.source_run = _latest_source_run(book)
    notes.save(update_fields=["status", "error_message", "model_name", "source_run", "updated_at"])

    try:
        notes.content_markdown = build_book_study_notes_markdown(book)
        notes.status = BookStudyNotes.Status.READY
        notes.error_message = ""
    except Exception as exc:
        notes.status = BookStudyNotes.Status.FAILED
        notes.error_message = str(exc)[:2000]
    notes.save(update_fields=["content_markdown", "status", "error_message", "updated_at"])
    return notes


def render_markdown_basic(markdown: str):
    html: list[str] = []
    in_ul = False
    in_table = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            html.append("</ul>")
            in_ul = False

    def close_table():
        nonlocal in_table
        if in_table:
            html.append("</tbody></table>")
            in_table = False

    for raw_line in str(markdown or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            close_ul()
            close_table()
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            close_ul()
            cells = [escape(cell.strip()) for cell in stripped.strip("|").split("|")]
            if all(set(cell) <= {"-"} for cell in cells):
                continue
            if not in_table:
                html.append('<table class="notes-table"><tbody>')
                in_table = True
            html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
            continue

        close_table()
        if stripped.startswith("#"):
            close_ul()
            level = min(4, len(stripped) - len(stripped.lstrip("#")))
            text = stripped[level:].strip()
            html.append(f"<h{level}>{escape(text)}</h{level}>")
            continue

        if stripped.startswith("- "):
            if not in_ul:
                html.append("<ul>")
                in_ul = True
            html.append(f"<li>{escape(stripped[2:].strip())}</li>")
            continue

        close_ul()
        text = escape(stripped)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        html.append(f"<p>{text}</p>")

    close_ul()
    close_table()
    return mark_safe("\n".join(html))
