from __future__ import annotations

import math
from typing import Any

from apps.books.models import BookTheme, ThemeSubtopic, UserBook
from apps.books.services.rag_service import cosine_similarity, create_embedding


def _circle_positions(count: int, center_x: float, center_y: float, radius: float, start_angle: float = 0.0) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    if count == 1:
        return [(center_x, center_y)]
    result: list[tuple[float, float]] = []
    for idx in range(count):
        angle = start_angle + (2 * math.pi * idx / count)
        result.append((center_x + math.cos(angle) * radius, center_y + math.sin(angle) * radius))
    return result


def _theme_text_for_embedding(theme: BookTheme, subtopics: list[ThemeSubtopic]) -> str:
    names = ", ".join(item.name for item in subtopics[:4])
    return f"{theme.title}. {theme.summary}. {theme.chapter_title}. {names}"[:3200]


def _build_related_theme_edges(theme_nodes: list[dict[str, Any]], theme_texts: dict[str, str]) -> list[dict[str, Any]]:
    if len(theme_nodes) < 2:
        return []

    # Keep complexity bounded for interactive map.
    if len(theme_nodes) > 220:
        theme_nodes = theme_nodes[:220]

    embeddings = {node["id"]: create_embedding(theme_texts.get(node["id"], node["label"])) for node in theme_nodes}

    edges: list[dict[str, Any]] = []
    for idx, left in enumerate(theme_nodes):
        for right in theme_nodes[idx + 1 :]:
            if left["global_book_id"] == right["global_book_id"]:
                continue
            score = cosine_similarity(embeddings[left["id"]], embeddings[right["id"]])
            if score < 0.82:
                continue
            edges.append(
                {
                    "source": left["id"],
                    "target": right["id"],
                    "weight": round(float(score), 4),
                    "type": "related",
                }
            )

    edges.sort(key=lambda item: item["weight"], reverse=True)
    return edges[:180]


def build_user_concept_map(user_id: int) -> dict[str, Any]:
    user_books = list(
        UserBook.objects.filter(user_id=user_id, global_cache__isnull=False)
        .select_related("global_cache")
        .order_by("title", "id")
    )
    if not user_books:
        return {
            "nodes": [],
            "edges": [],
            "books": [],
            "meta": {
                "books_count": 0,
                "themes_count": 0,
                "subtopics_count": 0,
            },
        }

    book_by_global: dict[int, UserBook] = {}
    for book in user_books:
        if book.global_cache_id and book.global_cache_id not in book_by_global:
            book_by_global[book.global_cache_id] = book

    global_ids = list(book_by_global.keys())
    themes = list(
        BookTheme.objects.filter(global_book_id__in=global_ids)
        .select_related("global_book")
        .prefetch_related("subtopics")
        .order_by("global_book_id", "order_number", "id")
    )
    if not themes:
        return {
            "nodes": [],
            "edges": [],
            "books": [],
            "meta": {
                "books_count": len(book_by_global),
                "themes_count": 0,
                "subtopics_count": 0,
            },
        }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    book_nodes: dict[int, dict[str, Any]] = {}
    themes_by_book: dict[int, list[BookTheme]] = {}
    subtopics_by_theme: dict[int, list[ThemeSubtopic]] = {}
    theme_nodes: list[dict[str, Any]] = []
    theme_texts: dict[str, str] = {}

    for theme in themes:
        themes_by_book.setdefault(theme.global_book_id, []).append(theme)
        subtopics_by_theme[theme.id] = list(theme.subtopics.all().order_by("-importance_score", "id"))

    themed_global_ids = [item for item in global_ids if themes_by_book.get(item)]
    if not themed_global_ids:
        return {
            "nodes": [],
            "edges": [],
            "books": [],
            "meta": {
                "books_count": 0,
                "themes_count": 0,
                "subtopics_count": 0,
            },
        }

    # Prepare canvas centers for first level: books with at least one theme.
    width = 2600.0
    height = 1500.0
    book_centers = _circle_positions(len(themed_global_ids), width / 2, height / 2, min(width, height) * 0.34)

    for index, global_id in enumerate(themed_global_ids):
        user_book = book_by_global[global_id]
        book_themes = themes_by_book.get(global_id, [])
        subtopics_count = sum(len(subtopics_by_theme.get(item.id, [])) for item in book_themes)

        bx, by = book_centers[index] if index < len(book_centers) else (width / 2, height / 2)
        book_node = {
            "id": f"book-{global_id}",
            "type": "book",
            "label": user_book.title or user_book.original_filename,
            "book_id": user_book.id,
            "global_book_id": global_id,
            "summary": (user_book.global_cache.full_summary or "")[:1600],
            "themes_count": len(book_themes),
            "subtopics_count": subtopics_count,
            "x": round(float(bx), 2),
            "y": round(float(by), 2),
            "size": 28,
        }
        nodes.append(book_node)
        book_nodes[global_id] = book_node

        if not book_themes:
            continue

        theme_positions = _circle_positions(
            len(book_themes),
            bx,
            by,
            max(120.0, min(220.0, 80.0 + len(book_themes) * 14.0)),
            start_angle=-math.pi / 2,
        )

        for theme_index, theme in enumerate(book_themes):
            tx, ty = theme_positions[theme_index]
            theme_node = {
                "id": f"theme-{theme.id}",
                "type": "theme",
                "parent_id": book_node["id"],
                "label": theme.title,
                "book_id": user_book.id,
                "global_book_id": global_id,
                "theme_id": theme.id,
                "chapter_title": theme.chapter_title,
                "summary": theme.summary,
                "start_paragraph": theme.start_paragraph,
                "end_paragraph": theme.end_paragraph,
                "start_block_number": theme.start_block_number,
                "end_block_number": theme.end_block_number,
                "subtopics_count": len(subtopics_by_theme.get(theme.id, [])),
                "x": round(float(tx), 2),
                "y": round(float(ty), 2),
                "size": 18,
            }
            nodes.append(theme_node)
            theme_nodes.append(theme_node)
            edges.append(
                {
                    "source": book_node["id"],
                    "target": theme_node["id"],
                    "weight": 1.0,
                    "type": "contains",
                }
            )

            subtopics = subtopics_by_theme.get(theme.id, [])
            theme_texts[theme_node["id"]] = _theme_text_for_embedding(theme, subtopics)
            if not subtopics:
                continue

            sub_positions = _circle_positions(
                len(subtopics),
                tx,
                ty,
                max(74.0, min(145.0, 62.0 + len(subtopics) * 10.0)),
                start_angle=math.pi / 2,
            )

            for sub_index, subtopic in enumerate(subtopics):
                sx, sy = sub_positions[sub_index]
                sub_node = {
                    "id": f"subtopic-{subtopic.id}",
                    "type": "subtopic",
                    "parent_id": theme_node["id"],
                    "label": subtopic.name,
                    "theme_id": theme.id,
                    "book_id": user_book.id,
                    "global_book_id": global_id,
                    "summary": subtopic.summary,
                    "source_quote": subtopic.source_quote,
                    "importance_score": float(subtopic.importance_score),
                    "start_paragraph": subtopic.start_paragraph,
                    "end_paragraph": subtopic.end_paragraph,
                    "x": round(float(sx), 2),
                    "y": round(float(sy), 2),
                    "size": round(10.0 + max(0.0, min(1.0, float(subtopic.importance_score))) * 7.0, 2),
                }
                nodes.append(sub_node)
                edges.append(
                    {
                        "source": theme_node["id"],
                        "target": sub_node["id"],
                        "weight": round(float(subtopic.importance_score), 3),
                        "type": "contains",
                    }
                )

    edges.extend(_build_related_theme_edges(theme_nodes, theme_texts))

    books_payload = [
        {
            "id": node["id"],
            "book_id": node["book_id"],
            "global_book_id": node["global_book_id"],
            "title": node["label"],
            "themes_count": node["themes_count"],
            "subtopics_count": node["subtopics_count"],
        }
        for node in nodes
        if node["type"] == "book"
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "books": books_payload,
        "meta": {
            "books_count": len([n for n in nodes if n["type"] == "book"]),
            "themes_count": len([n for n in nodes if n["type"] == "theme"]),
            "subtopics_count": len([n for n in nodes if n["type"] == "subtopic"]),
        },
    }
