from __future__ import annotations

import math
import re
from typing import Any

from apps.books.models import BookTheme, ConceptMention, LogicalBlock, ThemeSubtopic, UserBook
from apps.books.services.concept_normalizer import normalize_concept_name
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


def _theme_compare_text(theme: BookTheme, subtopics: list[ThemeSubtopic], book_title: str) -> str:
    subtopic_names = ", ".join(item.name for item in subtopics[:8])
    return " ".join(
        item
        for item in [
            theme.title,
            theme.chapter_title,
            theme.summary,
            subtopic_names,
            book_title,
        ]
        if item
    )[:4200]


def _avg_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    if size == 0:
        return []
    result = [0.0] * size
    for vector in vectors:
        if len(vector) != size:
            continue
        for index, value in enumerate(vector):
            result[index] += float(value)
    count = max(1, len(vectors))
    return [value / count for value in result]


def _cluster_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    return cosine_similarity(left.get("centroid", []), right.get("centroid", []))


def _agglomerative_cluster(items: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if not items:
        return []
    target_count = max(1, min(target_count, len(items)))
    clusters = [
        {
            "items": [item],
            "centroid": item["embedding"],
        }
        for item in items
    ]
    while len(clusters) > target_count:
        best_pair: tuple[int, int] | None = None
        best_score = -1.0
        for left_idx, left in enumerate(clusters):
            for right_idx in range(left_idx + 1, len(clusters)):
                score = _cluster_similarity(left, clusters[right_idx])
                if score > best_score:
                    best_score = score
                    best_pair = (left_idx, right_idx)
        if best_pair is None:
            break
        left_idx, right_idx = best_pair
        merged_items = clusters[left_idx]["items"] + clusters[right_idx]["items"]
        merged = {
            "items": merged_items,
            "centroid": _avg_vector([item["embedding"] for item in merged_items]),
            "similarity": round(float(best_score), 4),
        }
        clusters[left_idx] = merged
        clusters.pop(right_idx)
    clusters.sort(key=lambda cluster: (-len(cluster["items"]), str(_cluster_label(cluster["items"]))))
    return clusters


DOMAIN_HINTS: list[tuple[str, set[str]]] = [
    ("Физика", {"физика", "скорость", "свет", "движение", "механика", "энергия", "импульс", "молекула", "волна", "квант", "частица", "поле", "сила"}),
    ("Биология", {"биология", "клетка", "фотосинтез", "организм", "молекула", "растение", "животное", "ген", "днк", "белок", "ткань", "экосистема"}),
    ("Математика", {"математика", "число", "уравнение", "функция", "геометрия", "алгебра", "интеграл", "производная", "дробь", "формула"}),
    ("Информатика", {"информатика", "алгоритм", "программа", "данные", "сеть", "протокол", "сервер", "клиент", "код", "компьютер", "база"}),
    ("Химия", {"химия", "реакция", "вещество", "атом", "молекула", "кислота", "основание", "раствор", "элемент"}),
    ("История", {"история", "государство", "война", "революция", "общество", "империя", "правитель", "эпоха", "народ"}),
    ("Литература", {"литература", "герой", "сюжет", "персонаж", "роман", "повесть", "образ", "автор", "конфликт"}),
]


GENERIC_LABEL_WORDS = {
    "глава",
    "раздел",
    "тема",
    "книга",
    "материал",
    "пример",
    "вопрос",
    "часть",
    "основы",
    "понятие",
    "данные",
    "информация",
}


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", value or "") if len(token) >= 3]


def _domain_label(items: list[dict[str, Any]]) -> str:
    text = " ".join(str(item.get("text", "")) for item in items).lower()
    token_set = set(_tokens(text))
    best_label = ""
    best_score = 0
    for label, hints in DOMAIN_HINTS:
        score = len(token_set & hints)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label if best_score >= 2 else ""


def _representative_item(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) <= 1:
        return items[0]
    vectors = [item["embedding"] for item in items]
    centroid = _avg_vector(vectors)
    return max(items, key=lambda item: cosine_similarity(item["embedding"], centroid))


def _cluster_label(items: list[dict[str, Any]], *, allow_domain: bool = True) -> str:
    if not items:
        return "Темы"
    if allow_domain:
        domain = _domain_label(items)
        if domain:
            return domain
    representative = _representative_item(items)
    label = str(representative.get("title", "")).strip()
    if label:
        return label[:90]
    word_counts: dict[str, int] = {}
    for item in items:
        for token in _tokens(str(item.get("title", ""))):
            if token not in GENERIC_LABEL_WORDS:
                word_counts[token] = word_counts.get(token, 0) + 1
    if word_counts:
        top_words = sorted(word_counts.items(), key=lambda row: (-row[1], row[0]))[:3]
        return " / ".join(word for word, _ in top_words).title()
    return "Группа тем"


def _cluster_keywords(items: list[dict[str, Any]], limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        for token in _tokens(str(item.get("title", "")) + " " + str(item.get("chapter_title", ""))):
            if token not in GENERIC_LABEL_WORDS and not token.isdigit():
                counts[token] = counts.get(token, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda row: (-row[1], row[0]))[:limit]]


def _target_cluster_count(count: int, *, top: bool) -> int:
    if count <= 2:
        return count
    if top:
        return max(2, min(8, round(math.sqrt(count / 2.0))))
    return max(1, min(7, round(math.sqrt(count))))


def _build_theme_compare_map(
    *,
    user_books: list[UserBook],
    themes: list[BookTheme],
    subtopics_by_theme: dict[int, list[ThemeSubtopic]],
    book_by_global: dict[int, UserBook],
) -> dict[str, Any]:
    if not themes:
        return {"nodes": [], "edges": [], "meta": {"themes_count": 0, "top_clusters_count": 0, "topic_clusters_count": 0}}

    items: list[dict[str, Any]] = []
    for theme in themes:
        user_book = book_by_global.get(theme.global_book_id)
        if not user_book:
            continue
        subtopics = subtopics_by_theme.get(theme.id, [])
        text = _theme_compare_text(theme, subtopics, user_book.title or user_book.original_filename)
        embedding = create_embedding(text)
        items.append(
            {
                "theme": theme,
                "theme_id": theme.id,
                "title": theme.title,
                "chapter_title": theme.chapter_title,
                "summary": theme.summary,
                "global_book_id": theme.global_book_id,
                "book_id": user_book.id,
                "book_title": user_book.title or user_book.original_filename,
                "text": text,
                "embedding": embedding,
                "subtopics": subtopics,
            }
        )

    if not items:
        return {"nodes": [], "edges": [], "meta": {"themes_count": 0, "top_clusters_count": 0, "topic_clusters_count": 0}}

    top_clusters = _agglomerative_cluster(items, _target_cluster_count(len(items), top=True))
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    topic_cluster_count = 0

    width = 1200.0
    height = 760.0
    top_positions = _circle_positions(len(top_clusters), width / 2, height / 2, 265.0)

    for top_index, top_cluster in enumerate(top_clusters, start=1):
        top_id = f"compare-domain-{top_index}"
        top_items = top_cluster["items"]
        top_x, top_y = top_positions[top_index - 1] if top_index - 1 < len(top_positions) else (width / 2, height / 2)
        top_books = sorted({item["book_title"] for item in top_items})
        top_node = {
            "id": top_id,
            "type": "theme_cluster",
            "cluster_level": "domain",
            "label": _cluster_label(top_items, allow_domain=True),
            "theme_count": len(top_items),
            "books_count": len(top_books),
            "books": top_books[:12],
            "keywords": _cluster_keywords(top_items),
            "summary": "Крупный смысловой блок, собранный из похожих тем разных книг через embeddings.",
            "x": round(float(top_x), 2),
            "y": round(float(top_y), 2),
            "size": 42 + min(28, len(top_items) * 2),
        }
        nodes.append(top_node)

        topic_clusters = _agglomerative_cluster(top_items, _target_cluster_count(len(top_items), top=False))
        topic_cluster_count += len(topic_clusters)
        topic_positions = _circle_positions(
            len(topic_clusters),
            top_x,
            top_y,
            max(145.0, min(245.0, 115.0 + len(topic_clusters) * 22.0)),
            start_angle=-math.pi / 2,
        )

        for topic_index, topic_cluster in enumerate(topic_clusters, start=1):
            topic_id = f"{top_id}-topic-{topic_index}"
            topic_items = topic_cluster["items"]
            topic_x, topic_y = topic_positions[topic_index - 1] if topic_index - 1 < len(topic_positions) else (top_x, top_y)
            topic_books = sorted({item["book_title"] for item in topic_items})
            nodes.append(
                {
                    "id": topic_id,
                    "type": "theme_cluster",
                    "cluster_level": "topic",
                    "parent_id": top_id,
                    "label": _cluster_label(topic_items, allow_domain=False),
                    "theme_count": len(topic_items),
                    "books_count": len(topic_books),
                    "books": topic_books[:12],
                    "keywords": _cluster_keywords(topic_items),
                    "summary": "Подгруппа близких тем внутри крупного смыслового блока.",
                    "x": round(float(topic_x), 2),
                    "y": round(float(topic_y), 2),
                    "size": 27 + min(18, len(topic_items) * 2),
                }
            )
            edges.append({"source": top_id, "target": topic_id, "weight": 1.0, "type": "compare_contains"})

            leaf_positions = _circle_positions(
                len(topic_items),
                topic_x,
                topic_y,
                max(90.0, min(175.0, 74.0 + len(topic_items) * 9.0)),
                start_angle=math.pi / 2,
            )
            for theme_index, item in enumerate(sorted(topic_items, key=lambda row: (row["book_title"], row["theme"].order_number, row["theme_id"]))):
                theme = item["theme"]
                leaf_id = f"compare-theme-{theme.id}"
                leaf_x, leaf_y = leaf_positions[theme_index] if theme_index < len(leaf_positions) else (topic_x, topic_y)
                subtopic_names = [sub.name for sub in item["subtopics"][:8]]
                nodes.append(
                    {
                        "id": leaf_id,
                        "type": "compare_theme",
                        "cluster_level": "theme",
                        "parent_id": topic_id,
                        "domain_id": top_id,
                        "theme_id": theme.id,
                        "book_id": item["book_id"],
                        "global_book_id": item["global_book_id"],
                        "book_title": item["book_title"],
                        "label": theme.title,
                        "chapter_title": theme.chapter_title,
                        "summary": theme.summary,
                        "subtopics": subtopic_names,
                        "start_block_number": theme.start_block_number,
                        "end_block_number": theme.end_block_number,
                        "start_paragraph": theme.start_paragraph,
                        "end_paragraph": theme.end_paragraph,
                        "open_url": f"/library/books/{item['book_id']}/#theme-{theme.id}",
                        "x": round(float(leaf_x), 2),
                        "y": round(float(leaf_y), 2),
                        "size": 17,
                    }
                )
                edges.append({"source": topic_id, "target": leaf_id, "weight": 1.0, "type": "compare_contains"})

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "books_count": len(user_books),
            "themes_count": len(items),
            "top_clusters_count": len(top_clusters),
            "topic_clusters_count": topic_cluster_count,
            "mode": "theme_compare_embeddings",
        },
    }


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


def _block_link(
    user_book: UserBook,
    block: LogicalBlock,
    *,
    theme_id: int | None = None,
    subtopic_id: int | None = None,
) -> dict[str, Any]:
    suffix_parts = ["from=map"]
    if theme_id:
        suffix_parts.append(f"theme={theme_id}")
    if subtopic_id:
        suffix_parts.append(f"subtopic={subtopic_id}")
    suffix = "&".join(suffix_parts)
    return {
        "logical_block_id": block.id,
        "block_id": block.id,
        "block_index": block.order_number,
        "block_title": block.title,
        "chapter_title": block.chapter_title,
        "source_start": block.start_paragraph,
        "source_end": block.end_paragraph,
        "start_paragraph": block.start_paragraph,
        "end_paragraph": block.end_paragraph,
        "open_url": f"/library/books/{user_book.id}/blocks/{block.id}/?{suffix}",
        "summary_url": f"/library/books/{user_book.id}/?block={block.id}#block-{block.id}",
    }


def _score_subtopic_block(subtopic: ThemeSubtopic, block: LogicalBlock) -> int:
    normalized = normalize_concept_name(subtopic.name)
    haystack = " ".join(
        [
            block.title or "",
            block.short_summary or "",
            " ".join(block.concept_candidates or []),
        ]
    ).lower()
    score = 0
    if subtopic.name.lower() in haystack:
        score += 6
    for token in normalized.split():
        if len(token) >= 3 and token in haystack:
            score += 2
    if block.start_paragraph <= subtopic.start_paragraph <= block.end_paragraph:
        score += 1
    return score


def _linked_blocks_for_subtopic(
    *,
    user_book: UserBook,
    subtopic: ThemeSubtopic,
    theme_blocks: list[LogicalBlock],
    mentions_by_normalized: dict[str, list[ConceptMention]],
) -> list[dict[str, Any]]:
    normalized = normalize_concept_name(subtopic.name)
    links: list[dict[str, Any]] = []
    seen: set[int] = set()

    for mention in mentions_by_normalized.get(normalized, []):
        block = mention.logical_block
        if block.id in seen:
            continue
        if block not in theme_blocks and not (
            subtopic.theme.start_block_number <= block.order_number <= subtopic.theme.end_block_number
        ):
            continue
        links.append(_block_link(user_book, block, theme_id=subtopic.theme_id, subtopic_id=subtopic.id))
        seen.add(block.id)

    if links:
        return links[:5]

    scored = [
        (score, block)
        for block in theme_blocks
        if (score := _score_subtopic_block(subtopic, block)) > 0
    ]
    scored.sort(key=lambda item: (-item[0], item[1].order_number))
    for _, block in scored[:3]:
        if block.id not in seen:
            links.append(_block_link(user_book, block, theme_id=subtopic.theme_id, subtopic_id=subtopic.id))
            seen.add(block.id)

    if links:
        return links

    if theme_blocks:
        # Chapter-level subtopics can lack their own block id. In that case the
        # theme's first block is still a stable source anchor.
        first = theme_blocks[0]
        return [_block_link(user_book, first, theme_id=subtopic.theme_id, subtopic_id=subtopic.id)]

    return []


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
            "theme_compare": {"nodes": [], "edges": [], "meta": {"themes_count": 0, "top_clusters_count": 0, "topic_clusters_count": 0}},
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
            "theme_compare": {"nodes": [], "edges": [], "meta": {"themes_count": 0, "top_clusters_count": 0, "topic_clusters_count": 0}},
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

    blocks_by_global_order: dict[int, dict[int, LogicalBlock]] = {}
    for block in LogicalBlock.objects.filter(global_book_id__in=global_ids).order_by("global_book_id", "order_number"):
        blocks_by_global_order.setdefault(block.global_book_id, {})[block.order_number] = block

    mentions_by_global_normalized: dict[int, dict[str, list[ConceptMention]]] = {}
    mention_rows = (
        ConceptMention.objects.filter(global_book_id__in=global_ids)
        .select_related("concept", "logical_block")
        .order_by("global_book_id", "logical_block__order_number", "-importance_score", "id")
    )
    for mention in mention_rows:
        key = mention.concept.normalized_name
        mentions_by_global_normalized.setdefault(mention.global_book_id, {}).setdefault(key, []).append(mention)

    themed_global_ids = [item for item in global_ids if themes_by_book.get(item)]
    if not themed_global_ids:
        return {
            "nodes": [],
            "edges": [],
            "books": [],
            "theme_compare": {"nodes": [], "edges": [], "meta": {"themes_count": 0, "top_clusters_count": 0, "topic_clusters_count": 0}},
            "meta": {
                "books_count": 0,
                "themes_count": 0,
                "subtopics_count": 0,
            },
        }

    # Compact coordinate space so labels/nodes remain readable in SVG viewport.
    width = 1000.0
    height = 700.0
    book_centers = _circle_positions(len(themed_global_ids), width / 2, height / 2, min(width, height) * 0.30)

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
            "status": user_book.status,
            "current_stage": user_book.current_stage,
            "warnings": user_book.error_message,
            "themes_count": len(book_themes),
            "subtopics_count": subtopics_count,
            "x": round(float(bx), 2),
            "y": round(float(by), 2),
            "size": 36,
        }
        nodes.append(book_node)
        book_nodes[global_id] = book_node

        if not book_themes:
            continue

        theme_positions = _circle_positions(
            len(book_themes),
            bx,
            by,
            max(110.0, min(245.0, 90.0 + len(book_themes) * 11.0)),
            start_angle=-math.pi / 2,
        )

        for theme_index, theme in enumerate(book_themes):
            tx, ty = theme_positions[theme_index]
            theme_blocks = [
                block
                for order_number, block in blocks_by_global_order.get(global_id, {}).items()
                if theme.start_block_number <= order_number <= theme.end_block_number
            ]
            theme_block_links = [
                _block_link(user_book, block, theme_id=theme.id)
                for block in theme_blocks[:12]
            ]
            primary_theme_link = theme_block_links[0] if theme_block_links else None
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
                "logical_block_id": primary_theme_link["logical_block_id"] if primary_theme_link else None,
                "block_id": primary_theme_link["block_id"] if primary_theme_link else None,
                "block_index": primary_theme_link["block_index"] if primary_theme_link else None,
                "block_title": primary_theme_link["block_title"] if primary_theme_link else "",
                "open_url": primary_theme_link["summary_url"] if primary_theme_link else f"/library/books/{user_book.id}/#theme-{theme.id}",
                "first_block_url": primary_theme_link["open_url"] if primary_theme_link else "",
                "block_links": theme_block_links,
                "subtopics_count": len(subtopics_by_theme.get(theme.id, [])),
                "x": round(float(tx), 2),
                "y": round(float(ty), 2),
                "size": 24,
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
                max(72.0, min(165.0, 62.0 + len(subtopics) * 8.0)),
                start_angle=math.pi / 2,
            )

            for sub_index, subtopic in enumerate(subtopics):
                sx, sy = sub_positions[sub_index]
                block_links = _linked_blocks_for_subtopic(
                    user_book=user_book,
                    subtopic=subtopic,
                    theme_blocks=theme_blocks,
                    mentions_by_normalized=mentions_by_global_normalized.get(global_id, {}),
                )
                primary_link = block_links[0] if block_links else None
                sub_node = {
                    "id": f"subtopic-{subtopic.id}",
                    "type": "subtopic",
                    "parent_id": theme_node["id"],
                    "subtopic_id": subtopic.id,
                    "label": subtopic.name,
                    "theme_id": theme.id,
                    "book_id": user_book.id,
                    "global_book_id": global_id,
                    "summary": subtopic.summary,
                    "source_quote": subtopic.source_quote,
                    "importance_score": float(subtopic.importance_score),
                    "logical_block_id": primary_link["logical_block_id"] if primary_link else None,
                    "block_id": primary_link["block_id"] if primary_link else None,
                    "block_index": primary_link["block_index"] if primary_link else None,
                    "block_title": primary_link["block_title"] if primary_link else "",
                    "open_url": primary_link["open_url"] if primary_link else "",
                    "summary_url": primary_link["summary_url"] if primary_link else "",
                    "block_links": block_links,
                    "linked_blocks_count": len(block_links),
                    "source_start": primary_link["source_start"] if primary_link else subtopic.start_paragraph,
                    "source_end": primary_link["source_end"] if primary_link else subtopic.end_paragraph,
                    "start_paragraph": subtopic.start_paragraph,
                    "end_paragraph": subtopic.end_paragraph,
                    "x": round(float(sx), 2),
                    "y": round(float(sy), 2),
                    "size": round(13.0 + max(0.0, min(1.0, float(subtopic.importance_score))) * 8.0, 2),
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
    theme_compare = _build_theme_compare_map(
        user_books=user_books,
        themes=themes,
        subtopics_by_theme=subtopics_by_theme,
        book_by_global=book_by_global,
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "books": books_payload,
        "theme_compare": theme_compare,
        "meta": {
            "books_count": len([n for n in nodes if n["type"] == "book"]),
            "themes_count": len([n for n in nodes if n["type"] == "theme"]),
            "subtopics_count": len([n for n in nodes if n["type"] == "subtopic"]),
        },
    }
