from __future__ import annotations

from typing import Any

from apps.books.services.rag_service import cosine_similarity, create_embedding
from apps.books.services.semantic_block_builder import SemanticLogicalBlock


def _concept_overlap(left: list[str], right: list[str]) -> float:
    left_set = {item.strip().lower() for item in left if item.strip()}
    right_set = {item.strip().lower() for item in right if item.strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def build_semantic_map(book_title: str, blocks: list[SemanticLogicalBlock]) -> dict[str, Any]:
    if not blocks:
        return {"book_title": book_title, "blocks": [], "links": []}

    map_blocks: list[dict[str, Any]] = []
    for block in blocks:
        map_blocks.append(
            {
                "order": block.order_number,
                "title": block.title,
                "main_meaning": block.main_meaning,
                "source_range": f"p{block.start_paragraph}-{block.end_paragraph}",
                "concepts": block.concept_candidates,
                "children": [
                    {
                        "thought": item.get("thought", ""),
                        "quote": item.get("quote", ""),
                        "source_sentence_ids": item.get("source_sentence_ids", []),
                    }
                    for item in block.atomic_thoughts
                ],
            }
        )

    embeddings = {
        block.order_number: create_embedding(
            f"{block.main_meaning}. {' '.join(block.concept_candidates)}"[:2500]
        )
        for block in blocks
    }

    links: list[dict[str, Any]] = []
    for index, left in enumerate(blocks):
        for right in blocks[index + 1 :]:
            overlap = _concept_overlap(left.concept_candidates, right.concept_candidates)
            similarity = cosine_similarity(embeddings[left.order_number], embeddings[right.order_number])

            if overlap < 0.16 and similarity < 0.8:
                continue

            if overlap >= 0.22 and similarity >= 0.76:
                reason = "shared concepts + similar meaning"
            elif overlap >= 0.22:
                reason = "shared concept"
            else:
                reason = "similar meaning"

            links.append(
                {
                    "from_block": left.order_number,
                    "to_block": right.order_number,
                    "reason": reason,
                    "similarity": round(float(max(overlap, similarity)), 4),
                }
            )

    links.sort(key=lambda item: item["similarity"], reverse=True)

    return {
        "book_title": book_title,
        "blocks": map_blocks,
        "links": links[:180],
    }
