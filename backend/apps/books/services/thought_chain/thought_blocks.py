from __future__ import annotations

import re
from statistics import mean
from typing import Any

from apps.books.models import (
    GlobalLogicalThoughtBlock,
    SentenceThought,
    ThoughtBlockMembership,
    ThoughtRelation,
    UserBook,
)
from apps.books.services.llm_service import (
    check_thought_belongs_to_existing_block,
    compare_thought_with_current_block,
    summarize_thought_block,
)
from apps.books.services.thought_chain.prompts import (
    EXISTING_BLOCK_LIMIT,
    THOUGHT_BLOCK_MEMBERSHIP_THRESHOLD,
    THOUGHT_RELATION_THRESHOLD,
)


def _block_payload(block: GlobalLogicalThoughtBlock) -> dict[str, Any]:
    return {
        "title": block.title,
        "main_idea": block.main_idea,
        "summary": block.summary,
        "keywords": block.keywords or [],
    }


def _relation_related(relation: ThoughtRelation) -> bool:
    return relation.relation in {ThoughtRelation.RELATION_SAME, ThoughtRelation.RELATION_RELATED} and relation.score >= THOUGHT_RELATION_THRESHOLD


def _related_thought_ids(seed: SentenceThought, remaining_ids: set[int]) -> set[int]:
    rows = ThoughtRelation.objects.filter(source_thought=seed, target_thought_id__in=remaining_ids)
    result = {row.target_thought_id for row in rows if _relation_related(row)}
    reverse_rows = ThoughtRelation.objects.filter(target_thought=seed, source_thought_id__in=remaining_ids)
    result.update(row.source_thought_id for row in reverse_rows if _relation_related(row))
    return result


def _normalize_title(value: str) -> str:
    words = [
        word
        for word in re.findall(r"[\wА-Яа-яЁё-]+", (value or "").lower(), flags=re.UNICODE)
        if len(word) >= 3 and not word.isdigit()
    ]
    return " ".join(words)


def _title_similarity(left: str, right: str) -> float:
    left_norm = _normalize_title(left)
    right_norm = _normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _block_thoughts(block: GlobalLogicalThoughtBlock) -> list[str]:
    return [
        membership.thought.thought_text
        for membership in block.memberships.select_related("thought").order_by("thought__index", "id")
        if membership.thought.thought_text
    ]


def _create_block_from_thoughts(
    *,
    book: UserBook,
    thoughts: list[SentenceThought],
    model_name: str | None = None,
) -> tuple[GlobalLogicalThoughtBlock, dict[str, Any]]:
    ordered = sorted(thoughts, key=lambda item: item.index)
    payload = summarize_thought_block([item.thought_text for item in ordered], model_name=model_name)
    block = GlobalLogicalThoughtBlock.objects.create(
        title=str(payload.get("title", "Logical thought block"))[:512],
        main_idea=str(payload.get("main_idea", ""))[:4000],
        summary=str(payload.get("summary", ""))[:8000],
        keywords=payload.get("keywords", []) if isinstance(payload.get("keywords"), list) else [],
    )
    block.source_books.add(book)
    for thought in ordered:
        ThoughtBlockMembership.objects.update_or_create(
            thought=thought,
            block=block,
            defaults={
                "relevance_score": 1.0,
                "reason": "member of the LLM-related thought group used to create this block",
                "llm_raw_response": {"source": "created_from_llm_related_group"},
            },
        )
    return block, payload


def attach_to_existing_global_blocks(
    *,
    book: UserBook,
    thoughts: list[SentenceThought],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Compare new thoughts with existing global logical blocks before creating new ones."""

    blocks = list(
        GlobalLogicalThoughtBlock.objects.exclude(source_books=book)
        .filter(is_merged=False)
        .order_by("updated_at", "id")[:EXISTING_BLOCK_LIMIT]
    )
    comparisons = 0
    memberships = 0
    fallback = 0
    invalid_json = 0
    if not blocks or not thoughts:
        return {"existing_block_comparisons": 0, "existing_block_memberships": 0, "fallback_count": 0, "invalid_json_count": 0}

    for thought in thoughts:
        for block in blocks:
            payload = check_thought_belongs_to_existing_block(
                thought.thought_text,
                _block_payload(block),
                model_name=model_name,
            )
            comparisons += 1
            if payload.get("fallback_used"):
                fallback += 1
            if not payload.get("json_valid"):
                invalid_json += 1
            score = float(payload.get("relevance_score", 0.0) or 0.0)
            if bool(payload.get("belongs")) and score >= THOUGHT_BLOCK_MEMBERSHIP_THRESHOLD:
                ThoughtBlockMembership.objects.update_or_create(
                    thought=thought,
                    block=block,
                    defaults={
                        "relevance_score": score,
                        "reason": str(payload.get("reason", ""))[:1000],
                        "llm_raw_response": payload.get("llm_raw_response", {}),
                    },
                )
                block.source_books.add(book)
                memberships += 1
    return {
        "existing_block_comparisons": comparisons,
        "existing_block_memberships": memberships,
        "fallback_count": fallback,
        "invalid_json_count": invalid_json,
    }


def merge_same_title_blocks(
    *,
    book: UserBook,
    model_name: str | None = None,
    similarity_threshold: float = 0.78,
    block_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Merge duplicate or very similar logical thought blocks.

    Source blocks are not deleted. They are marked as merged and point to the
    surviving block through merged_into. Memberships are moved to the survivor.
    """

    block_qs = GlobalLogicalThoughtBlock.objects.filter(source_books=book, is_merged=False)
    if block_ids is not None:
        block_qs = block_qs.filter(id__in=block_ids)
    blocks = list(block_qs.prefetch_related("memberships__thought").order_by("id"))
    merged_blocks = 0
    memberships_moved = 0
    fallback = 0
    invalid_json = 0

    for primary in blocks:
        primary.refresh_from_db()
        if primary.is_merged:
            continue
        for other in blocks:
            other.refresh_from_db()
            if other.id == primary.id or other.is_merged:
                continue
            title_score = max(
                _title_similarity(primary.title, other.title),
                _title_similarity(primary.main_idea, other.main_idea),
            )
            if title_score < similarity_threshold:
                continue

            moved = 0
            for membership in list(other.memberships.select_related("thought")):
                ThoughtBlockMembership.objects.update_or_create(
                    thought=membership.thought,
                    block=primary,
                    defaults={
                        "relevance_score": max(1.0, float(membership.relevance_score or 0.0)),
                        "reason": f"moved from merged block {other.id}: {membership.reason}",
                        "llm_raw_response": {
                            "source": "merge_same_title_blocks",
                            "merged_from_block_id": other.id,
                            "similarity": round(title_score, 4),
                        },
                    },
                )
                moved += 1
            memberships_moved += moved
            other.memberships.all().delete()
            other.is_merged = True
            other.merged_into = primary
            other.save(update_fields=["is_merged", "merged_into", "updated_at"])
            primary.source_books.add(book)
            merged_blocks += 1

            payload = summarize_thought_block(_block_thoughts(primary), model_name=model_name)
            if payload.get("fallback_used"):
                fallback += 1
            if not payload.get("json_valid"):
                invalid_json += 1
            primary.title = str(payload.get("title", primary.title))[:512]
            primary.main_idea = str(payload.get("main_idea", primary.main_idea))[:4000]
            primary.summary = str(payload.get("summary", primary.summary))[:8000]
            if isinstance(payload.get("keywords"), list):
                primary.keywords = payload["keywords"]
            primary.save(update_fields=["title", "main_idea", "summary", "keywords", "updated_at"])

    return {
        "merged_blocks": merged_blocks,
        "memberships_moved": memberships_moved,
        "fallback_count": fallback,
        "invalid_json_count": invalid_json,
    }


def create_global_blocks_from_relations(
    *,
    book: UserBook,
    thoughts: list[SentenceThought],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build logical thought blocks from strict pairwise LLM-approved relations."""

    remaining_ids = {item.id for item in thoughts}
    by_id = {item.id: item for item in thoughts}
    blocks_created = 0
    memberships_created = 0
    relevance_scores: list[float] = []
    fallback = 0
    invalid_json = 0

    while remaining_ids:
        seed_id = min(remaining_ids, key=lambda item_id: by_id[item_id].index)
        seed = by_id[seed_id]
        group_ids = {seed_id} | _related_thought_ids(seed, remaining_ids - {seed_id})
        ordered_group = sorted((by_id[item_id] for item_id in group_ids), key=lambda item: item.index)
        _, payload = _create_block_from_thoughts(book=book, thoughts=ordered_group, model_name=model_name)
        blocks_created += 1
        memberships_created += len(ordered_group)
        relevance_scores.extend([1.0] * len(ordered_group))
        if payload.get("fallback_used"):
            fallback += 1
        if not payload.get("json_valid"):
            invalid_json += 1
        remaining_ids -= group_ids

    return {
        "global_blocks_created": blocks_created,
        "memberships_created": memberships_created,
        "avg_membership_score": round(mean(relevance_scores), 4) if relevance_scores else 0.0,
        "fallback_count": fallback,
        "invalid_json_count": invalid_json,
    }


def create_global_blocks_greedy(
    *,
    book: UserBook,
    thoughts: list[SentenceThought],
    model_name: str | None = None,
    merge_same_title_blocks_enabled: bool = True,
) -> dict[str, Any]:
    """Fast production mode for building global thought blocks.

    Greedy rules:
    - first unused thought becomes seed;
    - every remaining candidate is compared with the accumulated current block;
    - matching thoughts are added and removed from future seeds;
    - once a thought is inside a block, it cannot start another block;
    - same/similar title blocks are merged after block creation.
    """

    remaining: list[SentenceThought] = sorted(thoughts, key=lambda item: item.index)
    blocks_created = 0
    memberships_created = 0
    relevance_scores: list[float] = []
    comparisons = 0
    fallback = 0
    invalid_json = 0
    created_block_ids: list[int] = []

    while remaining:
        seed = remaining.pop(0)
        current = [seed]
        current_main_idea = seed.thought_text
        kept_remaining: list[SentenceThought] = []

        for candidate in remaining:
            payload = compare_thought_with_current_block(
                current_block_main_idea=current_main_idea,
                current_block_thoughts=[item.thought_text for item in current],
                new_thought=candidate.thought_text,
                model_name=model_name,
            )
            comparisons += 1
            if payload.get("fallback_used"):
                fallback += 1
            if not payload.get("json_valid"):
                invalid_json += 1
            same_block = bool(payload.get("same_block")) and float(payload.get("score", 0.0) or 0.0) >= THOUGHT_RELATION_THRESHOLD
            if same_block:
                current.append(candidate)
                updated = str(payload.get("updated_block_idea", "")).strip()
                current_main_idea = updated or f"{current_main_idea} {candidate.thought_text}"[:1400]
            else:
                kept_remaining.append(candidate)

        remaining = kept_remaining
        block, payload = _create_block_from_thoughts(book=book, thoughts=current, model_name=model_name)
        created_block_ids.append(block.id)
        blocks_created += 1
        memberships_created += len(current)
        relevance_scores.extend([1.0] * len(current))
        if payload.get("fallback_used"):
            fallback += 1
        if not payload.get("json_valid"):
            invalid_json += 1

    merge_stats = {"merged_blocks": 0, "memberships_moved": 0, "fallback_count": 0, "invalid_json_count": 0}
    if merge_same_title_blocks_enabled:
        merge_stats = merge_same_title_blocks(book=book, model_name=model_name, block_ids=created_block_ids)
        fallback += int(merge_stats.get("fallback_count", 0) or 0)
        invalid_json += int(merge_stats.get("invalid_json_count", 0) or 0)

    return {
        "global_blocks_created": blocks_created,
        "memberships_created": memberships_created,
        "avg_membership_score": round(mean(relevance_scores), 4) if relevance_scores else 0.0,
        "greedy_comparisons_done": comparisons,
        "greedy_seed_blocks": blocks_created,
        "merged_blocks": int(merge_stats.get("merged_blocks", 0) or 0),
        "merge_memberships_moved": int(merge_stats.get("memberships_moved", 0) or 0),
        "fallback_count": fallback,
        "invalid_json_count": invalid_json,
    }
