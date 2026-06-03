from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from apps.books.services.llm_service import (
    build_block_main_meaning,
    name_semantic_block,
    summarize_logical_block,
)
from apps.books.services.rag_service import cosine_similarity, create_embedding
from apps.books.services.sentence_segmenter import SourceSentence
from apps.books.services.thought_clusterer import ThoughtCluster

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
TRANSITION_MARKERS = (
    "итак",
    "таким образом",
    "с другой стороны",
    "однако",
    "далее",
    "в заключение",
)


@dataclass(slots=True)
class SemanticLogicalBlock:
    order_number: int
    chapter_title: str
    title: str
    main_meaning: str
    source_text: str
    source_sentence_ids: list[str]
    thought_cluster_ids: list[str]
    concept_candidates: list[str]
    start_paragraph: int
    end_paragraph: int
    token_count: int
    atomic_thoughts: list[dict[str, Any]]


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _sorted_sentence_ids(ids: list[str], sentence_order: dict[str, int]) -> list[str]:
    return sorted({item for item in ids if item in sentence_order}, key=lambda item: sentence_order[item])


def _build_source_text(sentence_ids: list[str], sentence_map: dict[str, SourceSentence], sentence_order: dict[str, int]) -> str:
    ordered_ids = _sorted_sentence_ids(sentence_ids, sentence_order)
    return " ".join(sentence_map[item].text for item in ordered_ids if item in sentence_map).strip()


def _cluster_word_count(cluster: ThoughtCluster) -> int:
    return _word_count(cluster.merged_thought)


def _cluster_concepts(cluster: ThoughtCluster) -> set[str]:
    return {item.strip().lower() for item in cluster.concept_candidates if item.strip()}


def _should_split(
    current_clusters: list[ThoughtCluster],
    next_cluster: ThoughtCluster,
    *,
    sentence_map: dict[str, SourceSentence],
    sentence_order: dict[str, int],
    min_words: int,
    max_words: int,
) -> bool:
    if not current_clusters:
        return False

    if current_clusters[-1].chapter_title != next_cluster.chapter_title:
        return True

    current_text = " ".join(cluster.merged_thought for cluster in current_clusters)
    current_words = _word_count(current_text)
    next_words = _cluster_word_count(next_cluster)
    if current_words >= min_words and current_words + next_words > max_words:
        return True

    current_concepts = set()
    for cluster in current_clusters:
        current_concepts.update(_cluster_concepts(cluster))
    next_concepts = _cluster_concepts(next_cluster)
    overlap = len(current_concepts & next_concepts) / max(1, len(current_concepts | next_concepts))

    current_embedding = create_embedding(current_text[:2500])
    next_embedding = create_embedding(next_cluster.merged_thought[:1200])
    similarity = cosine_similarity(current_embedding, next_embedding)

    if current_words >= min_words and similarity < 0.58 and overlap < 0.12:
        return True

    next_lower = next_cluster.merged_thought.lower().strip()
    if current_words >= min_words and any(next_lower.startswith(marker) for marker in TRANSITION_MARKERS):
        return True

    # Prevent giant spans even when cluster similarity is high.
    all_sentence_ids = []
    for cluster in current_clusters:
        all_sentence_ids.extend(cluster.source_sentence_ids)
    all_sentence_ids.extend(next_cluster.source_sentence_ids)
    source_text = _build_source_text(all_sentence_ids, sentence_map, sentence_order)
    if _word_count(source_text) > int(max_words * 1.25):
        return True

    return False


def _block_title(chapter_title: str, block_number: int, clusters: list[ThoughtCluster]) -> str:
    candidate = name_semantic_block([cluster.merged_thought for cluster in clusters])
    if candidate:
        return candidate[:512]
    if chapter_title:
        return f"{chapter_title} - block {block_number}"[:512]
    return f"Semantic block {block_number}"[:512]


def _block_main_meaning(clusters: list[ThoughtCluster], source_text: str) -> str:
    summary = build_block_main_meaning(
        [
            {
                "text": cluster.merged_thought,
                "concept_candidates": cluster.concept_candidates,
                "confidence": cluster.confidence,
            }
            for cluster in clusters
        ],
        source_text,
    )
    if not summary or len(summary) < 40:
        summary = summarize_logical_block(source_text)
    return _normalize_text(summary)[:2000]


def _collect_atomic_thoughts(clusters: list[ThoughtCluster], sentence_map: dict[str, SourceSentence]) -> list[dict[str, Any]]:
    atomic: list[dict[str, Any]] = []
    for cluster in clusters:
        quote = ""
        for sentence_id in cluster.source_sentence_ids:
            sentence = sentence_map.get(sentence_id)
            if sentence:
                quote = sentence.text[:280]
                break
        atomic.append(
            {
                "cluster_id": cluster.id,
                "thought": cluster.merged_thought,
                "quote": quote,
                "source_sentence_ids": cluster.source_sentence_ids,
                "concept_candidates": cluster.concept_candidates,
                "confidence": cluster.confidence,
            }
        )
    return atomic


def build_semantic_logical_blocks(
    clusters: list[ThoughtCluster],
    sentences: list[SourceSentence],
    *,
    min_words: int = 260,
    max_words: int = 1300,
) -> list[SemanticLogicalBlock]:
    if not clusters:
        return []

    sentence_map = {sentence.id: sentence for sentence in sentences}
    sentence_order = {sentence.id: index for index, sentence in enumerate(sentences, start=1)}
    ordered_clusters = sorted(clusters, key=lambda item: (item.chapter_title, item.start_sentence_order, item.id))

    grouped: list[list[ThoughtCluster]] = []
    current: list[ThoughtCluster] = []

    for cluster in ordered_clusters:
        if _should_split(
            current,
            cluster,
            sentence_map=sentence_map,
            sentence_order=sentence_order,
            min_words=min_words,
            max_words=max_words,
        ):
            if current:
                grouped.append(current)
            current = [cluster]
        else:
            current.append(cluster)

    if current:
        grouped.append(current)

    blocks: list[SemanticLogicalBlock] = []
    for order_number, group in enumerate(grouped, start=1):
        sentence_ids: list[str] = []
        cluster_ids: list[str] = []
        candidate_counter: Counter[str] = Counter()

        for cluster in group:
            sentence_ids.extend(cluster.source_sentence_ids)
            cluster_ids.append(cluster.id)
            for candidate in cluster.concept_candidates:
                clean = candidate.strip().lower()
                if clean:
                    candidate_counter[clean] += 1

        source_sentence_ids = _sorted_sentence_ids(sentence_ids, sentence_order)
        source_text = _build_source_text(source_sentence_ids, sentence_map, sentence_order)
        if not source_text:
            continue

        paragraph_indexes = [sentence_map[sentence_id].paragraph_index for sentence_id in source_sentence_ids if sentence_id in sentence_map]
        if not paragraph_indexes:
            continue

        chapter_title = group[0].chapter_title
        main_meaning = _block_main_meaning(group, source_text)
        title = _block_title(chapter_title, order_number, group)
        concept_candidates = [name for name, _ in candidate_counter.most_common(12)]
        if not concept_candidates:
            # fallback from cluster titles
            concept_candidates = [
                cluster.title.lower()[:80]
                for cluster in group
                if cluster.title.strip()
            ][:6]

        block = SemanticLogicalBlock(
            order_number=order_number,
            chapter_title=chapter_title,
            title=title,
            main_meaning=main_meaning,
            source_text=source_text,
            source_sentence_ids=source_sentence_ids,
            thought_cluster_ids=cluster_ids,
            concept_candidates=concept_candidates,
            start_paragraph=min(paragraph_indexes),
            end_paragraph=max(paragraph_indexes),
            token_count=_word_count(source_text),
            atomic_thoughts=_collect_atomic_thoughts(group, sentence_map),
        )
        blocks.append(block)

    if not blocks:
        # Final safety net: one block from all clusters.
        all_sentence_ids = _sorted_sentence_ids(
            [sentence_id for cluster in ordered_clusters for sentence_id in cluster.source_sentence_ids],
            sentence_order,
        )
        source_text = _build_source_text(all_sentence_ids, sentence_map, sentence_order)
        if source_text:
            paragraph_indexes = [sentence_map[sentence_id].paragraph_index for sentence_id in all_sentence_ids if sentence_id in sentence_map]
            blocks.append(
                SemanticLogicalBlock(
                    order_number=1,
                    chapter_title=ordered_clusters[0].chapter_title,
                    title=_block_title(ordered_clusters[0].chapter_title, 1, ordered_clusters),
                    main_meaning=_block_main_meaning(ordered_clusters, source_text),
                    source_text=source_text,
                    source_sentence_ids=all_sentence_ids,
                    thought_cluster_ids=[cluster.id for cluster in ordered_clusters],
                    concept_candidates=list(
                        dict.fromkeys(
                            candidate.strip().lower()
                            for cluster in ordered_clusters
                            for candidate in cluster.concept_candidates
                            if candidate.strip()
                        )
                    )[:12],
                    start_paragraph=min(paragraph_indexes) if paragraph_indexes else 0,
                    end_paragraph=max(paragraph_indexes) if paragraph_indexes else 0,
                    token_count=_word_count(source_text),
                    atomic_thoughts=_collect_atomic_thoughts(ordered_clusters, sentence_map),
                )
            )

    return blocks
