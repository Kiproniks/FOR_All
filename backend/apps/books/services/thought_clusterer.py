from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import pymorphy3
from django.core.cache import cache

from apps.books.services.atomic_thought_extractor import AtomicThought
from apps.books.services.llm_service import merge_thought_cluster
from apps.books.services.rag_service import cosine_similarity, create_embedding
from apps.books.services.sentence_segmenter import SourceSentence

morph = pymorphy3.MorphAnalyzer()
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")


@dataclass(slots=True)
class ThoughtCluster:
    id: str
    title: str
    merged_thought: str
    thought_ids: list[str]
    source_sentence_ids: list[str]
    concept_candidates: list[str]
    confidence: float
    chapter_title: str
    start_sentence_order: int
    end_sentence_order: int


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _embedding_cache_key(text: str) -> str:
    digest = hashlib.sha256(_normalize_text(text).lower().encode("utf-8")).hexdigest()
    return f"thought_embedding:v1:{digest}"


def _merge_cache_key(thoughts: list[str]) -> str:
    joined = "\n".join(sorted(_normalize_text(item).lower() for item in thoughts if item))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"merged_thought:v1:{digest}"


def _get_embedding(text: str) -> list[float]:
    key = _embedding_cache_key(text)
    cached = cache.get(key)
    if isinstance(cached, list) and cached:
        return [float(value) for value in cached]
    embedding = create_embedding(text)
    cache.set(key, embedding, timeout=60 * 60 * 24 * 14)
    return embedding


def _lemma_tokens(text: str) -> set[str]:
    result: set[str] = set()
    for raw in WORD_RE.findall(text or ""):
        token = raw.lower()
        if len(token) < 3 or token.isdigit():
            continue
        result.add(morph.parse(token)[0].normal_form)
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / max(1, union)


def _cluster_title(text: str) -> str:
    words = [word for word in _normalize_text(text).split() if word]
    if not words:
        return "Semantic cluster"
    return " ".join(words[:8])[:120]


def _dedupe_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def cluster_atomic_thoughts(
    thoughts: list[AtomicThought],
    sentences: list[SourceSentence],
) -> tuple[list[ThoughtCluster], dict[str, Any]]:
    if not thoughts:
        return [], {"clusters_count": 0, "merged_items": 0, "llm_merge_calls": 0}

    sentence_order = {sentence.id: index for index, sentence in enumerate(sentences, start=1)}

    def thought_start(item: AtomicThought) -> int:
        return min((sentence_order.get(sentence_id, 10**9) for sentence_id in item.source_sentence_ids), default=10**9)

    sorted_thoughts = sorted(thoughts, key=lambda item: (item.chapter_title, thought_start(item), item.id))

    cluster_states: list[dict[str, Any]] = []
    llm_merge_calls = 0
    merged_items = 0

    for thought in sorted_thoughts:
        tokens = _lemma_tokens(thought.text)
        embedding = _get_embedding(thought.text)
        source_orders = [sentence_order.get(item, 10**9) for item in thought.source_sentence_ids if item in sentence_order]
        start_order = min(source_orders) if source_orders else 10**9
        end_order = max(source_orders) if source_orders else start_order

        best_index = -1
        best_score = -1.0
        best_cosine = 0.0
        best_jaccard = 0.0
        best_concept_overlap = 0.0

        for index, state in enumerate(cluster_states):
            cluster = state["cluster"]
            if cluster.chapter_title != thought.chapter_title:
                continue
            if start_order - state["end_sentence_order"] > 120:
                continue

            jaccard = _jaccard(tokens, state["lemma_tokens"])
            cosine = cosine_similarity(embedding, state["embedding"])
            concept_overlap = _jaccard(set(thought.concept_candidates), set(cluster.concept_candidates))
            score = max(jaccard * 0.55 + cosine * 0.45, cosine * 0.75 + concept_overlap * 0.25)

            if score > best_score:
                best_score = score
                best_index = index
                best_cosine = cosine
                best_jaccard = jaccard
                best_concept_overlap = concept_overlap

        should_merge = False
        use_llm_merge = False
        if best_index >= 0:
            if best_score >= 0.74 or (best_cosine >= 0.83 and best_jaccard >= 0.32):
                should_merge = True
            elif best_concept_overlap >= 0.8 and start_order - cluster_states[best_index]["end_sentence_order"] <= 25:
                should_merge = True
            elif 0.6 <= best_score < 0.74:
                should_merge = True
                use_llm_merge = True

        if should_merge and best_index >= 0:
            state = cluster_states[best_index]
            cluster: ThoughtCluster = state["cluster"]

            merged_text = cluster.merged_thought
            if use_llm_merge:
                merge_key = _merge_cache_key([cluster.merged_thought, thought.text])
                cached_merge = cache.get(merge_key)
                if isinstance(cached_merge, str) and cached_merge.strip():
                    merged_text = cached_merge.strip()
                else:
                    merged_candidate = merge_thought_cluster([cluster.merged_thought, thought.text])
                    llm_merge_calls += 1
                    if merged_candidate:
                        merged_text = merged_candidate
                        cache.set(merge_key, merged_text, timeout=60 * 60 * 24 * 14)
            else:
                if thought.text.lower() not in cluster.merged_thought.lower():
                    merged_text = f"{cluster.merged_thought} {thought.text}"[:1500]

            merged_items += 1
            cluster.source_sentence_ids = _dedupe_ids(cluster.source_sentence_ids + thought.source_sentence_ids)
            cluster.thought_ids = _dedupe_ids(cluster.thought_ids + [thought.id])
            cluster.concept_candidates = _dedupe_ids(cluster.concept_candidates + thought.concept_candidates)[:12]
            cluster.merged_thought = _normalize_text(merged_text)
            cluster.title = _cluster_title(cluster.merged_thought)
            cluster.confidence = round((cluster.confidence + thought.confidence) / 2.0, 4)
            cluster.end_sentence_order = max(cluster.end_sentence_order, end_order)
            cluster.start_sentence_order = min(cluster.start_sentence_order, start_order)

            state["lemma_tokens"] = _lemma_tokens(cluster.merged_thought)
            state["embedding"] = _get_embedding(cluster.merged_thought)
            state["end_sentence_order"] = cluster.end_sentence_order
            continue

        cluster_id = f"c{len(cluster_states) + 1}"
        cluster = ThoughtCluster(
            id=cluster_id,
            title=_cluster_title(thought.text),
            merged_thought=_normalize_text(thought.text),
            thought_ids=[thought.id],
            source_sentence_ids=_dedupe_ids(thought.source_sentence_ids),
            concept_candidates=_dedupe_ids(thought.concept_candidates)[:12],
            confidence=round(max(0.0, min(1.0, thought.confidence)), 4),
            chapter_title=thought.chapter_title,
            start_sentence_order=start_order,
            end_sentence_order=end_order,
        )
        cluster_states.append(
            {
                "cluster": cluster,
                "lemma_tokens": tokens,
                "embedding": embedding,
                "end_sentence_order": end_order,
            }
        )

    clusters = [item["cluster"] for item in cluster_states]
    return clusters, {
        "clusters_count": len(clusters),
        "merged_items": merged_items,
        "llm_merge_calls": llm_merge_calls,
    }
