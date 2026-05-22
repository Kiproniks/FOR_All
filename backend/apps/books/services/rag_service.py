from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from functools import lru_cache
from typing import Any

from apps.books.models import Concept, ConceptMention, LogicalBlock, UserBook

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
EMBED_DIM = 128

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


def _vector_store() -> str:
    return os.getenv("VECTOR_STORE", "chroma").lower()


def _use_chroma() -> bool:
    return _vector_store() == "chroma" and chromadb is not None


@lru_cache(maxsize=1)
def _sentence_model():
    if os.getenv("EMBEDDING_DISABLE_ST", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    if SentenceTransformer is None:
        return None
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    try:
        return SentenceTransformer(model_name)
    except Exception:
        logger.exception("Failed to load sentence-transformer model")
        return None


@lru_cache(maxsize=1)
def _chroma_collection():
    if not _use_chroma():
        return None
    path = os.getenv("CHROMA_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name="logical_blocks")


def _hash_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBED_DIM
    words = [item.lower() for item in WORD_RE.findall(text)]
    if not words:
        return vector
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        for idx, byte in enumerate(digest[:16]):
            slot = (idx * 11 + byte) % EMBED_DIM
            vector[slot] += (byte / 255.0)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def create_embedding(text: str) -> list[float]:
    model = _sentence_model()
    if model is not None:
        try:
            embedding = model.encode(text, normalize_embeddings=True)
            return [float(value) for value in embedding]
        except Exception:
            logger.exception("SentenceTransformer encode failed, fallback to hash embedding")
    return _hash_embedding(text)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    denominator_a = math.sqrt(sum(a * a for a in vec_a))
    denominator_b = math.sqrt(sum(b * b for b in vec_b))
    if denominator_a == 0.0 or denominator_b == 0.0:
        return 0.0
    return float(numerator / (denominator_a * denominator_b))


def save_logical_block_embedding(block_id: int, text: str, metadata: dict[str, Any]) -> str:
    embedding = create_embedding(text)
    embedding_id = f"block:{block_id}"

    collection = _chroma_collection()
    if collection is not None:
        try:
            collection.upsert(
                ids=[embedding_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        except Exception:
            logger.exception("Failed to upsert block embedding in Chroma")

    LogicalBlock.objects.filter(id=block_id).update(embedding_id=embedding_id)
    return embedding_id


def delete_embeddings(embedding_ids: list[str]) -> None:
    ids = sorted({item for item in embedding_ids if item})
    if not ids:
        return

    collection = _chroma_collection()
    if collection is None:
        return

    try:
        collection.delete(ids=ids)
    except Exception:
        logger.exception("Failed to delete embeddings from Chroma")


def search_similar_blocks(query: str, user_id: int, limit: int = 5) -> list[dict[str, Any]]:
    query_vector = create_embedding(query)
    blocks = (
        LogicalBlock.objects.filter(global_book__user_books__user_id=user_id)
        .select_related("global_book")
        .distinct()
    )
    scored = []
    for block in blocks:
        candidate_vector = create_embedding(block.short_summary or block.source_text[:3000])
        scored.append(
            {
                "block": block,
                "score": cosine_similarity(query_vector, candidate_vector),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def search_similar_concepts(concept_name: str, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    query_vector = create_embedding(concept_name)
    concepts = (
        Concept.objects.filter(mentions__global_book__user_books__user_id=user_id)
        .distinct()
    )
    scored = []
    for concept in concepts:
        text = f"{concept.name}. {concept.description}"
        scored.append(
            {
                "concept": concept,
                "score": cosine_similarity(query_vector, create_embedding(text)),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def get_context_for_concept(concept_id: int) -> list[ConceptMention]:
    return list(
        ConceptMention.objects.filter(concept_id=concept_id)
        .select_related("logical_block", "global_book", "concept")
        .order_by("-importance_score")
    )
