from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from django.core.cache import cache

from apps.books.services.llm_service import extract_atomic_thoughts
from apps.books.services.sentence_segmenter import SourceSentence
from apps.books.services.sentence_window_builder import SentenceWindow

logger = logging.getLogger(__name__)
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")


@dataclass(slots=True)
class AtomicThought:
    id: str
    window_id: str
    chapter_title: str
    text: str
    source_sentence_ids: list[str]
    concept_candidates: list[str]
    confidence: float
    quote: str


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _cache_key_for_window(window_text: str) -> str:
    digest = hashlib.sha256(_normalize_text(window_text).lower().encode("utf-8")).hexdigest()
    return f"atomic_thoughts:v1:{digest}"


def _informative_score(text: str) -> float:
    words = WORD_RE.findall(text or "")
    if not words:
        return 0.0
    long_words = sum(1 for word in words if len(word) >= 6)
    unique_ratio = len({word.lower() for word in words}) / len(words)
    return long_words * 0.45 + unique_ratio * 3.2 + min(50, len(words)) * 0.06


def _fallback_thoughts(window: SentenceWindow, sentence_map: dict[str, SourceSentence]) -> list[dict[str, Any]]:
    source_sentences = [sentence_map[item] for item in window.sentence_ids if item in sentence_map]
    if not source_sentences:
        return []

    ranked = sorted(source_sentences, key=lambda item: _informative_score(item.text), reverse=True)
    picked = ranked[:2] if len(ranked) > 1 else ranked[:1]

    thoughts: list[dict[str, Any]] = []
    for sentence in picked:
        text = _normalize_text(sentence.text)
        if len(text) < 20:
            continue
        words = [word.lower() for word in WORD_RE.findall(text) if len(word) > 3 and not word.isdigit()]
        concepts = list(dict.fromkeys(words[:4]))
        thoughts.append(
            {
                "text": text,
                "source_sentence_ids": [sentence.id],
                "concept_candidates": concepts,
                "confidence": 0.35,
                "quote": text[:280],
            }
        )
    return thoughts


def extract_atomic_thoughts_from_windows(
    windows: list[SentenceWindow],
    sentences: list[SourceSentence],
    *,
    cache_timeout_seconds: int = 60 * 60 * 24 * 14,
) -> tuple[list[AtomicThought], dict[str, Any]]:
    """Extract atomic thoughts in window batches with resilient fallback."""

    sentence_map = {sentence.id: sentence for sentence in sentences}
    thoughts: list[AtomicThought] = []
    llm_calls = 0
    cache_hits = 0
    fallback_calls = 0

    for window_index, window in enumerate(windows, start=1):
        cache_key = _cache_key_for_window(window.text)
        cached = cache.get(cache_key)
        if isinstance(cached, list):
            raw_items = cached
            cache_hits += 1
        else:
            metadata = [
                {"id": sentence_id, "text": sentence_map[sentence_id].text}
                for sentence_id in window.sentence_ids
                if sentence_id in sentence_map
            ]
            raw_items = extract_atomic_thoughts(window.text, metadata)
            llm_calls += 1
            if not isinstance(raw_items, list):
                raw_items = []
            if not raw_items:
                raw_items = _fallback_thoughts(window, sentence_map)
                fallback_calls += 1
            cache.set(cache_key, raw_items, timeout=cache_timeout_seconds)

        if not raw_items:
            raw_items = _fallback_thoughts(window, sentence_map)
            if raw_items:
                fallback_calls += 1

        for local_index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                continue
            text = _normalize_text(str(raw_item.get("text", "")))
            if not text:
                continue

            source_sentence_ids = [
                sentence_id
                for sentence_id in raw_item.get("source_sentence_ids", [])
                if sentence_id in sentence_map
            ]
            if not source_sentence_ids:
                source_sentence_ids = [item for item in window.sentence_ids if item in sentence_map][:2]
            if not source_sentence_ids:
                continue

            concept_candidates = [
                _normalize_text(str(value)).lower()
                for value in raw_item.get("concept_candidates", [])
                if _normalize_text(str(value))
            ]
            concept_candidates = list(dict.fromkeys(concept_candidates))[:8]

            try:
                confidence = float(raw_item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            quote = _normalize_text(str(raw_item.get("quote", "")))
            if not quote:
                quote = " ".join(sentence_map[item].text for item in source_sentence_ids)[:280]

            thoughts.append(
                AtomicThought(
                    id=f"t{window_index}_{local_index}",
                    window_id=window.id,
                    chapter_title=window.chapter_title,
                    text=text,
                    source_sentence_ids=source_sentence_ids,
                    concept_candidates=concept_candidates,
                    confidence=confidence,
                    quote=quote[:380],
                )
            )

    return thoughts, {
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
        "fallback_calls": fallback_calls,
        "thoughts_count": len(thoughts),
    }
