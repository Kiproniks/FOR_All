from __future__ import annotations

import re
from typing import Any

import pymorphy3

from apps.books.services.atomic_thought_extractor import AtomicThought
from apps.books.services.sentence_segmenter import SourceSentence

morph = pymorphy3.MorphAnalyzer()
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
GENERIC_PATTERNS = (
    "в тексте говорится",
    "автор рассказывает",
    "главная мысль",
    "данный фрагмент посвящен",
    "в тексте рассматривается",
    "в этой главе",
)


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _lemma_tokens(text: str) -> set[str]:
    result: set[str] = set()
    for raw in WORD_RE.findall(text or ""):
        word = raw.lower()
        if len(word) < 3 or word.isdigit():
            continue
        lemma = morph.parse(word)[0].normal_form
        if len(lemma) < 3:
            continue
        result.add(lemma)
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / max(1, union)


def _quote_grounded(quote: str, source_text: str) -> bool:
    if not quote:
        return False
    clean_quote = _normalize_text(quote).lower()
    clean_source = _normalize_text(source_text).lower()
    if not clean_quote or not clean_source:
        return False
    if clean_quote in clean_source:
        return True
    # Lightweight fuzzy check by 4-word shingles.
    words = clean_quote.split()
    if len(words) < 4:
        return False
    for index in range(len(words) - 3):
        shingle = " ".join(words[index : index + 4])
        if shingle and shingle in clean_source:
            return True
    return False


def _is_template_thought(text: str) -> bool:
    low = (text or "").lower()
    return any(pattern in low for pattern in GENERIC_PATTERNS)


def clean_and_validate_thoughts(
    thoughts: list[AtomicThought],
    sentences: list[SourceSentence],
) -> tuple[list[AtomicThought], dict[str, Any]]:
    sentence_map = {sentence.id: sentence for sentence in sentences}
    removed_short = 0
    removed_template = 0
    removed_ungrounded = 0
    removed_missing_source = 0

    validated: list[AtomicThought] = []

    for thought in thoughts:
        text = _normalize_text(thought.text)
        if len(text) < 18:
            removed_short += 1
            continue
        if _is_template_thought(text):
            removed_template += 1
            continue

        source_ids = [item for item in thought.source_sentence_ids if item in sentence_map]
        if not source_ids:
            removed_missing_source += 1
            continue

        source_text = " ".join(sentence_map[item].text for item in source_ids)
        thought_tokens = _lemma_tokens(text)
        source_tokens = _lemma_tokens(source_text)
        lexical_overlap = _jaccard(thought_tokens, source_tokens)
        quote_ok = _quote_grounded(thought.quote, source_text)

        confidence = max(0.0, min(1.0, thought.confidence))
        if lexical_overlap < 0.09 and not quote_ok:
            removed_ungrounded += 1
            continue
        if lexical_overlap < 0.16:
            confidence *= 0.72
        if quote_ok:
            confidence = min(1.0, confidence + 0.08)

        validated.append(
            AtomicThought(
                id=thought.id,
                window_id=thought.window_id,
                chapter_title=thought.chapter_title,
                text=text,
                source_sentence_ids=list(dict.fromkeys(source_ids)),
                concept_candidates=list(dict.fromkeys(thought.concept_candidates))[:10],
                confidence=round(confidence, 4),
                quote=_normalize_text(thought.quote)[:380],
            )
        )

    deduped: dict[str, AtomicThought] = {}
    for item in validated:
        key = " ".join(sorted(_lemma_tokens(item.text))) or item.text.lower()
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = item
            continue

        merged_source_ids = list(dict.fromkeys(existing.source_sentence_ids + item.source_sentence_ids))
        merged_candidates = list(dict.fromkeys(existing.concept_candidates + item.concept_candidates))
        picked = existing if existing.confidence >= item.confidence else item
        deduped[key] = AtomicThought(
            id=picked.id,
            window_id=picked.window_id,
            chapter_title=picked.chapter_title,
            text=picked.text,
            source_sentence_ids=merged_source_ids,
            concept_candidates=merged_candidates[:10],
            confidence=max(existing.confidence, item.confidence),
            quote=picked.quote,
        )

    cleaned = list(deduped.values())
    avg_confidence = sum(item.confidence for item in cleaned) / len(cleaned) if cleaned else 0.0
    return cleaned, {
        "input_count": len(thoughts),
        "validated_count": len(cleaned),
        "removed_short": removed_short,
        "removed_template": removed_template,
        "removed_ungrounded": removed_ungrounded,
        "removed_missing_source": removed_missing_source,
        "average_confidence": round(avg_confidence, 4),
    }
