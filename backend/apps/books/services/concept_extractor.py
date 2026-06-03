from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymorphy3
from razdel import sentenize

from apps.books.services.concept_normalizer import normalize_concept_name

morph = pymorphy3.MorphAnalyzer()
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
ABBR_RE = re.compile(r"\b[A-ZА-ЯЁ]{2,8}\b")

GENERIC_STOPLIST = {
    "такой образ",
    "данный случай",
    "следующий раздел",
    "эта глава",
    "этот пример",
    "другой способ",
    "таким образом",
    "в настоящее время",
    "с одной стороны",
    "с другой стороны",
    "большое количество",
    "основная проблема",
    "важный вопрос",
    "данный пример",
    "этот раздел",
}

CONCEPT_TYPE_HINTS = {
    "method": {"метод", "алгоритм", "подход", "способ", "model", "method"},
    "definition": {"определение", "термин", "понятие", "definition", "concept"},
    "entity": {"система", "протокол", "модель", "архитектура", "device", "network"},
    "character": {"герой", "персонаж", "character"},
    "event": {"событие", "конфликт", "эпизод", "event"},
}


@dataclass(slots=True)
class ConceptCandidate:
    title: str
    normalized_title: str
    short_explanation: str
    source_quote: str
    source_block_id: int
    source_block_order: int
    confidence: float
    concept_type: str


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _lemma_tokens(text: str) -> list[str]:
    lemmas: list[str] = []
    for raw in WORD_RE.findall(text or ""):
        token = raw.lower()
        if len(token) < 3 or token.isdigit():
            continue
        parsed = morph.parse(token)[0]
        lemma = parsed.normal_form
        if len(lemma) < 3:
            continue
        lemmas.append(lemma)
    return lemmas


def _extract_noun_phrases(text: str, max_len: int = 3) -> Counter[str]:
    words = [word.lower() for word in WORD_RE.findall(text or "") if len(word) > 2]
    counter: Counter[str] = Counter()
    if not words:
        return counter

    parsed = [morph.parse(word)[0] for word in words]
    for idx in range(len(parsed)):
        for span in range(1, max_len + 1):
            chunk = parsed[idx : idx + span]
            if len(chunk) < span:
                continue

            if "NOUN" not in chunk[-1].tag:
                continue
            if span > 1 and not all(("NOUN" in item.tag) or ("ADJF" in item.tag) or ("PRTF" in item.tag) for item in chunk):
                continue

            normalized = " ".join(item.normal_form for item in chunk)
            if len(normalized) < 4:
                continue
            if normalized in GENERIC_STOPLIST:
                continue
            counter[normalized] += 1

    return counter


def _detect_concept_type(name: str, context: str) -> str:
    low = f"{name} {context}".lower()
    for concept_type, hints in CONCEPT_TYPE_HINTS.items():
        if any(hint in low for hint in hints):
            return concept_type
    if ABBR_RE.fullmatch(name.strip()):
        return "abbreviation"
    return "topic"


def _find_quote_for_candidate(candidate: str, text: str) -> str:
    for sent in sentenize(text):
        sentence = _normalize_text(sent.text)
        if not sentence:
            continue
        if candidate.lower() in sentence.lower():
            return sentence[:320]
    # fallback
    return _normalize_text(text)[:320]


def _short_explanation(name: str, quote: str) -> str:
    quote = _normalize_text(quote)
    if not quote:
        return f"Концепт «{name}» раскрывается в данном фрагменте."[:260]
    if len(quote) > 220:
        quote = quote[:220].rstrip() + "..."
    return f"Концепт «{name}» раскрывается в фрагменте: {quote}"[:420]


def extract_concept_candidates_from_text(
    text: str,
    *,
    block_id: int,
    block_order: int,
    max_items: int = 8,
) -> list[ConceptCandidate]:
    clean = _normalize_text(text)
    if not clean:
        return []

    phrase_counter = _extract_noun_phrases(clean, max_len=3)
    abbreviation_counter = Counter(ABBR_RE.findall(clean))

    candidates: list[tuple[str, float]] = []
    for phrase, freq in phrase_counter.most_common(max_items * 4):
        if phrase in GENERIC_STOPLIST:
            continue
        score = float(freq)
        candidates.append((phrase, score))

    for abbr, freq in abbreviation_counter.most_common(max_items * 2):
        if len(abbr) < 2:
            continue
        candidates.append((abbr, float(freq) + 1.2))

    # Deduplicate by normalized title.
    normalized_best: dict[str, tuple[str, float]] = {}
    for original, score in candidates:
        normalized = normalize_concept_name(original) or original.lower()
        if normalized in GENERIC_STOPLIST or len(normalized) < 3:
            continue
        best = normalized_best.get(normalized)
        if best is None or score > best[1]:
            normalized_best[normalized] = (original, score)

    if not normalized_best:
        return []

    max_score = max(score for _, score in normalized_best.values())
    result: list[ConceptCandidate] = []

    for normalized, (original, score) in sorted(normalized_best.items(), key=lambda item: item[1][1], reverse=True):
        if normalized in GENERIC_STOPLIST:
            continue
        quote = _find_quote_for_candidate(original, clean)
        confidence = 0.35 + 0.65 * (score / max_score if max_score else 0.0)
        concept_type = _detect_concept_type(original, quote)
        result.append(
            ConceptCandidate(
                title=original[:255],
                normalized_title=normalized[:255],
                short_explanation=_short_explanation(original, quote),
                source_quote=quote,
                source_block_id=block_id,
                source_block_order=block_order,
                confidence=max(0.0, min(1.0, round(confidence, 4))),
                concept_type=concept_type,
            )
        )
        if len(result) >= max_items:
            break

    return result


def build_theme_subtopics_from_blocks(
    theme_title: str,
    blocks: list[Any],
    *,
    max_items: int = 6,
) -> list[dict[str, Any]]:
    """
    Create subtopics for a theme from block clean text.

    Expected block fields:
    - id
    - order_number
    - short_summary
    - semantic_data.clean_text_for_analysis (optional)
    - source_text
    - start_paragraph/end_paragraph
    """

    candidates: list[ConceptCandidate] = []
    for block in blocks:
        semantic_data = getattr(block, "semantic_data", None) or {}
        clean_text = semantic_data.get("clean_text_for_analysis") if isinstance(semantic_data, dict) else None
        if not clean_text:
            clean_text = getattr(block, "source_text", "")

        block_candidates = extract_concept_candidates_from_text(
            clean_text,
            block_id=getattr(block, "id", 0),
            block_order=getattr(block, "order_number", 0),
            max_items=5,
        )
        candidates.extend(block_candidates)

    if not candidates:
        return []

    deduped: dict[str, ConceptCandidate] = {}
    for item in candidates:
        if item.normalized_title in GENERIC_STOPLIST:
            continue
        existing = deduped.get(item.normalized_title)
        if existing is None or item.confidence > existing.confidence:
            deduped[item.normalized_title] = item

    result = sorted(deduped.values(), key=lambda item: item.confidence, reverse=True)[:max_items]

    subtopics: list[dict[str, Any]] = []
    for item in result:
        subtopics.append(
            {
                "name": item.title,
                "summary": item.short_explanation,
                "source_quote": item.source_quote,
                "importance_score": item.confidence,
                "source_block_order": item.source_block_order,
                "concept_type": item.concept_type,
            }
        )

    return subtopics
