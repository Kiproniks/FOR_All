from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymorphy3
from razdel import sentenize

morph = pymorphy3.MorphAnalyzer()

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
CLEAN_SPACES_RE = re.compile(r"\s+")

STOP_WORDS = {
    "это",
    "и",
    "в",
    "на",
    "к",
    "с",
    "по",
    "как",
    "под",
    "для",
    "или",
    "из",
    "о",
    "об",
    "а",
    "но",
    "то",
    "что",
}

MARKER_PATTERNS = [
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s*[—-]\s*это\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+это\s+так(?:ой|ая|ое)\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+это\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+называется\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+называют\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+представляет собой\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+представляют собой\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^под\s+(?P<term>[^.!?]{1,120}?)\s+понимается\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+понимается как\s+(?P<def>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<term>[^.!?]{1,120}?)\s+является\s+(?P<def>.+)$", re.IGNORECASE),
]


@dataclass
class TermCandidate:
    term: str
    normalized_term: str
    definition: str
    source_chapter: str
    source_paragraph_index: int
    source_quote: str
    frequency: int


def sanitize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return CLEAN_SPACES_RE.sub(" ", text).strip()


def split_to_sentences(text: str) -> list[str]:
    sanitized = sanitize_text(text)
    return [fragment.text.strip() for fragment in sentenize(sanitized) if fragment.text.strip()]


def normalize_term(term: str) -> str:
    words = WORD_RE.findall(term.lower())
    normalized_words = []
    for word in words:
        if word.isdigit():
            normalized_words.append(word)
            continue
        parsed = morph.parse(word)[0]
        normalized_words.append(parsed.normal_form)
    return " ".join(normalized_words).strip()


def is_noun_phrase(candidate: str) -> bool:
    words = [w for w in WORD_RE.findall(candidate.lower()) if w]
    if not words or len(words) > 4:
        return False
    parsed_words = [morph.parse(word)[0] for word in words if not word.isdigit()]
    if not parsed_words:
        return False
    return any("NOUN" in p.tag for p in parsed_words)


def clean_term(raw_term: str) -> str:
    term = sanitize_text(raw_term)
    term = term.strip(".,:;!?\"'()[]{}")
    term = term.replace("  ", " ")
    words = WORD_RE.findall(term)
    words = words[:4]
    return " ".join(words).strip()


def valid_term(term: str) -> bool:
    if len(term) < 2:
        return False
    if term.isdigit():
        return False
    low = term.lower()
    if low in STOP_WORDS:
        return False
    words = [w.lower() for w in WORD_RE.findall(term)]
    if not words:
        return False
    if any(word in STOP_WORDS and len(words) == 1 for word in words):
        return False
    return is_noun_phrase(term)


def collect_frequency(chapters: list[dict[str, Any]]) -> Counter[str]:
    normalized_words = []
    for chapter in chapters:
        for paragraph in chapter.get("paragraphs", []):
            for word in WORD_RE.findall(paragraph.lower()):
                if word.isdigit():
                    continue
                normalized_words.append(morph.parse(word)[0].normal_form)
    return Counter(normalized_words)


def estimate_term_frequency(term_normalized: str, word_frequency: Counter[str]) -> int:
    if not term_normalized:
        return 0
    words = term_normalized.split()
    if len(words) == 1:
        return word_frequency.get(words[0], 1)
    return min((word_frequency.get(word, 1) for word in words), default=1)


def extract_terms(book_structure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    word_frequency = collect_frequency(book_structure)
    deduplicated: dict[str, TermCandidate] = {}

    paragraph_index = 0
    for chapter in book_structure:
        chapter_title = chapter.get("chapter_title") or "Без названия главы"
        for paragraph in chapter.get("paragraphs", []):
            paragraph_index += 1
            for sentence in split_to_sentences(paragraph):
                sentence_clean = sanitize_text(sentence)
                for pattern in MARKER_PATTERNS:
                    match = pattern.match(sentence_clean)
                    if not match:
                        continue

                    # Rule-based split: term is extracted around a linguistic marker.
                    term = clean_term(match.group("term"))
                    if not valid_term(term):
                        continue

                    normalized = normalize_term(term)
                    if not normalized:
                        continue
                    definition = sentence_clean
                    frequency = estimate_term_frequency(normalized, word_frequency)
                    candidate = TermCandidate(
                        term=term,
                        normalized_term=normalized,
                        definition=definition,
                        source_chapter=chapter_title,
                        source_paragraph_index=paragraph_index,
                        source_quote=sentence_clean,
                        frequency=max(1, frequency),
                    )

                    existing = deduplicated.get(normalized)
                    # Keep the most informative definition when term repeats.
                    if not existing or len(existing.definition) < len(candidate.definition):
                        deduplicated[normalized] = candidate
                    break

    return [
        {
            "term": candidate.term,
            "normalized_term": candidate.normalized_term,
            "definition": candidate.definition,
            "source_chapter": candidate.source_chapter,
            "source_paragraph_index": candidate.source_paragraph_index,
            "source_quote": candidate.source_quote,
            "frequency": candidate.frequency,
        }
        for candidate in sorted(deduplicated.values(), key=lambda item: item.term.lower())
    ]
