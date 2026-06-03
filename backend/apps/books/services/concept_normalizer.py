from __future__ import annotations

import re
from difflib import SequenceMatcher

import pymorphy3

from apps.books.models import Concept

morph = pymorphy3.MorphAnalyzer()
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")

BAD_SINGLE_WORDS = {
    "текст",
    "книга",
    "глава",
    "раздел",
    "пример",
    "задача",
    "тема",
    "автор",
}
BAD_PHRASES = {
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
}
SKIP_PARTS = {"PREP", "CONJ", "PRCL", "INTJ"}


def normalize_concept_name(name: str) -> str:
    words = [word.lower() for word in WORD_RE.findall(name)]
    normalized_words = []
    for word in words:
        if word.isdigit():
            continue
        parsed = morph.parse(word)[0]
        if any(part in parsed.tag for part in SKIP_PARTS):
            continue
        normalized_words.append(parsed.normal_form)
    return " ".join(normalized_words).strip()


def is_bad_concept(name: str) -> bool:
    cleaned = normalize_concept_name(name)
    if not cleaned:
        return True
    words = cleaned.split()
    if len(cleaned) < 2:
        return True
    if len(words) > 7:
        return True
    if len(words) == 1 and words[0] in BAD_SINGLE_WORDS:
        return True
    if cleaned in BAD_PHRASES:
        return True
    if any(cleaned.startswith(prefix) for prefix in ("этот ", "данный ", "такой ")):
        return True
    if cleaned.isdigit():
        return True
    return False


def find_existing_similar_concept(normalized_name: str) -> Concept | None:
    exact = Concept.objects.filter(normalized_name=normalized_name).first()
    if exact:
        return exact

    # Fast lexical fallback before embedding check.
    candidates = Concept.objects.all().only("id", "normalized_name", "name")
    best_match = None
    best_score = 0.0
    for concept in candidates:
        score = SequenceMatcher(None, normalized_name, concept.normalized_name).ratio()
        if score > best_score:
            best_score = score
            best_match = concept
    if best_match and best_score >= 0.9:
        return best_match

    # Embedding-based similarity fallback.
    try:
        from apps.books.services.rag_service import cosine_similarity, create_embedding
    except Exception:
        return None

    query_embedding = create_embedding(normalized_name)
    best_emb_score = 0.0
    best_emb_match = None
    for concept in Concept.objects.all().only("id", "normalized_name", "name", "description"):
        concept_embedding = create_embedding(f"{concept.name}. {concept.description}")
        score = cosine_similarity(query_embedding, concept_embedding)
        if score > best_emb_score:
            best_emb_score = score
            best_emb_match = concept
    if best_emb_match and best_emb_score >= 0.85:
        return best_emb_match
    return None
