from __future__ import annotations

from typing import Any

from apps.books.services.llm_service import extract_sentence_thought


def extract_thought_from_sentence(sentence_text: str, *, model_name: str | None = None) -> dict[str, Any]:
    return extract_sentence_thought(sentence_text, model_name=model_name)
