from __future__ import annotations

from typing import Any

from apps.books.services.llm_service import compare_thought_pair, compare_thought_with_current_block


def compare_with_current_block(
    *,
    current_block_main_idea: str,
    current_block_thoughts: list[str],
    new_thought: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    return compare_thought_with_current_block(
        current_block_main_idea=current_block_main_idea,
        current_block_thoughts=current_block_thoughts,
        new_thought=new_thought,
        model_name=model_name,
    )


def compare_pair(thought_a: str, thought_b: str, *, model_name: str | None = None) -> dict[str, Any]:
    return compare_thought_pair(thought_a, thought_b, model_name=model_name)
