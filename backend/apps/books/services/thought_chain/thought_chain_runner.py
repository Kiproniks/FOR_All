from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections import Counter
import os
import re
import time
from typing import Any, Callable

from django.db import transaction
from django.utils import timezone

from apps.books.models import (
    BookSentence,
    GlobalBookCache,
    GlobalLogicalThoughtBlock,
    SentenceThought,
    SequentialThoughtGroup,
    ThoughtBlockMembership,
    ThoughtChainAnalysisRun,
    ThoughtRelation,
    UserBook,
)
from apps.books.services.thought_chain.prompts import (
    DEFAULT_DRY_RUN_MAX_PAIRS,
    THOUGHT_RELATION_THRESHOLD,
    THOUGHT_SAME_BLOCK_THRESHOLD,
)
from apps.books.services.thought_chain.sentence_splitter import ThoughtChainSentence, split_book_into_sentences
from apps.books.services.thought_chain.thought_blocks import (
    attach_to_existing_global_blocks,
    create_global_blocks_greedy,
    create_global_blocks_from_relations,
)
from apps.books.services.thought_chain.thought_extractor import extract_thought_from_sentence
from apps.books.services.thought_chain.thought_relation import compare_pair, compare_with_current_block


@dataclass(slots=True)
class RuntimeThought:
    index: int
    sentence_index: int
    sentence_text: str
    thought_text: str
    normalized_thought: str
    terms: list[str]
    chapter_title: str = ""
    section_title: str = ""
    paragraph_index: int = 0
    is_meaningful: bool = True
    noise: bool = False
    skip_reason: str = ""
    json_valid: bool = True
    fallback_used: bool = False
    terms_removed_count: int = 0
    terms_removed_examples: list[str] | None = None
    quality_flags: list[str] | None = None
    pre_repair_quality_flags: list[str] | None = None
    pre_repair_thought: str = ""
    weird_token_examples: list[str] | None = None
    db_id: int | None = None


def _empty_report(book_title: str, *, model_name: str | None, status: str) -> dict[str, Any]:
    return {
        "book_title": book_title,
        "mode": "llm_thought_chain",
        "analysis_mode": "",
        "block_generation_mode": "",
        "model": model_name or "",
        "total_sentences": 0,
        "meaningful_sentences": 0,
        "noise_sentences": 0,
        "skipped_sentences": 0,
        "thoughts_created": 0,
        "sequential_groups_created": 0,
        "pairwise_comparisons_total": 0,
        "pairwise_comparisons_done": 0,
        "pairwise_llm_calls": 0,
        "pairwise_prefiltered_no_llm": 0,
        "pairwise_slow_calls": 0,
        "skipped_pairs": 0,
        "strict_mode": False,
        "strict_pairwise_llm": False,
        "relations_created": 0,
        "global_blocks_created": 0,
        "memberships_created": 0,
        "avg_membership_score": 0.0,
        "greedy_comparisons_done": 0,
        "greedy_seed_blocks": 0,
        "merged_blocks": 0,
        "merge_memberships_moved": 0,
        "fallback_count": 0,
        "invalid_json_count": 0,
        "timeout_count": 0,
        "started_at": timezone.now().isoformat(),
        "finished_at": "",
        "status": status,
        "sequential_groups_sample": [],
        "sentence_thoughts_sample": [],
        "noise_examples": [],
        "skip_reason_top": {},
        "pairwise_relations_sample": [],
        "relation_score_inconsistencies": 0,
        "relation_score_fixed": 0,
        "relation_score_fix_examples": [],
        "english_explanations_sanitized": 0,
        "english_explanations_detected_total": 0,
        "english_explanations_retried": 0,
        "english_explanations_remaining": 0,
        "semantic_guard_applied_count": 0,
        "same_count": 0,
        "related_count": 0,
        "different_count": 0,
        "relation_rate": 0.0,
        "same_ratio_too_high": False,
        "related_ratio_too_high": False,
        "hub_thought_overlinked": False,
        "hub_thoughts": [],
        "suspicious_same_examples": [],
        "weird_tokens_in_thoughts": 0,
        "mixed_language_tokens": 0,
        "ungrounded_thoughts": 0,
        "bad_thoughts_detected": 0,
        "bad_thoughts_after_repair": 0,
        "english_thoughts_detected": 0,
        "english_thoughts_after_repair": 0,
        "mixed_language_tokens_after_repair": 0,
        "weird_tokens_after_repair": 0,
        "ungrounded_thoughts_after_repair": 0,
        "thought_retries": 0,
        "thought_retry_success": 0,
        "safe_sentence_fallback_used": 0,
        "safe_group_summary_fallback_used": 0,
        "bad_group_summaries_after_repair": 0,
        "relation_explanation_contradictions": 0,
        "relation_explanation_rewritten": 0,
        "quality_gate_passed": False,
        "quality_gate_blockers": [],
        "thought_quality_examples": [],
        "terms_removed_count": 0,
        "terms_removed_examples": [],
        "final_terms_examples": [],
    }


def _pairwise_total(count: int) -> int:
    return max(0, count * (count - 1) // 2)


def _pairwise_prefilter_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.split(r"\s+", value.lower()):
        cleaned = "".join(ch for ch in raw if ch.isalnum() or ch == "-").strip("-")
        if len(cleaned) >= 5 and not cleaned.isdigit():
            tokens.add(cleaned)
    return tokens


def _deterministic_pair_payload(left: RuntimeThought, right: RuntimeThought, *, force: bool = False) -> dict[str, Any] | None:
    """Skip Ollama for pairs that are clearly unrelated before semantic scoring."""

    left_terms = {item.lower() for item in left.terms if len(item) >= 4}
    right_terms = {item.lower() for item in right.terms if len(item) >= 4}
    term_overlap = left_terms & right_terms
    left_tokens = _pairwise_prefilter_tokens(left.normalized_thought or left.thought_text)
    right_tokens = _pairwise_prefilter_tokens(right.normalized_thought or right.thought_text)
    lexical_score = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

    if not force and (term_overlap or lexical_score >= 0.08):
        return None
    if term_overlap or lexical_score >= 0.18:
        relation = "related"
        score = 0.68
        explanation = "Мысли имеют частичное пересечение по ключевым словам или терминам, но не являются одной и той же мыслью."
    else:
        relation = "different"
        score = 0.2
        explanation = "Мысли не имеют достаточного пересечения по ключевым словам и терминам."

    return {
        "relation": relation,
        "score": score,
        "explanation": explanation,
        "json_valid": True,
        "fallback_used": False,
        "quality_flags": ["pairwise_prefilter_no_llm"],
        "relation_score_inconsistent": False,
        "relation_score_fixed": False,
        "score_before_fix": score,
        "english_explanation_detected": False,
        "english_explanation_retried": False,
        "english_explanation_remaining": False,
        "english_explanation_sanitized": False,
        "relation_explanation_consistent": True,
        "relation_explanation_contradiction": False,
        "relation_explanation_rewritten": False,
        "relation_explanation_problem": "",
        "semantic_guard_applied": False,
        "semantic_guard_reason": "pairwise_prefilter_no_llm",
    }


def _runtime_thought_from_payload(sentence: ThoughtChainSentence, payload: dict[str, Any], index: int) -> RuntimeThought:
    return RuntimeThought(
        index=index,
        sentence_index=sentence.index,
        sentence_text=sentence.text,
        thought_text=str(payload.get("thought", "")).strip(),
        normalized_thought=str(payload.get("normalized_thought", "")).strip(),
        terms=[str(item).strip() for item in payload.get("terms", []) if str(item).strip()] if isinstance(payload.get("terms"), list) else [],
        chapter_title=sentence.chapter_title,
        section_title=sentence.section_title,
        paragraph_index=sentence.paragraph_index,
        is_meaningful=bool(payload.get("is_meaningful", True)) and bool(str(payload.get("thought", "")).strip()) and not bool(payload.get("noise", False)),
        noise=bool(payload.get("noise", False)),
        skip_reason=str(payload.get("skip_reason", "")).strip(),
        json_valid=bool(payload.get("json_valid", False)),
        fallback_used=bool(payload.get("fallback_used", False)),
        terms_removed_count=int(payload.get("terms_removed_count", 0) or 0),
        terms_removed_examples=[
            str(item).strip() for item in payload.get("terms_removed_examples", []) if str(item).strip()
        ]
        if isinstance(payload.get("terms_removed_examples"), list)
        else [],
        quality_flags=[str(item).strip() for item in payload.get("quality_flags", []) if str(item).strip()]
        if isinstance(payload.get("quality_flags"), list)
        else [],
        pre_repair_quality_flags=[
            str(item).strip() for item in payload.get("pre_repair_quality_flags", []) if str(item).strip()
        ]
        if isinstance(payload.get("pre_repair_quality_flags"), list)
        else [],
        pre_repair_thought=str(payload.get("pre_repair_thought", "")).strip(),
        weird_token_examples=[
            str(item).strip() for item in payload.get("weird_token_examples", []) if str(item).strip()
        ]
        if isinstance(payload.get("weird_token_examples"), list)
        else [],
    )


def _update_meaning_stats(report: dict[str, Any], thoughts: list[RuntimeThought]) -> None:
    meaningful = [item for item in thoughts if item.is_meaningful and not item.noise and item.thought_text]
    skipped = [item for item in thoughts if item.noise or not item.is_meaningful or not item.thought_text]
    report["meaningful_sentences"] = len(meaningful)
    report["noise_sentences"] = sum(1 for item in thoughts if item.noise)
    report["skipped_sentences"] = len(skipped)
    report["thoughts_created"] = len(meaningful)
    reasons = Counter(item.skip_reason or "not meaningful" for item in skipped)
    report["skip_reason_top"] = dict(reasons.most_common(10))
    report["noise_examples"] = [
        {
            "sentence_index": item.sentence_index,
            "text": item.sentence_text,
            "reason": item.skip_reason or "not meaningful",
        }
        for item in skipped[:20]
    ]
    report["sentence_thoughts_sample"] = [
        {
            "sentence_index": item.sentence_index,
            "sentence_text": item.sentence_text,
            "thought": item.thought_text,
            "is_meaningful": item.is_meaningful,
            "noise": item.noise,
            "skip_reason": item.skip_reason,
            "terms": item.terms,
            "terms_removed_count": item.terms_removed_count,
            "terms_removed_examples": item.terms_removed_examples or [],
            "quality_flags": item.quality_flags or [],
            "pre_repair_quality_flags": item.pre_repair_quality_flags or [],
            "pre_repair_thought": item.pre_repair_thought,
            "weird_token_examples": item.weird_token_examples or [],
        }
        for item in thoughts[:15]
    ]
    quality_examples = report.setdefault("thought_quality_examples", [])
    bad_flags = {"weird_token", "mixed_language_token", "cjk_token", "english_service_text", "ungrounded_thought"}
    for item in thoughts:
        flags = item.quality_flags or []
        pre_flags = item.pre_repair_quality_flags or []
        if bad_flags & set(pre_flags):
            report["bad_thoughts_detected"] += 1
        if bad_flags & set(flags):
            report["bad_thoughts_after_repair"] += 1
        if "english_service_text" in pre_flags:
            report["english_thoughts_detected"] += 1
        if "english_service_text" in flags:
            report["english_thoughts_after_repair"] += 1
        if "weird_token" in pre_flags or "cjk_token" in pre_flags:
            report["weird_tokens_in_thoughts"] += 1
        if "weird_token" in flags or "cjk_token" in flags:
            report["weird_tokens_after_repair"] += 1
        if "mixed_language_token" in pre_flags:
            report["mixed_language_tokens"] += 1
        if "mixed_language_token" in flags:
            report["mixed_language_tokens_after_repair"] += 1
        if "ungrounded_thought" in pre_flags:
            report["ungrounded_thoughts"] += 1
        if "ungrounded_thought" in flags:
            report["ungrounded_thoughts_after_repair"] += 1
        if "thought_retry_used" in flags:
            report["thought_retries"] += 1
            report["thought_retry_success"] += 1
        if "safe_sentence_fallback_used" in flags:
            report["safe_sentence_fallback_used"] += 1
        tracked = bad_flags | {"safe_sentence_fallback_used", "thought_retry_failed", "thought_retry_used"}
        if (tracked & set(flags) or tracked & set(pre_flags)) and len(quality_examples) < 20:
            quality_examples.append(
                {
                    "sentence_index": item.sentence_index,
                    "thought": item.thought_text,
                    "pre_repair_thought": item.pre_repair_thought,
                    "flags": flags,
                    "pre_repair_flags": pre_flags,
                    "examples": item.weird_token_examples or [],
                    "sentence_text": item.sentence_text,
                }
            )
    report["terms_removed_count"] = sum(item.terms_removed_count for item in thoughts)
    removed_examples: list[str] = []
    final_examples: list[str] = []
    for item in thoughts:
        for term in item.terms_removed_examples or []:
            if term and term not in removed_examples:
                removed_examples.append(term)
        for term in item.terms:
            if term and term not in final_examples:
                final_examples.append(term)
        if len(removed_examples) >= 20 and len(final_examples) >= 20:
            break
    report["terms_removed_examples"] = removed_examples[:20]
    report["final_terms_examples"] = final_examples[:20]


def _update_relation_quality_stats(report: dict[str, Any], payload: dict[str, Any], left_index: int, right_index: int) -> None:
    if payload.get("relation_score_inconsistent"):
        report["relation_score_inconsistencies"] += 1
    if payload.get("relation_score_fixed"):
        report["relation_score_fixed"] += 1
        examples = report.setdefault("relation_score_fix_examples", [])
        if len(examples) < 10:
            examples.append(
                {
                    "source_index": left_index,
                    "target_index": right_index,
                    "relation": payload.get("relation"),
                    "score_before_fix": payload.get("score_before_fix"),
                    "score_after_fix": payload.get("score"),
                }
            )
    if payload.get("english_explanation_sanitized"):
        report["english_explanations_sanitized"] += 1
    if payload.get("english_explanation_detected"):
        report["english_explanations_detected_total"] += 1
    if payload.get("english_explanation_retried"):
        report["english_explanations_retried"] += 1
    if payload.get("english_explanation_remaining"):
        report["english_explanations_remaining"] += 1
    if payload.get("semantic_guard_applied"):
        report["semantic_guard_applied_count"] += 1
    if payload.get("relation_explanation_contradiction"):
        report["relation_explanation_contradictions"] += 1
    if payload.get("relation_explanation_rewritten"):
        report["relation_explanation_rewritten"] += 1


def _finalize_pairwise_distribution(report: dict[str, Any], relations: list[dict[str, Any]]) -> None:
    done = len(relations)
    counts = Counter(str(item.get("relation", "different")) for item in relations)
    same_count = int(counts.get("same", 0))
    related_count = int(counts.get("related", 0))
    different_count = int(counts.get("different", 0))
    report["same_count"] = same_count
    report["related_count"] = related_count
    report["different_count"] = different_count
    report["relation_rate"] = round((same_count + related_count) / max(1, done), 4)
    report["same_ratio_too_high"] = done > 0 and same_count / done > 0.25
    report["related_ratio_too_high"] = done > 0 and related_count / done > 0.60

    same_by_thought: Counter[int] = Counter()
    seen_by_thought: Counter[int] = Counter()
    suspicious: list[dict[str, Any]] = []
    for item in relations:
        left = int(item.get("source_index", 0) or 0)
        right = int(item.get("target_index", 0) or 0)
        relation = str(item.get("relation", "different"))
        if left:
            seen_by_thought[left] += 1
        if right:
            seen_by_thought[right] += 1
        flags = item.get("quality_flags", [])
        if relation == "same":
            if left:
                same_by_thought[left] += 1
            if right:
                same_by_thought[right] += 1
            if len(suspicious) < 10 and (
                any(str(flag).startswith("semantic_guard") for flag in flags)
                or float(item.get("score", 0.0) or 0.0) < 0.92
            ):
                suspicious.append(item)

    hubs: list[dict[str, Any]] = []
    for index, same_links in same_by_thought.items():
        seen = seen_by_thought.get(index, 0)
        if seen >= 5 and same_links / max(1, seen) > 0.30:
            hubs.append({"thought_index": index, "same_links": same_links, "seen_pairs": seen})
    report["hub_thought_overlinked"] = bool(hubs)
    report["hub_thoughts"] = sorted(hubs, key=lambda item: (-item["same_links"], item["thought_index"]))[:10]
    report["suspicious_same_examples"] = suspicious


def _apply_quality_gate(report: dict[str, Any]) -> None:
    checks = {
        "invalid_json_count": report.get("invalid_json_count", 0),
        "timeout_count": report.get("timeout_count", 0),
        "bad_thoughts_after_repair": report.get("bad_thoughts_after_repair", 0),
        "english_thoughts_after_repair": report.get("english_thoughts_after_repair", 0),
        "mixed_language_tokens_after_repair": report.get("mixed_language_tokens_after_repair", 0),
        "weird_tokens_after_repair": report.get("weird_tokens_after_repair", 0),
        "ungrounded_thoughts_after_repair": report.get("ungrounded_thoughts_after_repair", 0),
        "bad_group_summaries_after_repair": report.get("bad_group_summaries_after_repair", 0),
        "relation_explanation_contradictions": report.get("relation_explanation_contradictions", 0),
        "english_explanations_remaining": report.get("english_explanations_remaining", 0),
    }
    blockers = [name for name, value in checks.items() if value]
    report["quality_gate_blockers"] = blockers
    report["quality_gate_passed"] = not blockers
    report["quality_gate_status"] = "passed" if not blockers else "quality_failed"


def _close_runtime_group(
    groups: list[dict[str, Any]],
    current_thoughts: list[RuntimeThought],
    current_main_idea: str,
) -> None:
    if not current_thoughts:
        return
    groups.append(
        {
            "index": len(groups) + 1,
            "start_sentence_index": current_thoughts[0].sentence_index,
            "end_sentence_index": current_thoughts[-1].sentence_index,
            "main_thought": current_main_idea or current_thoughts[0].thought_text,
            "sentence_indexes": [item.sentence_index for item in current_thoughts],
            "thought_runtime_indexes": [item.index for item in current_thoughts],
            "thought_ids": [item.db_id for item in current_thoughts if item.db_id],
        }
    )


def _build_sequential_groups(
    thoughts: list[RuntimeThought],
    *,
    model_name: str | None = None,
    report: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Mandatory sequential accumulation algorithm from the assignment."""

    current_block_thoughts: list[RuntimeThought] = []
    current_block_main_idea = ""
    groups: list[dict[str, Any]] = []

    for thought in thoughts:
        if not thought.is_meaningful or thought.noise or not thought.thought_text:
            continue

        if not current_block_thoughts:
            current_block_thoughts = [thought]
            current_block_main_idea = thought.thought_text
            continue

        result = compare_with_current_block(
            current_block_main_idea=current_block_main_idea,
            current_block_thoughts=[item.thought_text for item in current_block_thoughts],
            new_thought=thought.thought_text,
            model_name=model_name,
        )
        if progress_callback:
            progress_callback(
                f"sequential_compare sentence={thought.sentence_index} current_block_size={len(current_block_thoughts)} "
                f"same={result.get('same_block')} score={result.get('score')}"
            )
        if result.get("fallback_used"):
            report["fallback_count"] += 1
        if not result.get("json_valid"):
            report["invalid_json_count"] += 1
        result_flags = [str(item) for item in result.get("quality_flags", [])] if isinstance(result.get("quality_flags"), list) else []
        group_flags = [str(item) for item in result.get("group_summary_flags", [])] if isinstance(result.get("group_summary_flags"), list) else []
        if "safe_group_summary_fallback_used" in result_flags:
            report["safe_group_summary_fallback_used"] += 1
        if group_flags and "safe_group_summary_fallback_used" not in result_flags:
            report["bad_group_summaries_after_repair"] += 1

        same_block = bool(result.get("same_block")) and float(result.get("score", 0.0) or 0.0) >= THOUGHT_SAME_BLOCK_THRESHOLD
        if same_block:
            current_block_thoughts.append(thought)
            updated = str(result.get("updated_block_idea", "")).strip()
            current_block_main_idea = updated or f"{current_block_main_idea} {thought.thought_text}"[:1200]
        else:
            _close_runtime_group(groups, current_block_thoughts, current_block_main_idea)
            current_block_thoughts = [thought]
            current_block_main_idea = thought.thought_text

    if current_block_thoughts:
        _close_runtime_group(groups, current_block_thoughts, current_block_main_idea)
    return groups


def _run_pairwise_preview(
    thoughts: list[RuntimeThought],
    *,
    model_name: str | None,
    max_pairs: int | None,
    report: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    meaningful = [item for item in thoughts if item.is_meaningful and not item.noise and item.thought_text]
    total = _pairwise_total(len(meaningful))
    report["pairwise_comparisons_total"] = total
    limit = max_pairs if max_pairs is not None else DEFAULT_DRY_RUN_MAX_PAIRS
    llm_budget = max(0, int(os.getenv("THOUGHT_CHAIN_PREVIEW_PAIRWISE_LLM_LIMIT", "0")))
    relations: list[dict[str, Any]] = []
    for done, (left, right) in enumerate(combinations(meaningful, 2), start=1):
        if limit is not None and len(relations) >= limit:
            break
        payload = _deterministic_pair_payload(left, right)
        if payload is not None:
            report["pairwise_prefiltered_no_llm"] += 1
        elif report["pairwise_llm_calls"] < llm_budget:
            started = time.monotonic()
            payload = compare_pair(left.thought_text, right.thought_text, model_name=model_name)
            report["pairwise_llm_calls"] += 1
            if time.monotonic() - started > 30:
                report["pairwise_slow_calls"] += 1
        else:
            payload = _deterministic_pair_payload(left, right, force=True)
            report["pairwise_prefiltered_no_llm"] += 1
        if payload.get("fallback_used"):
            report["fallback_count"] += 1
        if not payload.get("json_valid"):
            report["invalid_json_count"] += 1
        _update_relation_quality_stats(report, payload, left.index, right.index)
        if payload.get("relation") in {"same", "related"} and float(payload.get("score", 0.0) or 0.0) >= THOUGHT_RELATION_THRESHOLD:
            report["relations_created"] += 1
        relations.append(
            {
                "source_index": left.index,
                "target_index": right.index,
                "relation": payload.get("relation"),
                "score": payload.get("score"),
                "explanation": payload.get("explanation", ""),
                "quality_flags": payload.get("quality_flags", []),
                "score_before_fix": payload.get("score_before_fix"),
                "english_explanation_detected": payload.get("english_explanation_detected", False),
                "english_explanation_retried": payload.get("english_explanation_retried", False),
                "english_explanation_remaining": payload.get("english_explanation_remaining", False),
                "english_explanation_sanitized": payload.get("english_explanation_sanitized", False),
                "relation_explanation_consistent": payload.get("relation_explanation_consistent", True),
                "relation_explanation_contradiction": payload.get("relation_explanation_contradiction", False),
                "relation_explanation_rewritten": payload.get("relation_explanation_rewritten", False),
                "relation_explanation_problem": payload.get("relation_explanation_problem", ""),
                "semantic_guard_applied": payload.get("semantic_guard_applied", False),
                "semantic_guard_reason": payload.get("semantic_guard_reason", ""),
            }
        )
        report["pairwise_comparisons_done"] = done
        if progress_callback and (done == 1 or done % 10 == 0):
            progress_callback(f"pairwise_preview done={done} total={total} limit={limit}")
    _finalize_pairwise_distribution(report, relations)
    return relations


def run_thought_chain_preview(
    parsed_book: Any,
    *,
    max_sentences: int = 30,
    max_pairs: int | None = None,
    model_name: str | None = None,
    skip_pairwise: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Dry-run thought-chain analysis without DB writes."""

    report = _empty_report(getattr(parsed_book, "title", ""), model_name=model_name, status="dry_run")
    sentences = split_book_into_sentences(parsed_book, max_sentences=max_sentences)
    report["total_sentences"] = len(sentences)

    thoughts: list[RuntimeThought] = []
    for index, sentence in enumerate(sentences, start=1):
        payload = extract_thought_from_sentence(sentence.text, model_name=model_name)
        if payload.get("fallback_used"):
            report["fallback_count"] += 1
        if not payload.get("json_valid"):
            report["invalid_json_count"] += 1
        thought = _runtime_thought_from_payload(sentence, payload, index)
        thoughts.append(thought)
        report["processed_sentences"] = index
        if progress_callback:
            progress_callback(
                f"thought_extraction sentence={index}/{len(sentences)} meaningful={thought.is_meaningful} "
                f"json_valid={thought.json_valid} fallback={thought.fallback_used}"
            )

    _update_meaning_stats(report, thoughts)
    meaningful = [item for item in thoughts if item.is_meaningful and not item.noise and item.thought_text]

    groups = _build_sequential_groups(
        meaningful,
        model_name=model_name,
        report=report,
        progress_callback=progress_callback,
    )
    report["sequential_groups_created"] = len(groups)
    report["sequential_groups_sample"] = groups[:10]

    if not skip_pairwise:
        relations = _run_pairwise_preview(
            meaningful,
            model_name=model_name,
            max_pairs=max_pairs,
            report=report,
            progress_callback=progress_callback,
        )
        report["pairwise_relations_sample"] = relations[:20]

    _apply_quality_gate(report)
    report["finished_at"] = timezone.now().isoformat()
    return report


def _cleanup_previous_run(book: UserBook, global_book: GlobalBookCache) -> None:
    thoughts = SentenceThought.objects.filter(global_book=global_book)
    ThoughtRelation.objects.filter(source_thought__in=thoughts).delete()
    ThoughtRelation.objects.filter(target_thought__in=thoughts).delete()
    ThoughtBlockMembership.objects.filter(thought__in=thoughts).delete()
    SequentialThoughtGroup.objects.filter(global_book=global_book).delete()
    BookSentence.objects.filter(global_book=global_book).delete()

    for block in GlobalLogicalThoughtBlock.objects.filter(source_books=book).distinct():
        if block.source_books.count() <= 1:
            block.delete()
        else:
            block.source_books.remove(book)


def _persist_sentences_and_thoughts(
    *,
    book: UserBook,
    global_book: GlobalBookCache,
    sentences: list[ThoughtChainSentence],
    run: ThoughtChainAnalysisRun,
    model_name: str | None,
    report: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> list[RuntimeThought]:
    thoughts: list[RuntimeThought] = []
    for index, sentence in enumerate(sentences, start=1):
        with transaction.atomic():
            book_sentence, _ = BookSentence.objects.update_or_create(
                global_book=global_book,
                index=sentence.index,
                defaults={
                    "book": book,
                    "text": sentence.text,
                    "source_start": sentence.source_start,
                    "source_end": sentence.source_end,
                    "chapter_title": sentence.chapter_title[:512],
                    "section_title": sentence.section_title[:512],
                    "paragraph_index": sentence.paragraph_index,
                },
            )
            existing = getattr(book_sentence, "thought", None)
            if existing:
                runtime = RuntimeThought(
                    index=existing.index,
                    sentence_index=book_sentence.index,
                    sentence_text=book_sentence.text,
                    thought_text=existing.thought_text,
                    normalized_thought=existing.normalized_thought,
                    terms=list(existing.terms or []),
                    chapter_title=book_sentence.chapter_title,
                    section_title=book_sentence.section_title,
                    paragraph_index=book_sentence.paragraph_index,
                    is_meaningful=existing.is_meaningful,
                    noise=existing.noise,
                    skip_reason=existing.skip_reason,
                    json_valid=existing.json_valid,
                    fallback_used=existing.fallback_used,
                    terms_removed_count=0,
                    terms_removed_examples=[],
                    db_id=existing.id,
                )
            else:
                payload = extract_thought_from_sentence(sentence.text, model_name=model_name)
                if payload.get("fallback_used"):
                    report["fallback_count"] += 1
                if not payload.get("json_valid"):
                    report["invalid_json_count"] += 1
                thought_obj = SentenceThought.objects.create(
                    sentence=book_sentence,
                    book=book,
                    global_book=global_book,
                    index=sentence.index,
                    thought_text=str(payload.get("thought", ""))[:4000],
                    normalized_thought=str(payload.get("normalized_thought", ""))[:4000],
                    terms=payload.get("terms", []) if isinstance(payload.get("terms"), list) else [],
                    is_meaningful=bool(payload.get("is_meaningful", True)) and not bool(payload.get("noise", False)),
                    noise=bool(payload.get("noise", False)),
                    skip_reason=str(payload.get("skip_reason", ""))[:2000],
                    quality_flags=payload.get("quality_flags", []) if isinstance(payload.get("quality_flags"), list) else [],
                    llm_raw_response=payload.get("llm_raw_response", {}) if isinstance(payload.get("llm_raw_response"), dict) else {},
                    json_valid=bool(payload.get("json_valid", False)),
                    fallback_used=bool(payload.get("fallback_used", False)),
                )
                runtime = _runtime_thought_from_payload(sentence, payload, sentence.index)
                runtime.db_id = thought_obj.id

            thoughts.append(runtime)
            run.processed_sentences = index
            run.total_thoughts = len(thoughts)
            run.checkpoint = {"stage": "thought_extraction", "sentence_index": sentence.index}
            run.save(update_fields=["processed_sentences", "total_thoughts", "checkpoint"])
            report["processed_sentences"] = index
            if progress_callback:
                progress_callback(
                    f"persist_thought sentence={index}/{len(sentences)} meaningful={runtime.is_meaningful} noise={runtime.noise} "
                    f"json_valid={runtime.json_valid} fallback={runtime.fallback_used}"
                )
    return thoughts


def _persist_sequential_groups(
    *,
    book: UserBook,
    global_book: GlobalBookCache,
    groups: list[dict[str, Any]],
) -> None:
    SequentialThoughtGroup.objects.filter(global_book=global_book).delete()
    for group in groups:
        SequentialThoughtGroup.objects.create(
            book=book,
            global_book=global_book,
            index=int(group["index"]),
            start_sentence_index=int(group["start_sentence_index"]),
            end_sentence_index=int(group["end_sentence_index"]),
            main_thought=str(group.get("main_thought", ""))[:4000],
            sentence_indexes=group.get("sentence_indexes", []),
            thought_ids=group.get("thought_ids", []),
            llm_raw_response={"source": "sequential_accumulation"},
        )


def _run_pairwise_persistent(
    *,
    thoughts: list[SentenceThought],
    run: ThoughtChainAnalysisRun,
    model_name: str | None,
    max_pairs: int | None,
    report: dict[str, Any],
    strict_pairwise_llm: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    total = _pairwise_total(len(thoughts))
    report["pairwise_comparisons_total"] = total
    done = 0
    related_created = 0
    for left, right in combinations(thoughts, 2):
        if max_pairs is not None and done >= max_pairs:
            break
        existing = ThoughtRelation.objects.filter(source_thought=left, target_thought=right).first()
        if existing:
            done += 1
            report["skipped_pairs"] += 1
            report["pairwise_comparisons_done"] = done
            if done % 100 == 0:
                run.total_relations_checked = done
                run.checkpoint = {"stage": "pairwise", "pairs_done": done, "pairs_total": total, "resume_existing": True}
                run.save(update_fields=["total_relations_checked", "checkpoint"])
                if progress_callback:
                    progress = round(done / max(1, total) * 100, 2)
                    progress_callback(
                        f"pairwise_persistent done={done} total={total} progress={progress}% "
                        f"llm_calls={report.get('pairwise_llm_calls', 0)} skipped_existing={report.get('skipped_pairs', 0)}"
                    )
            continue
        started = time.monotonic()
        payload = compare_pair(left.thought_text, right.thought_text, model_name=model_name)
        report["pairwise_llm_calls"] += 1
        if time.monotonic() - started > 30:
            report["pairwise_slow_calls"] += 1
        if payload.get("fallback_used"):
            report["fallback_count"] += 1
        if not payload.get("json_valid"):
            report["invalid_json_count"] += 1
        _update_relation_quality_stats(report, payload, left.index, right.index)
        relation = str(payload.get("relation", "different"))
        score = float(payload.get("score", 0.0) or 0.0)
        ThoughtRelation.objects.create(
            source_thought=left,
            target_thought=right,
            relation=relation if relation in {"same", "related", "different"} else "different",
            score=max(0.0, min(1.0, score)),
            explanation=str(payload.get("explanation", ""))[:1000],
            llm_raw_response=payload.get("llm_raw_response", {}) if isinstance(payload.get("llm_raw_response"), dict) else {},
        )
        done += 1
        if relation in {"same", "related"} and score >= THOUGHT_RELATION_THRESHOLD:
            related_created += 1
        if done % 10 == 0:
            run.total_relations_checked = done
            run.total_relations_created = related_created
            run.checkpoint = {
                "stage": "pairwise",
                "pairs_done": done,
                "pairs_total": total,
                "llm_calls": report.get("pairwise_llm_calls", 0),
                "skipped_pairs": report.get("skipped_pairs", 0),
                "strict_pairwise_llm": strict_pairwise_llm,
            }
            run.save(update_fields=["total_relations_checked", "total_relations_created", "checkpoint"])
            if progress_callback:
                progress = round(done / max(1, total) * 100, 2)
                progress_callback(
                    f"pairwise_persistent done={done} total={total} progress={progress}% "
                    f"llm_calls={report.get('pairwise_llm_calls', 0)} skipped_pairs={report.get('skipped_pairs', 0)} "
                    f"max_pairs={max_pairs}"
                )
    report["pairwise_comparisons_done"] = done
    report["relations_created"] = related_created
    run.total_relations_checked = done
    run.total_relations_created = related_created
    run.save(update_fields=["total_relations_checked", "total_relations_created"])


def run_thought_chain_analysis(
    *,
    book: UserBook,
    parsed_book: Any,
    global_book: GlobalBookCache,
    model_name: str | None = None,
    max_sentences: int | None = None,
    max_pairs: int | None = None,
    skip_pairwise: bool = False,
    skip_global_blocks: bool = False,
    force_refresh: bool = False,
    resume: bool = False,
    strict: bool = False,
    strict_pairwise_llm: bool = False,
    analysis_mode: str = "greedy",
    merge_same_title_blocks: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Persistent production-safe thought-chain analysis with checkpoints."""

    if force_refresh and not resume:
        _cleanup_previous_run(book, global_book)

    run = book.thought_chain_runs.order_by("-created_at").first() if resume else None
    if run is None or run.status not in {ThoughtChainAnalysisRun.Status.PENDING, ThoughtChainAnalysisRun.Status.RUNNING, ThoughtChainAnalysisRun.Status.FAILED}:
        run = ThoughtChainAnalysisRun.objects.create(
            book=book,
            status=ThoughtChainAnalysisRun.Status.RUNNING,
            model_name=model_name or "",
            started_at=timezone.now(),
        )
    else:
        run.status = ThoughtChainAnalysisRun.Status.RUNNING
        run.error_message = ""
        run.save(update_fields=["status", "error_message"])

    normalized_mode = "strict" if analysis_mode == "strict" or strict_pairwise_llm else "greedy"
    report = _empty_report(getattr(parsed_book, "title", book.title), model_name=model_name, status="running")
    report["strict_mode"] = bool(strict)
    report["strict_pairwise_llm"] = bool(strict_pairwise_llm)
    report["analysis_mode"] = normalized_mode
    report["block_generation_mode"] = normalized_mode
    report["merge_same_title_blocks"] = bool(merge_same_title_blocks)
    try:
        sentences = split_book_into_sentences(parsed_book, max_sentences=max_sentences)
        if progress_callback:
            progress_callback(f"sentence_split total={len(sentences)}")
        run.total_sentences = len(sentences)
        run.checkpoint = {"stage": "sentence_split", "sentences": len(sentences)}
        run.save(update_fields=["total_sentences", "checkpoint"])
        report["total_sentences"] = len(sentences)

        runtime_thoughts = _persist_sentences_and_thoughts(
            book=book,
            global_book=global_book,
            sentences=sentences,
            run=run,
            model_name=model_name,
            report=report,
            progress_callback=progress_callback,
        )
        _update_meaning_stats(report, runtime_thoughts)
        meaningful_runtime = [item for item in runtime_thoughts if item.is_meaningful and not item.noise and item.thought_text]

        groups = _build_sequential_groups(
            meaningful_runtime,
            model_name=model_name,
            report=report,
            progress_callback=progress_callback,
        )
        _persist_sequential_groups(book=book, global_book=global_book, groups=groups)
        report["sequential_groups_created"] = len(groups)
        report["sequential_groups_sample"] = groups[:10]

        thought_qs = list(
            SentenceThought.objects.filter(global_book=global_book, is_meaningful=True, noise=False)
            .select_related("sentence")
            .order_by("index")
        )
        if normalized_mode == "strict" and not skip_pairwise:
            _run_pairwise_persistent(
                thoughts=thought_qs,
                run=run,
                model_name=model_name,
                max_pairs=max_pairs,
                report=report,
                strict_pairwise_llm=strict_pairwise_llm,
                progress_callback=progress_callback,
            )

        if not skip_global_blocks:
            existing_stats = attach_to_existing_global_blocks(book=book, thoughts=thought_qs, model_name=model_name)
            for key in ("fallback_count", "invalid_json_count"):
                report[key] += int(existing_stats.get(key, 0) or 0)

            attached_ids = set(
                ThoughtBlockMembership.objects.filter(thought__in=thought_qs).values_list("thought_id", flat=True)
            )
            candidates = [item for item in thought_qs if item.id not in attached_ids]
            if normalized_mode == "strict":
                block_stats = create_global_blocks_from_relations(book=book, thoughts=candidates, model_name=model_name)
            else:
                block_stats = create_global_blocks_greedy(
                    book=book,
                    thoughts=candidates,
                    model_name=model_name,
                    merge_same_title_blocks_enabled=merge_same_title_blocks,
                )
            for key in ("fallback_count", "invalid_json_count"):
                report[key] += int(block_stats.get(key, 0) or 0)
            report["global_blocks_created"] = int(block_stats.get("global_blocks_created", 0) or 0)
            report["memberships_created"] = int(existing_stats.get("existing_block_memberships", 0) or 0) + int(block_stats.get("memberships_created", 0) or 0)
            report["avg_membership_score"] = float(block_stats.get("avg_membership_score", 0.0) or 0.0)
            report["greedy_comparisons_done"] = int(block_stats.get("greedy_comparisons_done", 0) or 0)
            report["greedy_seed_blocks"] = int(block_stats.get("greedy_seed_blocks", 0) or 0)
            report["merged_blocks"] = int(block_stats.get("merged_blocks", 0) or 0)
            report["merge_memberships_moved"] = int(block_stats.get("merge_memberships_moved", 0) or 0)

        _apply_quality_gate(report)
        if not report.get("quality_gate_passed"):
            report["status"] = "quality_failed"
            report["finished_at"] = timezone.now().isoformat()
            book.status = UserBook.Status.FAILED
            book.current_stage = "thought_chain_quality_failed"
            book.error_message = "Thought-chain quality gate failed: " + ", ".join(report.get("quality_gate_blockers", []))
            book.save(update_fields=["status", "current_stage", "error_message"])
            run.status = ThoughtChainAnalysisRun.Status.FAILED
            run.error_message = book.error_message[:2000]
            run.report = report
            run.finished_at = timezone.now()
            run.checkpoint = {"stage": "quality_failed", "blockers": report.get("quality_gate_blockers", [])}
            run.save(update_fields=["status", "error_message", "report", "finished_at", "checkpoint"])
            return report

        report["status"] = "ready"
        report["finished_at"] = timezone.now().isoformat()
        book.status = UserBook.Status.READY
        book.current_stage = "thought_chain_ready"
        book.progress_percent = 100
        book.processed_at = timezone.now()
        book.error_message = ""
        book.save(update_fields=["status", "current_stage", "progress_percent", "processed_at", "error_message"])
        run.status = ThoughtChainAnalysisRun.Status.READY
        run.total_blocks_created = int(report.get("global_blocks_created", 0) or 0)
        run.report = report
        run.finished_at = timezone.now()
        run.checkpoint = {"stage": "finished"}
        run.save(update_fields=["status", "total_blocks_created", "report", "finished_at", "checkpoint"])
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error_message"] = str(exc)
        report["finished_at"] = timezone.now().isoformat()
        book.status = UserBook.Status.FAILED
        book.current_stage = "thought_chain_failed"
        book.error_message = str(exc)[:2000]
        book.save(update_fields=["status", "current_stage", "error_message"])
        run.status = ThoughtChainAnalysisRun.Status.FAILED
        run.error_message = str(exc)[:2000]
        run.report = report
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error_message", "report", "finished_at"])
        raise
