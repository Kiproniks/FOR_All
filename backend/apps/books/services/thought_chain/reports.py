from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from apps.books.services.thought_chain.sentence_splitter import ThoughtChainSentence


NOISE_RE = re.compile(
    r"(?:isbn|copyright|all rights reserved|все права защищены|©|оглавление|содержание|рис\.|таблица)",
    re.IGNORECASE,
)


def _sentence_flags(text: str) -> list[str]:
    value = " ".join((text or "").split()).strip()
    words = re.findall(r"[\wА-Яа-яЁё]+", value, flags=re.UNICODE)
    flags: list[str] = []
    if len(value) < 25 or len(words) < 4:
        flags.append("too_short")
    if len(value) > 700 or len(words) > 120:
        flags.append("too_long")
    punctuation = sum(1 for char in value if not char.isalnum() and not char.isspace())
    if (
        NOISE_RE.search(value)
        or value.isdigit()
        or (len(value) <= 12 and not re.search(r"[А-Яа-яA-Za-z]", value))
        or (value and punctuation / max(1, len(value)) > 0.35)
    ):
        flags.append("maybe_noise")
    return flags or ["ok"]


def _sentence_row(sentence: ThoughtChainSentence) -> dict[str, Any]:
    flags = _sentence_flags(sentence.text)
    return {
        "index": sentence.index,
        "length": len(sentence.text),
        "word_count": len(re.findall(r"[\wА-Яа-яЁё]+", sentence.text, flags=re.UNICODE)),
        "flags": flags,
        "chapter_title": sentence.chapter_title,
        "section_title": sentence.section_title,
        "paragraph_index": sentence.paragraph_index,
        "text": sentence.text,
    }


def _escape_md(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def _markdown_table(rows: list[dict[str, Any]], *, text_limit: int = 260) -> list[str]:
    lines = [
        "| # | Length | Flags | Chapter | Section | Text |",
        "|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        text = _escape_md(row["text"])
        if len(text) > text_limit:
            text = text[: text_limit - 1].rstrip() + "…"
        lines.append(
            f"| {row['index']} | {row['length']} | {', '.join(row['flags'])} | "
            f"{_escape_md(row['chapter_title'])} | {_escape_md(row['section_title'])} | {text} |"
        )
    return lines


def write_sentence_preview_reports(
    *,
    file_name: str,
    sentences: list[ThoughtChainSentence],
    preview_count: int,
    output_dir: str | Path = ".",
) -> dict[str, str]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "sentence_preview.json"
    md_path = path / "sentence_preview.md"

    rows = [_sentence_row(sentence) for sentence in sentences]
    shown = rows[:preview_count]
    shortest = sorted(rows, key=lambda row: (row["length"], row["index"]))[:20]
    longest = sorted(rows, key=lambda row: (-row["length"], row["index"]))[:20]
    noise = [row for row in rows if "maybe_noise" in row["flags"]][:50]
    too_short_count = sum(1 for row in rows if "too_short" in row["flags"])
    too_long_count = sum(1 for row in rows if "too_long" in row["flags"])
    noise_count = sum(1 for row in rows if "maybe_noise" in row["flags"])
    can_run_llm_test = bool(rows) and noise_count / max(1, len(rows)) < 0.25 and too_long_count / max(1, len(rows)) < 0.10

    payload = {
        "file_name": file_name,
        "total_sentences": len(rows),
        "preview_count": len(shown),
        "can_run_llm_test": can_run_llm_test,
        "stats": {
            "too_short_count": too_short_count,
            "too_long_count": too_long_count,
            "maybe_noise_count": noise_count,
        },
        "first_sentences": shown,
        "shortest_sentences": shortest,
        "longest_sentences": longest,
        "potential_noise": noise,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Sentence Preview",
        "",
        f"File: {file_name}",
        f"Total sentences: {len(rows)}",
        f"Preview count: {len(shown)}",
        f"Can run LLM test: {can_run_llm_test}",
        "",
        "## Stats",
        "",
        f"- too_short: {too_short_count}",
        f"- too_long: {too_long_count}",
        f"- maybe_noise: {noise_count}",
        "",
        "## First sentences",
        "",
    ]
    lines.extend(_markdown_table(shown))
    lines.extend(["", "## Shortest sentences", ""])
    lines.extend(_markdown_table(shortest, text_limit=180))
    lines.extend(["", "## Longest sentences", ""])
    lines.extend(_markdown_table(longest, text_limit=220))
    lines.extend(["", "## Potential noise", ""])
    if noise:
        lines.extend(_markdown_table(noise[:20], text_limit=220))
    else:
        lines.append("No obvious noise detected in sentence preview.")
    lines.extend([
        "",
        "## Conclusion",
        "",
        "LLM mini-test can be started." if can_run_llm_test else "Review sentence splitting/noise before LLM mini-test.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def write_thought_chain_reports(report: dict[str, Any], *, output_dir: str | Path = ".") -> dict[str, str]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "thought_chain_analysis_report.json"
    md_path = path / "thought_chain_analysis_report.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Thought-chain analysis report",
        "",
        f"- book_title: {report.get('book_title', '')}",
        f"- mode: {report.get('mode', '')}",
        f"- analysis_mode: {report.get('analysis_mode', '')}",
        f"- block_generation_mode: {report.get('block_generation_mode', '')}",
        f"- model: {report.get('model', '')}",
        f"- status: {report.get('status', '')}",
        f"- strict_mode: {report.get('strict_mode', False)}",
        f"- strict_pairwise_llm: {report.get('strict_pairwise_llm', False)}",
        f"- merge_same_title_blocks: {report.get('merge_same_title_blocks', False)}",
        f"- total_sentences: {report.get('total_sentences', 0)}",
        f"- meaningful_sentences: {report.get('meaningful_sentences', 0)}",
        f"- noise_sentences: {report.get('noise_sentences', 0)}",
        f"- skipped_sentences: {report.get('skipped_sentences', 0)}",
        f"- thoughts_created: {report.get('thoughts_created', 0)}",
        f"- sequential_groups_created: {report.get('sequential_groups_created', 0)}",
        f"- pairwise_comparisons_total: {report.get('pairwise_comparisons_total', 0)}",
        f"- pairwise_comparisons_done: {report.get('pairwise_comparisons_done', 0)}",
        f"- pairwise_llm_calls: {report.get('pairwise_llm_calls', 0)}",
        f"- pairwise_prefiltered_no_llm: {report.get('pairwise_prefiltered_no_llm', 0)}",
        f"- pairwise_slow_calls: {report.get('pairwise_slow_calls', 0)}",
        f"- skipped_pairs: {report.get('skipped_pairs', 0)}",
        f"- relations_created: {report.get('relations_created', 0)}",
        f"- global_blocks_created: {report.get('global_blocks_created', 0)}",
        f"- memberships_created: {report.get('memberships_created', 0)}",
        f"- avg_membership_score: {report.get('avg_membership_score', 0.0)}",
        f"- greedy_comparisons_done: {report.get('greedy_comparisons_done', 0)}",
        f"- greedy_seed_blocks: {report.get('greedy_seed_blocks', 0)}",
        f"- merged_blocks: {report.get('merged_blocks', 0)}",
        f"- merge_memberships_moved: {report.get('merge_memberships_moved', 0)}",
        f"- fallback_count: {report.get('fallback_count', 0)}",
        f"- invalid_json_count: {report.get('invalid_json_count', 0)}",
        f"- timeout_count: {report.get('timeout_count', 0)}",
        f"- relation_score_inconsistencies: {report.get('relation_score_inconsistencies', 0)}",
        f"- relation_score_fixed: {report.get('relation_score_fixed', 0)}",
        f"- same_count: {report.get('same_count', 0)}",
        f"- related_count: {report.get('related_count', 0)}",
        f"- different_count: {report.get('different_count', 0)}",
        f"- relation_rate: {report.get('relation_rate', 0.0)}",
        f"- same_ratio_too_high: {report.get('same_ratio_too_high', False)}",
        f"- related_ratio_too_high: {report.get('related_ratio_too_high', False)}",
        f"- hub_thought_overlinked: {report.get('hub_thought_overlinked', False)}",
        f"- semantic_guard_applied_count: {report.get('semantic_guard_applied_count', 0)}",
        f"- relation_explanation_contradictions: {report.get('relation_explanation_contradictions', 0)}",
        f"- relation_explanation_rewritten: {report.get('relation_explanation_rewritten', 0)}",
        f"- english_explanations_detected_total: {report.get('english_explanations_detected_total', 0)}",
        f"- english_explanations_retried: {report.get('english_explanations_retried', 0)}",
        f"- english_explanations_sanitized: {report.get('english_explanations_sanitized', 0)}",
        f"- english_explanations_remaining: {report.get('english_explanations_remaining', 0)}",
        f"- weird_tokens_in_thoughts: {report.get('weird_tokens_in_thoughts', 0)}",
        f"- mixed_language_tokens: {report.get('mixed_language_tokens', 0)}",
        f"- ungrounded_thoughts: {report.get('ungrounded_thoughts', 0)}",
        f"- bad_thoughts_detected: {report.get('bad_thoughts_detected', 0)}",
        f"- bad_thoughts_after_repair: {report.get('bad_thoughts_after_repair', 0)}",
        f"- english_thoughts_after_repair: {report.get('english_thoughts_after_repair', 0)}",
        f"- mixed_language_tokens_after_repair: {report.get('mixed_language_tokens_after_repair', 0)}",
        f"- weird_tokens_after_repair: {report.get('weird_tokens_after_repair', 0)}",
        f"- ungrounded_thoughts_after_repair: {report.get('ungrounded_thoughts_after_repair', 0)}",
        f"- thought_retries: {report.get('thought_retries', 0)}",
        f"- thought_retry_success: {report.get('thought_retry_success', 0)}",
        f"- safe_sentence_fallback_used: {report.get('safe_sentence_fallback_used', 0)}",
        f"- safe_group_summary_fallback_used: {report.get('safe_group_summary_fallback_used', 0)}",
        f"- bad_group_summaries_after_repair: {report.get('bad_group_summaries_after_repair', 0)}",
        f"- quality_gate_passed: {report.get('quality_gate_passed', False)}",
        f"- quality_gate_blockers: {report.get('quality_gate_blockers', [])}",
        f"- terms_removed_count: {report.get('terms_removed_count', 0)}",
        "",
        "## First Sentences / Thoughts",
    ]
    for item in report.get("sentence_thoughts_sample", [])[:15]:
        lines.extend([
            "",
            f"### Sentence {item.get('sentence_index')}",
            f"- sentence: {item.get('sentence_text', '')}",
            f"- thought: {item.get('thought', '')}",
            f"- pre_repair_thought: {item.get('pre_repair_thought', '')}",
            f"- is_meaningful: {item.get('is_meaningful')}",
            f"- noise: {item.get('noise')}",
            f"- skip_reason: {item.get('skip_reason', '')}",
            f"- terms: {', '.join(item.get('terms', [])[:8])}",
            f"- terms_removed_count: {item.get('terms_removed_count', 0)}",
            f"- quality_flags: {item.get('quality_flags', [])}",
            f"- weird_token_examples: {item.get('weird_token_examples', [])}",
        ])

    lines.extend(["", "## Terms Quality", ""])
    if report.get("terms_removed_examples"):
        lines.append("Removed term examples:")
        for term in report.get("terms_removed_examples", [])[:20]:
            lines.append(f"- {term}")
    else:
        lines.append("Removed term examples: none")
    lines.append("")
    if report.get("final_terms_examples"):
        lines.append("Final term examples:")
        for term in report.get("final_terms_examples", [])[:20]:
            lines.append(f"- {term}")
    else:
        lines.append("Final term examples: none")

    lines.extend([
        "",
        "## Skipped / Noise Sentences",
        "",
        "| # | Text | Reason |",
        "|---:|---|---|",
    ])
    for item in report.get("noise_examples", [])[:20]:
        text = str(item.get("text", "")).replace("|", "\\|")
        reason = str(item.get("reason", "")).replace("|", "\\|")
        lines.append(f"| {item.get('sentence_index')} | {text[:240]} | {reason[:160]} |")

    lines.extend([
        "",
        "## Skip Reason Top",
        "",
    ])
    if report.get("skip_reason_top"):
        for reason, count in report.get("skip_reason_top", {}).items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Sequential groups sample",
    ]
    )
    for item in report.get("sequential_groups_sample", [])[:10]:
        lines.extend([
            "",
            f"### Group {item.get('index')}",
            f"- sentences: {item.get('start_sentence_index')}..{item.get('end_sentence_index')}",
            f"- main_thought: {item.get('main_thought', '')}",
        ])
    lines.extend(["", "## Pairwise relations sample"])
    for item in report.get("pairwise_relations_sample", [])[:10]:
        lines.extend([
            "",
            f"- source_index: {item.get('source_index')}",
            f"- target_index: {item.get('target_index')}",
            f"- relation: {item.get('relation')}",
            f"- score: {item.get('score')}",
            f"- explanation: {item.get('explanation', '')}",
            f"- quality_flags: {item.get('quality_flags', [])}",
            f"- score_before_fix: {item.get('score_before_fix', '')}",
            f"- english_explanation_sanitized: {item.get('english_explanation_sanitized', False)}",
        ])
    lines.extend(["", "## Relation quality", ""])
    lines.append(f"- relation_score_inconsistencies: {report.get('relation_score_inconsistencies', 0)}")
    lines.append(f"- relation_score_fixed: {report.get('relation_score_fixed', 0)}")
    lines.append(f"- same_count: {report.get('same_count', 0)}")
    lines.append(f"- related_count: {report.get('related_count', 0)}")
    lines.append(f"- different_count: {report.get('different_count', 0)}")
    lines.append(f"- relation_rate: {report.get('relation_rate', 0.0)}")
    lines.append(f"- same_ratio_too_high: {report.get('same_ratio_too_high', False)}")
    lines.append(f"- related_ratio_too_high: {report.get('related_ratio_too_high', False)}")
    lines.append(f"- hub_thought_overlinked: {report.get('hub_thought_overlinked', False)}")
    lines.append(f"- semantic_guard_applied_count: {report.get('semantic_guard_applied_count', 0)}")
    lines.append(f"- relation_explanation_contradictions: {report.get('relation_explanation_contradictions', 0)}")
    lines.append(f"- relation_explanation_rewritten: {report.get('relation_explanation_rewritten', 0)}")
    lines.append(f"- english_explanations_detected_total: {report.get('english_explanations_detected_total', 0)}")
    lines.append(f"- english_explanations_retried: {report.get('english_explanations_retried', 0)}")
    lines.append(f"- english_explanations_sanitized: {report.get('english_explanations_sanitized', 0)}")
    lines.append(f"- english_explanations_remaining: {report.get('english_explanations_remaining', 0)}")
    if report.get("hub_thoughts"):
        lines.append("")
        lines.append("Hub thought warnings:")
        for item in report.get("hub_thoughts", [])[:10]:
            lines.append(
                f"- thought {item.get('thought_index')}: same_links={item.get('same_links')} "
                f"seen_pairs={item.get('seen_pairs')}"
            )
    if report.get("suspicious_same_examples"):
        lines.append("")
        lines.append("Suspicious same examples:")
        for item in report.get("suspicious_same_examples", [])[:10]:
            lines.append(
                f"- {item.get('source_index')}->{item.get('target_index')}: "
                f"score={item.get('score')} flags={item.get('quality_flags', [])}"
            )
    lines.extend(["", "## Thought quality", ""])
    lines.append(f"- weird_tokens_in_thoughts: {report.get('weird_tokens_in_thoughts', 0)}")
    lines.append(f"- mixed_language_tokens: {report.get('mixed_language_tokens', 0)}")
    lines.append(f"- ungrounded_thoughts: {report.get('ungrounded_thoughts', 0)}")
    lines.append(f"- bad_thoughts_detected: {report.get('bad_thoughts_detected', 0)}")
    lines.append(f"- bad_thoughts_after_repair: {report.get('bad_thoughts_after_repair', 0)}")
    lines.append(f"- english_thoughts_detected: {report.get('english_thoughts_detected', 0)}")
    lines.append(f"- english_thoughts_after_repair: {report.get('english_thoughts_after_repair', 0)}")
    lines.append(f"- mixed_language_tokens_after_repair: {report.get('mixed_language_tokens_after_repair', 0)}")
    lines.append(f"- weird_tokens_after_repair: {report.get('weird_tokens_after_repair', 0)}")
    lines.append(f"- ungrounded_thoughts_after_repair: {report.get('ungrounded_thoughts_after_repair', 0)}")
    lines.append(f"- thought_retries: {report.get('thought_retries', 0)}")
    lines.append(f"- thought_retry_success: {report.get('thought_retry_success', 0)}")
    lines.append(f"- safe_sentence_fallback_used: {report.get('safe_sentence_fallback_used', 0)}")
    lines.append(f"- safe_group_summary_fallback_used: {report.get('safe_group_summary_fallback_used', 0)}")
    lines.append(f"- bad_group_summaries_after_repair: {report.get('bad_group_summaries_after_repair', 0)}")
    lines.append("")
    lines.append("## Quality gate")
    lines.append("")
    lines.append(f"- quality_gate_passed: {report.get('quality_gate_passed', False)}")
    lines.append(f"- quality_gate_status: {report.get('quality_gate_status', '')}")
    lines.append(f"- quality_gate_blockers: {report.get('quality_gate_blockers', [])}")
    if report.get("thought_quality_examples"):
        for item in report.get("thought_quality_examples", [])[:10]:
            lines.append(
                f"- sentence {item.get('sentence_index')}: flags={item.get('flags', [])}; "
                f"pre_flags={item.get('pre_repair_flags', [])}; examples={item.get('examples', [])}; "
                f"before={str(item.get('pre_repair_thought', ''))[:160]}; after={str(item.get('thought', ''))[:160]}"
            )
    if report.get("relation_score_fix_examples"):
        lines.append("")
        lines.append("Score fix examples:")
        for item in report.get("relation_score_fix_examples", [])[:10]:
            lines.append(
                f"- {item.get('source_index')}->{item.get('target_index')}: "
                f"{item.get('relation')} {item.get('score_before_fix')} -> {item.get('score_after_fix')}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
