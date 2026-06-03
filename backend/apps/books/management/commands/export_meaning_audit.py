from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.books.models import BookTheme, ConceptMention, LogicalBlock, ThemeSubtopic, UserBook

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
NOISE_RE = re.compile(
    r"(?:\bISBN\b|©|copyright|all rights reserved|все права защищены|переводч|издательств|тираж)",
    re.IGNORECASE,
)
GENERIC_RE = re.compile(
    r"(?:в тексте рассматривается|данный раздел|основная идея|source material|general theme|key ideas)",
    re.IGNORECASE,
)


def _norm_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _word_set(value: str) -> set[str]:
    result: set[str] = set()
    for token in WORD_RE.findall((value or "").lower()):
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        result.add(token)
    return result


def _overlap_score(summary: str, source: str) -> float:
    left = _word_set(summary)
    right = _word_set(source)
    if not left or not right:
        return 0.0
    return round(len(left & right) / max(1, len(left)), 4)


class Command(BaseCommand):
    help = "Export audit file for manual validation of semantic correctness"

    def add_arguments(self, parser):
        parser.add_argument("--user-email", type=str, default="", help="Filter by user email")
        parser.add_argument("--user-id", type=int, default=0, help="Filter by user id")
        parser.add_argument("--book-id", type=int, default=0, help="Filter by user book id")
        parser.add_argument("--max-source-chars", type=int, default=1600, help="Source excerpt size in output")
        parser.add_argument("--output-dir", type=str, default="", help="Directory for JSON/CSV files")

    def handle(self, *args, **options):
        user_email = str(options.get("user_email") or "").strip()
        user_id = int(options.get("user_id") or 0)
        book_id = int(options.get("book_id") or 0)
        max_source_chars = max(400, int(options.get("max_source_chars") or 1600))
        output_dir_raw = str(options.get("output_dir") or "").strip()

        qs = (
            UserBook.objects.filter(global_cache__isnull=False, status=UserBook.Status.READY)
            .select_related("user", "global_cache")
            .order_by("user_id", "-uploaded_at")
        )

        if user_email:
            qs = qs.filter(user__email=user_email)
        if user_id:
            qs = qs.filter(user_id=user_id)
        if book_id:
            qs = qs.filter(id=book_id)

        books = list(qs)
        if not books:
            raise CommandError("No ready analyzed books matched filters.")

        if output_dir_raw:
            output_dir = Path(output_dir_raw).expanduser().resolve()
        else:
            output_dir = Path(__file__).resolve().parents[5] / "run_logs"
        output_dir.mkdir(parents=True, exist_ok=True)

        now = timezone.localtime()
        stamp = now.strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"meaning_audit_{stamp}.json"
        csv_path = output_dir / f"meaning_audit_blocks_{stamp}.csv"

        report: dict[str, Any] = {
            "generated_at": now.isoformat(),
            "filters": {
                "user_email": user_email or None,
                "user_id": user_id or None,
                "book_id": book_id or None,
            },
            "books_count": len(books),
            "books": [],
        }
        csv_rows: list[dict[str, Any]] = []

        for user_book in books:
            cache = user_book.global_cache
            if not cache:
                continue

            blocks = list(
                LogicalBlock.objects.filter(global_book=cache)
                .order_by("order_number")
            )
            themes = list(
                BookTheme.objects.filter(global_book=cache)
                .prefetch_related("subtopics")
                .order_by("order_number", "id")
            )
            mentions = list(
                ConceptMention.objects.filter(global_book=cache)
                .select_related("concept", "logical_block")
                .order_by("-importance_score", "id")
            )

            mentions_by_block: dict[int, list[ConceptMention]] = defaultdict(list)
            for mention in mentions:
                mentions_by_block[mention.logical_block_id].append(mention)

            summary_source_parts = []
            for block in blocks:
                semantic_data = block.semantic_data if isinstance(block.semantic_data, dict) else {}
                if semantic_data.get("front_matter"):
                    continue
                if block.short_summary:
                    summary_source_parts.append(block.short_summary)
            if not summary_source_parts:
                summary_source_parts = [block.short_summary for block in blocks if block.short_summary]

            summary_source = "\n".join(summary_source_parts)
            full_summary = cache.full_summary or ""
            summary_overlap = _overlap_score(full_summary, summary_source)

            summary_flags: list[str] = []
            if not full_summary.strip():
                summary_flags.append("summary_empty")
            if summary_overlap < 0.08:
                summary_flags.append("summary_low_overlap")
            if NOISE_RE.search(full_summary):
                summary_flags.append("summary_has_service_noise")
            if GENERIC_RE.search(full_summary):
                summary_flags.append("summary_generic_phrasing")

            block_payloads: list[dict[str, Any]] = []
            for block in blocks:
                semantic_data = block.semantic_data if isinstance(block.semantic_data, dict) else {}
                clean_text = _norm_text(str(semantic_data.get("clean_text_for_analysis", "")))
                source_text = _norm_text(block.source_text)
                source_for_check = clean_text or source_text
                overlap = _overlap_score(block.short_summary or "", source_for_check)
                block_mentions = mentions_by_block.get(block.id, [])

                block_flags: list[str] = []
                if len(_norm_text(block.short_summary)) < 40:
                    block_flags.append("block_summary_too_short")
                if overlap < 0.07:
                    block_flags.append("block_summary_low_overlap")
                if semantic_data.get("front_matter"):
                    block_flags.append("front_matter_block")
                if not block_mentions:
                    block_flags.append("no_concept_mentions")
                if block_mentions and all(not _norm_text(item.source_quote) for item in block_mentions):
                    block_flags.append("mentions_without_quotes")

                top_mentions = []
                for mention in block_mentions[:7]:
                    top_mentions.append(
                        {
                            "concept": mention.concept.name,
                            "importance_score": float(mention.importance_score),
                            "short_explanation": mention.short_explanation,
                            "source_quote": mention.source_quote,
                        }
                    )

                block_payload = {
                    "block_id": block.id,
                    "order_number": block.order_number,
                    "title": block.title,
                    "chapter_title": block.chapter_title,
                    "start_paragraph": block.start_paragraph,
                    "end_paragraph": block.end_paragraph,
                    "token_count": block.token_count,
                    "summary": block.short_summary,
                    "summary_to_source_overlap": overlap,
                    "flags": block_flags,
                    "front_matter": bool(semantic_data.get("front_matter")),
                    "content_types": semantic_data.get("content_types", {}),
                    "source_excerpt": source_text[:max_source_chars],
                    "clean_excerpt": source_for_check[:max_source_chars],
                    "top_concepts": top_mentions,
                }
                block_payloads.append(block_payload)

                csv_rows.append(
                    {
                        "user_email": user_book.user.email,
                        "user_book_id": user_book.id,
                        "book_title": user_book.title,
                        "block_order": block.order_number,
                        "block_title": block.title,
                        "chapter_title": block.chapter_title,
                        "summary_overlap": overlap,
                        "front_matter": bool(semantic_data.get("front_matter")),
                        "flags": "|".join(block_flags),
                        "summary": _norm_text(block.short_summary)[:500],
                        "source_excerpt": source_for_check[:500].replace("\n", " "),
                    }
                )

            theme_payloads: list[dict[str, Any]] = []
            block_by_order = {block.order_number: block for block in blocks}
            for theme in themes:
                theme_blocks = []
                for order in range(theme.start_block_number, theme.end_block_number + 1):
                    candidate = block_by_order.get(order)
                    if candidate:
                        theme_blocks.append(candidate)

                theme_source = " ".join(item.short_summary for item in theme_blocks if item.short_summary)
                theme_overlap = _overlap_score(theme.summary or "", theme_source)
                theme_flags: list[str] = []
                if len(_norm_text(theme.summary)) < 60:
                    theme_flags.append("theme_summary_short")
                if theme_overlap < 0.07 and theme_source:
                    theme_flags.append("theme_summary_low_overlap")
                if NOISE_RE.search(theme.summary or ""):
                    theme_flags.append("theme_summary_has_service_noise")

                subtopics_payload = []
                for sub in theme.subtopics.all().order_by("-importance_score", "id"):
                    subtopics_payload.append(
                        {
                            "subtopic_id": sub.id,
                            "name": sub.name,
                            "importance_score": float(sub.importance_score),
                            "summary": sub.summary,
                            "source_quote": sub.source_quote,
                            "start_paragraph": sub.start_paragraph,
                            "end_paragraph": sub.end_paragraph,
                        }
                    )

                theme_payloads.append(
                    {
                        "theme_id": theme.id,
                        "order_number": theme.order_number,
                        "title": theme.title,
                        "chapter_title": theme.chapter_title,
                        "start_block_number": theme.start_block_number,
                        "end_block_number": theme.end_block_number,
                        "start_paragraph": theme.start_paragraph,
                        "end_paragraph": theme.end_paragraph,
                        "summary": theme.summary,
                        "summary_to_blocks_overlap": theme_overlap,
                        "flags": theme_flags,
                        "subtopics": subtopics_payload,
                    }
                )

            report["books"].append(
                {
                    "user_book_id": user_book.id,
                    "user_email": user_book.user.email,
                    "book_title": user_book.title,
                    "authors": user_book.authors,
                    "uploaded_at": user_book.uploaded_at.isoformat() if user_book.uploaded_at else None,
                    "processed_at": user_book.processed_at.isoformat() if user_book.processed_at else None,
                    "analysis_version": cache.analysis_version,
                    "quality_diagnostics": (cache.metadata or {}).get("quality_diagnostics", {}),
                    "summary": {
                        "text": full_summary,
                        "summary_to_blocks_overlap": summary_overlap,
                        "flags": summary_flags,
                    },
                    "counts": {
                        "blocks": len(blocks),
                        "themes": len(themes),
                        "subtopics": ThemeSubtopic.objects.filter(theme__global_book=cache).count(),
                        "concept_mentions": len(mentions),
                    },
                    "themes": theme_payloads,
                    "blocks": block_payloads,
                }
            )

        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)

        with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
            fieldnames = [
                "user_email",
                "user_book_id",
                "book_title",
                "block_order",
                "block_title",
                "chapter_title",
                "summary_overlap",
                "front_matter",
                "flags",
                "summary",
                "source_excerpt",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        self.stdout.write(self.style.SUCCESS(f"Meaning audit JSON: {json_path}"))
        self.stdout.write(self.style.SUCCESS(f"Meaning audit CSV: {csv_path}"))
