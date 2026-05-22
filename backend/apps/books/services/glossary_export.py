from __future__ import annotations

import csv
import json
from io import BytesIO, StringIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from apps.books.models import ConceptMention, LogicalBlock, UserBook, UserConceptEdit


def build_export_payload(user_book: UserBook) -> dict:
    if not user_book.global_cache:
        return {
            "book": {},
            "summary": "",
            "blocks": [],
            "concepts": [],
        }

    global_book = user_book.global_cache
    blocks = list(
        LogicalBlock.objects.filter(global_book=global_book)
        .order_by("order_number")
    )
    mentions = list(
        ConceptMention.objects.filter(global_book=global_book)
        .select_related("concept", "logical_block")
        .order_by("logical_block__order_number", "-importance_score")
    )
    edits = {
        edit.concept_mention_id: edit.custom_explanation
        for edit in UserConceptEdit.objects.filter(
            user=user_book.user,
            concept_mention__global_book=global_book,
        )
    }

    concepts = []
    for mention in mentions:
        concepts.append(
            {
                "book_title": user_book.title,
                "block_title": mention.logical_block.title,
                "concept_name": mention.concept.name,
                "short_explanation": edits.get(mention.id, mention.short_explanation),
                "source_quote": mention.source_quote,
                "chapter_title": mention.logical_block.chapter_title,
            }
        )

    block_items = []
    for block in blocks:
        block_mentions = [item for item in concepts if item["block_title"] == block.title]
        block_items.append(
            {
                "id": block.id,
                "title": block.title,
                "chapter_title": block.chapter_title,
                "short_summary": block.short_summary,
                "source_text": block.source_text,
                "concepts": block_mentions,
            }
        )

    return {
        "book": {
            "id": user_book.id,
            "title": user_book.title,
            "authors": user_book.authors,
            "original_filename": user_book.original_filename,
        },
        "summary": global_book.full_summary or "",
        "blocks": block_items,
        "concepts": concepts,
    }


def export_csv(user_book: UserBook) -> bytes:
    payload = build_export_payload(user_book)
    rows = payload["concepts"]
    sio = StringIO()
    writer = csv.writer(sio)
    writer.writerow(["book_title", "block_title", "concept_name", "short_explanation", "source_quote"])
    for row in rows:
        writer.writerow(
            [
                row["book_title"],
                row["block_title"],
                row["concept_name"],
                row["short_explanation"],
                row["source_quote"],
            ]
        )
    return sio.getvalue().encode("utf-8-sig")


def export_txt(user_book: UserBook) -> bytes:
    payload = build_export_payload(user_book)
    lines = [
        f"Book: {payload['book'].get('title', '')}",
        f"Authors: {payload['book'].get('authors', '') or '-'}",
        "",
        "Summary:",
        payload["summary"] or "-",
        "",
        "Logical blocks:",
    ]
    for idx, block in enumerate(payload["blocks"], start=1):
        lines.extend(
            [
                f"{idx}. {block['title']}",
                f"   Chapter: {block['chapter_title'] or '-'}",
                f"   Short summary: {block['short_summary'] or '-'}",
                "",
            ]
        )
    lines.append("Concepts:")
    for idx, concept in enumerate(payload["concepts"], start=1):
        lines.extend(
            [
                f"{idx}. {concept['concept_name']}",
                f"   Block: {concept['block_title']}",
                f"   Explanation: {concept['short_explanation']}",
                f"   Quote: {concept['source_quote']}",
                "",
            ]
        )
    return "\n".join(lines).encode("utf-8")


def export_pdf(user_book: UserBook) -> bytes:
    payload = build_export_payload(user_book)
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"<b>{payload['book'].get('title', '')}</b>", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Authors: {payload['book'].get('authors', '') or '-'}", styles["Normal"]),
        Spacer(1, 8),
        Paragraph("<b>Summary</b>", styles["Heading3"]),
        Paragraph(payload["summary"] or "-", styles["BodyText"]),
        Spacer(1, 12),
    ]

    table_data = [["Block", "Concept", "Explanation", "Quote"]]
    for row in payload["concepts"]:
        table_data.append(
            [
                Paragraph(row["block_title"], styles["BodyText"]),
                Paragraph(row["concept_name"], styles["BodyText"]),
                Paragraph(row["short_explanation"], styles["BodyText"]),
                Paragraph(row["source_quote"][:500], styles["BodyText"]),
            ]
        )

    if len(table_data) == 1:
        table_data.append(["-", "-", "-", "-"])

    table = Table(table_data, colWidths=[35 * mm, 35 * mm, 65 * mm, 45 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    content.append(table)
    document.build(content)
    return buffer.getvalue()


def export_json(user_book: UserBook) -> bytes:
    payload = build_export_payload(user_book)
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
