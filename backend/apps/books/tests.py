from __future__ import annotations

import os
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from reportlab.pdfgen import canvas
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from apps.books.models import BookTheme, GlobalBookCache, LogicalBlock, UserBook
from apps.books.services.analysis_quality import evaluate_analysis_quality
from apps.books.services.book_parser import parse_uploaded_book
from apps.books.services.concept_extractor import build_theme_subtopics_from_blocks
from apps.books.services.content_filter import classify_paragraph, filter_content_for_analysis
from apps.books.services.fb2_parser import parse_fb2
from apps.books.services.hashing import sha256_bytes
from apps.books.services.llm_service import summarize_book_representative
from apps.books.services.logical_block_splitter import split_into_logical_blocks_improved
from apps.books.services.pdf_parser import parse_pdf
from apps.books.services.structure_detector import build_canonical_outline
from apps.books.services.theme_hierarchy import build_theme_hierarchy
from apps.books.tasks import PreparedBlock, analyze_book_task

VALID_FB2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Учебник по гидродинамике</book-title>
      <author><first-name>Иван</first-name><last-name>Петров</last-name></author>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Глава 1. Основы</p></title>
      <p>Введение в гидродинамику и основные понятия движения жидкости.</p>
      <p>Закон сохранения массы описывает постоянство расхода в потоке.</p>
      <section>
        <title><p>1.1 Уравнения</p></title>
        <p>Закон сохранения импульса связывает силы и изменение количества движения.</p>
      </section>
    </section>
    <section>
      <title><p>Глава 2. Применение</p></title>
      <p>Рассмотрим примеры расчета для каналов и трубопроводов.</p>
      <p>Докажите, что уравнение применимо для идеальной жидкости.</p>
    </section>
  </body>
</FictionBook>
""".encode("utf-8")


def build_test_pdf_bytes() -> bytes:
    stream = BytesIO()
    pdf = canvas.Canvas(stream)
    pdf.setTitle("Network Methods")
    pdf.drawString(72, 800, "Chapter 1 Introduction")
    pdf.drawString(72, 780, "Computer networks connect systems and protocols.")
    pdf.drawString(72, 760, "TCP provides reliable transport.")
    pdf.showPage()
    pdf.drawString(72, 800, "Chapter 2 Routing")
    pdf.drawString(72, 780, "Routing algorithms select efficient paths.")
    pdf.save()
    return stream.getvalue()


VALID_PDF = build_test_pdf_bytes()


class ParserAndFilterTests(TestCase):
    def test_fb2_parser_extracts_sections_and_titles(self):
        parsed = parse_fb2(VALID_FB2)
        self.assertEqual(parsed.title, "Учебник по гидродинамике")
        self.assertGreaterEqual(len(parsed.chapters), 2)
        titles = [item["chapter_title"] for item in parsed.chapters]
        self.assertIn("Глава 1. Основы", titles)
        self.assertTrue(any("1.1" in title for title in titles))

    def test_pdf_parser_detects_headings_heuristically(self):
        parsed = parse_pdf(VALID_PDF)
        self.assertGreaterEqual(len(parsed.chapters), 2)
        titles = [item["chapter_title"] for item in parsed.chapters]
        self.assertTrue(any("Chapter 1" in title for title in titles))

    def test_content_filter_detects_copyright(self):
        ctype = classify_paragraph("© 2020. Все права защищены. ISBN 978-5-12345")
        self.assertEqual(ctype, "copyright")

    def test_content_filter_detects_exercises(self):
        ctype = classify_paragraph("1. Докажите, что функция непрерывна на отрезке.")
        self.assertIn(ctype, {"exercise", "question"})

    def test_content_filter_detects_code(self):
        ctype = classify_paragraph("#include <stdio.h> int main() { return 0; }")
        self.assertEqual(ctype, "code")

    def test_content_filter_keeps_dialogue(self):
        result = filter_content_for_analysis(
            ["— Я согласен, — сказал герой.", "Он посмотрел в окно."],
            mode="summary",
            allow_dialogue=True,
        )
        kept_types = {row["content_type"] for row in result["kept"]}
        self.assertIn("dialogue", kept_types)

    def test_logical_splitter_does_not_mix_sections(self):
        parsed = parse_fb2(VALID_FB2)
        blocks, _diag = split_into_logical_blocks_improved(parsed, min_words=40, target_words=80, max_words=140)
        self.assertGreaterEqual(len(blocks), 2)
        chapter_titles = {block.chapter_title for block in blocks}
        self.assertIn("Глава 1. Основы", chapter_titles)
        self.assertIn("Глава 2. Применение", chapter_titles)

    def test_structure_detector_finds_main_content_start(self):
        fb2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description><title-info><book-title>Sample</book-title></title-info></description>
  <body>
    <section><title><p>Предисловие</p></title><p>Служебный текст.</p></section>
    <section><title><p>Глава 1. Введение</p></title><p>Основной материал 1.</p></section>
    <section><title><p>1.1 Основные понятия</p></title><p>Основной материал 2.</p></section>
  </body>
</FictionBook>""".encode("utf-8")
        parsed = parse_fb2(fb2)
        outline = build_canonical_outline(parsed)
        self.assertGreaterEqual(outline["main_content_start"], 2)
        self.assertGreaterEqual(outline["main_sections_count"], 2)


class ThemeConceptQualityTests(TestCase):
    def _preview_blocks(self):
        blocks = []
        blocks.append(
            SimpleNamespace(
                id=1,
                order_number=1,
                title="Глава 1. Основы",
                chapter_title="Глава 1. Основы",
                source_text="Закон сохранения массы описывает постоянство расхода в потоке.",
                short_summary="Рассматривается закон сохранения массы и базовые определения.",
                start_paragraph=1,
                end_paragraph=2,
                semantic_data={
                    "clean_text_for_analysis": "Закон сохранения массы описывает постоянство расхода в потоке.",
                    "section_title": "Глава 1. Основы",
                    "section_path": [{"level": 1, "title": "Глава 1. Основы"}],
                    "paragraph_records": [
                        {"text": "Закон сохранения массы описывает постоянство расхода в потоке.", "content_type": "main_text"}
                    ],
                },
            )
        )
        blocks.append(
            SimpleNamespace(
                id=2,
                order_number=2,
                title="Глава 2. Применение",
                chapter_title="Глава 2. Применение",
                source_text="Рассмотрим примеры расчета для каналов и трубопроводов.",
                short_summary="Показаны практические примеры расчетов.",
                start_paragraph=3,
                end_paragraph=4,
                semantic_data={
                    "clean_text_for_analysis": "Рассмотрим примеры расчета для каналов и трубопроводов.",
                    "section_title": "Глава 2. Применение",
                    "section_path": [{"level": 1, "title": "Глава 2. Применение"}],
                    "paragraph_records": [
                        {"text": "Рассмотрим примеры расчета для каналов и трубопроводов.", "content_type": "main_text"}
                    ],
                },
            )
        )
        return blocks

    def test_summary_representative_has_no_copyright_noise(self):
        summary = summarize_book_representative(
            section_titles=["Глава 1. Основы", "Глава 2. Применение"],
            block_summaries=[
                "Основные понятия движения жидкости.",
                "Методы расчета трубопроводов.",
            ],
            top_concepts=["закон сохранения массы", "расход", "трубопровод"],
        )
        self.assertNotIn("ISBN", summary)
        self.assertNotIn("Все права защищены", summary)

    def test_theme_titles_not_generic_part_names(self):
        themes = build_theme_hierarchy(self._preview_blocks())
        self.assertGreaterEqual(len(themes), 1)
        self.assertTrue(all("Без названия" not in item.title for item in themes))
        self.assertTrue(all(not item.title.lower().startswith("часть ") for item in themes))

    def test_subtopics_do_not_contain_stoplist_garbage(self):
        blocks = self._preview_blocks()
        subtopics = build_theme_subtopics_from_blocks("Глава 1", blocks, max_items=8)
        names = {item["name"].lower() for item in subtopics}
        self.assertNotIn("такой образ", names)
        self.assertNotIn("данный случай", names)

    def test_analysis_quality_reports_problems_for_bad_result(self):
        bad_block = SimpleNamespace(
            order_number=1,
            title="Блок",
            chapter_title="Без названия главы",
            source_text="© 2020. Все права защищены. ISBN 978",
            semantic_data={
                "front_matter": True,
                "paragraph_records": [
                    {"text": "© 2020. Все права защищены. ISBN 978", "content_type": "copyright"}
                ]
            },
        )
        bad_theme = SimpleNamespace(title="Часть 1")
        bad_subtopic = SimpleNamespace(name="данный случай")
        bad_mention = SimpleNamespace(source_quote="int main() { return 0; }", logical_block=bad_block)

        diagnostics = evaluate_analysis_quality(
            summary_text="ISBN 978-5-12345 Все права защищены",
            blocks=[bad_block],
            themes=[bad_theme],
            subtopics=[bad_subtopic],
            concept_mentions=[bad_mention],
            parser_metadata={"sections_count": 12},
            content_filter_stats={"total_paragraphs": 100, "removed_count": 80, "kept_count": 20},
        )
        self.assertLess(diagnostics["quality_score"], 0.7)
        self.assertIn("summary_contains_copyright", diagnostics["problems"])
        self.assertIn("front_matter_dominates_blocks", diagnostics["problems"])

    def test_small_book_produces_at_least_one_theme(self):
        themes = build_theme_hierarchy(self._preview_blocks()[:1])
        self.assertGreaterEqual(len(themes), 1)

    def test_structured_textbook_themes_follow_chapters(self):
        parsed = parse_fb2(VALID_FB2)
        blocks, _ = split_into_logical_blocks_improved(parsed, min_words=40, target_words=80, max_words=160)
        preview = [
            SimpleNamespace(
                id=i + 1,
                order_number=block.order_number,
                title=block.title,
                chapter_title=block.chapter_title,
                source_text=block.source_text,
                short_summary=block.clean_text_for_analysis[:220] or block.source_text[:220],
                start_paragraph=block.start_paragraph,
                end_paragraph=block.end_paragraph,
                semantic_data={
                    "clean_text_for_analysis": block.clean_text_for_analysis,
                    "section_title": block.section_title,
                    "section_path": block.section_path,
                    "paragraph_records": block.paragraph_records or [],
                },
            )
            for i, block in enumerate(blocks)
        ]
        themes = build_theme_hierarchy(preview)
        theme_titles = [item.title for item in themes]
        self.assertTrue(any("Глава 1" in title for title in theme_titles))
        self.assertTrue(any("Глава 2" in title for title in theme_titles))


class ApiAndFallbackTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(email="user@example.com", password="StrongPass123")
        self.user2 = user_model.objects.create_user(email="user2@example.com", password="StrongPass123")
        token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")

    def _upload(self, *, name="book.fb2", content=VALID_FB2, extra_data=None):
        content_type = "application/pdf" if name.lower().endswith(".pdf") else "application/xml"
        data = {"files": [SimpleUploadedFile(name, content, content_type=content_type)]}
        if extra_data:
            data.update(extra_data)
        return self.client.post("/api/books/upload/", data, format="multipart")

    def test_upload_valid_fb2(self):
        with patch("apps.books.views.analyze_book_task.delay"):
            response = self._upload(extra_data={"analysis_mode": "classic_improved"})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(UserBook.objects.count(), 1)

    @patch("apps.books.views.ensure_llm_ready", return_value={"ok": False, "error": "LLM disabled"})
    def test_llm_full_requires_llm_enabled(self, _llm_ready):
        response = self._upload(extra_data={"analysis_mode": "llm_full"})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("LLM is required", response.data["detail"])

    @override_settings(MAX_FB2_FILE_SIZE=10)
    def test_reject_large_file(self):
        response = self._upload(content=b"x" * 20)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_sha256(self):
        digest = sha256_bytes(VALID_FB2)
        self.assertEqual(digest, sha256_bytes(VALID_FB2))
        self.assertEqual(len(digest), 64)

    def test_reupload_uses_cache(self):
        file_hash = sha256_bytes(VALID_FB2)
        cache = GlobalBookCache.objects.create(file_hash=file_hash, title="Cached", authors="A", metadata={})
        block = LogicalBlock.objects.create(
            global_book=cache,
            title="Block",
            order_number=1,
            source_text="Text",
            short_summary="Summary",
            start_paragraph=1,
            end_paragraph=1,
            chapter_title="Глава",
            token_count=5,
        )
        BookTheme.objects.create(
            global_book=cache,
            chapter_title="Глава",
            title="Theme",
            order_number=1,
            start_block_number=block.order_number,
            end_block_number=block.order_number,
            start_paragraph=1,
            end_paragraph=1,
            summary="Theme summary",
        )

        response = self._upload(content=VALID_FB2)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        book = UserBook.objects.latest("id")
        self.assertEqual(book.status, UserBook.Status.READY)
        self.assertEqual(book.global_cache_id, cache.id)

    def test_user_cannot_access_foreign_book(self):
        cache = GlobalBookCache.objects.create(file_hash="b" * 64, title="T", authors="", metadata={})
        book = UserBook.objects.create(
            user=self.user,
            global_cache=cache,
            title="Book",
            authors="",
            original_filename="book.fb2",
            file_hash="b" * 64,
            status=UserBook.Status.READY,
        )
        token2 = Token.objects.create(user=self.user2)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token2.key}")
        response = self.client.get(f"/api/books/{book.id}/summary/")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    @patch("apps.books.tasks._prepare_semantic_fast_blocks", side_effect=ValueError("semantic down"))
    @patch("apps.books.tasks._prepare_classic_improved_blocks")
    @patch("apps.books.tasks.save_logical_block_embedding", return_value="block:1")
    @patch("apps.books.tasks.summarize_book_representative", return_value="Содержательный конспект")
    def test_fallback_works_if_llm_unavailable(
        self,
        _summary_mock,
        _embed_mock,
        classic_improved_mock,
        _semantic_mock,
    ):
        user = get_user_model().objects.create_user(email="t@example.com", password="pass12345")
        book = UserBook.objects.create(
            user=user,
            original_filename="book.fb2",
            file_hash=sha256_bytes(VALID_FB2),
            status=UserBook.Status.PROCESSING,
        )
        book.file.save("book.fb2", ContentFile(VALID_FB2), save=True)

        classic_improved_mock.return_value = (
            [
                PreparedBlock(
                    title="Improved block",
                    order_number=1,
                    source_text="Основной содержательный текст.",
                    short_summary="Краткое содержание блока.",
                    chapter_title="Глава 1",
                    start_paragraph=1,
                    end_paragraph=2,
                    token_count=6,
                    source_sentence_ids=[],
                    concept_candidates=[],
                    thought_cluster_ids=[],
                    semantic_data={"pipeline": "classic_improved", "clean_text_for_analysis": "Основной текст"},
                )
            ],
            {},
            {"pipeline": "classic_improved"},
        )

        analyze_book_task(book.id, force_reanalyze=True, analysis_mode="hybrid")
        book.refresh_from_db()
        self.assertEqual(book.status, UserBook.Status.READY)
        self.assertIn(book.global_cache.metadata.get("pipeline_used"), {"classic_improved", "classic"})
