from unittest.mock import patch
from io import BytesIO

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from reportlab.pdfgen import canvas
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from apps.books.models import (
    Concept,
    ConceptMention,
    GlobalBookCache,
    LogicalBlock,
    UserBook,
)
from apps.books.services.concept_normalizer import find_existing_similar_concept, normalize_concept_name
from apps.books.services.hashing import sha256_bytes
from apps.books.services.logical_block_splitter import split_into_logical_blocks
from apps.books.tasks import analyze_book_task

VALID_FB2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Physics intro</book-title>
      <author><first-name>Ivan</first-name><last-name>Petrov</last-name></author>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Chapter 1</p></title>
      <p>Квантовая физика рассматривает поведение частиц.</p>
      <p>Энергия сохраняется в замкнутой системе.</p>
      <p>Волновые свойства описывают вероятностную природу.</p>
    </section>
  </body>
</FictionBook>
""".encode("utf-8")


def build_test_pdf_bytes() -> bytes:
    stream = BytesIO()
    pdf = canvas.Canvas(stream)
    pdf.setTitle("PDF Physics intro")
    pdf.drawString(72, 800, "Quantum physics studies particle behavior.")
    pdf.drawString(72, 780, "Energy is conserved in a closed system.")
    pdf.drawString(72, 760, "Wave properties describe probabilistic nature.")
    pdf.showPage()
    pdf.drawString(72, 800, "Second page with additional context.")
    pdf.save()
    return stream.getvalue()


VALID_PDF = build_test_pdf_bytes()


class BooksConceptPipelineTests(APITestCase):
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

    def _build_analyzed_book(self):
        book = UserBook.objects.create(
            user=self.user,
            original_filename="book.fb2",
            file_hash=sha256_bytes(VALID_FB2),
            status=UserBook.Status.PROCESSING,
        )
        book.file.save("book.fb2", ContentFile(VALID_FB2), save=True)
        with patch("apps.books.tasks.summarize_logical_block", return_value="Short block summary"), patch(
            "apps.books.tasks.extract_concepts_from_block",
            return_value=[
                {
                    "name": "Квантовая физика",
                    "short_explanation": "Ключевая идея блока.",
                    "source_quote": "Квантовая физика рассматривает поведение частиц.",
                    "importance_score": 0.9,
                }
            ],
        ), patch("apps.books.tasks.summarize_book", return_value="General summary"), patch(
            "apps.books.tasks.save_logical_block_embedding", return_value="block:1"
        ):
            analyze_book_task(book.id, force_reanalyze=True)
        book.refresh_from_db()
        return book

    def test_upload_valid_fb2(self):
        with patch("apps.books.views.analyze_book_task.delay"):
            response = self._upload()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(UserBook.objects.count(), 1)

    def test_upload_valid_pdf(self):
        with patch("apps.books.views.analyze_book_task.delay"):
            response = self._upload(name="book.pdf", content=VALID_PDF)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(UserBook.objects.count(), 1)

    def test_reject_unsupported_extension(self):
        response = self._upload(name="book.txt")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_sha256(self):
        digest = sha256_bytes(VALID_FB2)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, sha256_bytes(VALID_FB2))

    def test_reupload_uses_global_cache(self):
        file_hash = sha256_bytes(VALID_FB2)
        cache = GlobalBookCache.objects.create(
            file_hash=file_hash,
            title="Cached title",
            authors="Cached author",
            metadata={},
            full_summary="Cached summary",
            analysis_version="concept_rag_v1",
        )
        LogicalBlock.objects.create(
            global_book=cache,
            title="Block 1",
            order_number=1,
            source_text="Text",
            short_summary="Short",
            start_paragraph=1,
            end_paragraph=2,
            chapter_title="Chapter",
            token_count=10,
        )
        response = self._upload(content=VALID_FB2)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        book = UserBook.objects.latest("id")
        self.assertEqual(book.status, UserBook.Status.READY)
        self.assertEqual(book.global_cache_id, cache.id)

    def test_split_into_logical_blocks(self):
        parsed = type(
            "Parsed",
            (),
            {
                "chapters": [
                    {
                        "chapter_title": "Chapter",
                        "paragraphs": ["word " * 300, "word " * 350, "word " * 300],
                    }
                ]
            },
        )
        blocks = split_into_logical_blocks(parsed, min_words=200, max_words=500)
        self.assertTrue(len(blocks) >= 2)

    def test_pipeline_creates_summary_and_concepts(self):
        book = self._build_analyzed_book()
        self.assertEqual(book.status, UserBook.Status.READY)
        self.assertTrue(book.global_cache.full_summary)
        self.assertTrue(LogicalBlock.objects.filter(global_book=book.global_cache).exists())
        self.assertTrue(ConceptMention.objects.filter(global_book=book.global_cache).exists())

    def test_concept_normalization(self):
        normalized = normalize_concept_name("Решение задач на движение")
        self.assertEqual(normalized, "решение задача на движение")

    def test_concept_merging(self):
        concept = Concept.objects.create(name="Квантовая физика", normalized_name="квантовый физика", description="")
        found = find_existing_similar_concept("квантовый физика")
        self.assertEqual(found.id, concept.id)

    def test_book_summary_endpoint(self):
        book = self._build_analyzed_book()
        response = self.client.get(f"/api/books/{book.id}/summary/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("full_summary", response.data)

    def test_book_concepts_endpoint(self):
        book = self._build_analyzed_book()
        response = self.client.get(f"/api/books/{book.id}/concepts/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["count"] >= 1)

    def test_all_concepts_endpoint(self):
        self._build_analyzed_book()
        response = self.client.get("/api/concepts/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(len(response.data) >= 1)

    def test_compare_concept_endpoint(self):
        book = self._build_analyzed_book()
        # Duplicate mention in second book to have 2 sources for comparison.
        book2 = UserBook.objects.create(
            user=self.user,
            global_cache=book.global_cache,
            title="Physics copy",
            authors="A",
            original_filename="copy.fb2",
            file_hash=book.file_hash,
            status=UserBook.Status.READY,
        )
        block2 = LogicalBlock.objects.create(
            global_book=book.global_cache,
            title="Block extra",
            order_number=99,
            source_text="Extra concept context",
            short_summary="Extra",
            start_paragraph=10,
            end_paragraph=12,
            chapter_title="Ch2",
            token_count=10,
        )
        concept = Concept.objects.first()
        ConceptMention.objects.get_or_create(
            concept=concept,
            logical_block=block2,
            defaults={
                "global_book": book.global_cache,
                "short_explanation": "Extra explanation",
                "source_quote": "Quote",
                "importance_score": 0.7,
            },
        )
        response = self.client.get(f"/api/concepts/{concept.id}/compare/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("comparison", response.data)
        self.assertTrue(book2.id > 0)

    def test_protect_and_rotation(self):
        response_book = self._upload()
        self.assertEqual(response_book.status_code, status.HTTP_201_CREATED)
        book_id = response_book.data["results"][0]["id"]
        response = self.client.post(f"/api/books/{book_id}/protect/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        cache = GlobalBookCache.objects.create(
            file_hash="a" * 64,
            title="Book",
            authors="A",
            metadata={},
        )
        LogicalBlock.objects.create(
            global_book=cache,
            title="block",
            order_number=1,
            source_text="text",
            short_summary="summary",
            start_paragraph=1,
            end_paragraph=1,
            chapter_title="c",
            token_count=1,
        )
        for index in range(49):
            UserBook.objects.create(
                user=self.user,
                global_cache=cache,
                title=f"book-{index}",
                authors="A",
                original_filename=f"book-{index}.fb2",
                file_hash=f"{index:064d}"[-64:],
                status=UserBook.Status.READY,
                is_protected=index != 0,
            )
        response_limit = self._upload()
        self.assertEqual(response_limit.status_code, status.HTTP_409_CONFLICT)

    def test_export_formats(self):
        book = self._build_analyzed_book()
        for fmt in ("pdf", "txt", "csv", "json"):
            response = self.client.get(f"/api/books/{book.id}/export/?format={fmt}")
            self.assertEqual(response.status_code, status.HTTP_200_OK)

    @override_settings(MAX_FB2_FILE_SIZE=10)
    def test_reject_large_file(self):
        response = self._upload(content=b"x" * 20)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_user_cannot_access_foreign_book(self):
        foreign_book = self._build_analyzed_book()
        token2 = Token.objects.create(user=self.user2)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token2.key}")
        response = self.client.get(f"/api/books/{foreign_book.id}/summary/")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
