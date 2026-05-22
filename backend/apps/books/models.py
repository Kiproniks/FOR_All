from django.conf import settings
from django.db import models
from django.utils import timezone


class GlobalBookCache(models.Model):
    file_hash = models.CharField(max_length=64, unique=True, db_index=True)
    title = models.CharField(max_length=512)
    authors = models.CharField(max_length=512, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    full_summary = models.TextField(blank=True)
    analysis_version = models.CharField(max_length=64, default="concept_rag_v1")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)

    def __str__(self):
        return f"{self.title} ({self.file_hash[:12]})"


class UserBook(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="books")
    global_cache = models.ForeignKey(
        GlobalBookCache,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="user_books",
    )
    title = models.CharField(max_length=512, blank=True)
    authors = models.CharField(max_length=512, blank=True)
    original_filename = models.CharField(max_length=512)
    file_hash = models.CharField(max_length=64, db_index=True)
    file = models.FileField(upload_to="books/%Y/%m/%d/", null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.UPLOADED)
    error_message = models.TextField(blank=True)
    is_protected = models.BooleanField(default=False)
    views_count = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-uploaded_at",)
        indexes = [
            models.Index(fields=("user", "uploaded_at")),
            models.Index(fields=("user", "status")),
        ]

    def mark_failed(self, message: str):
        self.status = UserBook.Status.FAILED
        self.error_message = message[:2000]
        self.processed_at = timezone.now()
        self.save(update_fields=["status", "error_message", "processed_at"])

    def __str__(self):
        return f"{self.user_id}: {self.title or self.original_filename}"


class LogicalBlock(models.Model):
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="logical_blocks")
    title = models.CharField(max_length=512)
    order_number = models.PositiveIntegerField()
    source_text = models.TextField()
    short_summary = models.TextField(blank=True)
    start_paragraph = models.PositiveIntegerField(default=0)
    end_paragraph = models.PositiveIntegerField(default=0)
    chapter_title = models.CharField(max_length=512, blank=True)
    embedding_id = models.CharField(max_length=255, null=True, blank=True)
    token_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("order_number",)
        unique_together = ("global_book", "order_number")
        indexes = [
            models.Index(fields=("global_book", "order_number")),
        ]

    def __str__(self):
        return f"{self.global_book_id}:{self.order_number}:{self.title}"


class Concept(models.Model):
    name = models.CharField(max_length=255)
    normalized_name = models.CharField(max_length=255, unique=True, db_index=True)
    description = models.TextField(blank=True)
    embedding_id = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("name",)

    def __str__(self):
        return self.name


class ConceptMention(models.Model):
    concept = models.ForeignKey(Concept, on_delete=models.CASCADE, related_name="mentions")
    logical_block = models.ForeignKey(LogicalBlock, on_delete=models.CASCADE, related_name="concept_mentions")
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="concept_mentions")
    short_explanation = models.TextField()
    source_quote = models.TextField(blank=True)
    importance_score = models.FloatField(default=0.5)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-importance_score", "id")
        unique_together = ("concept", "logical_block")
        indexes = [
            models.Index(fields=("global_book", "concept")),
        ]

    def __str__(self):
        return f"{self.concept.name} @ {self.logical_block.title}"


class UserConceptEdit(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="concept_edits")
    concept_mention = models.ForeignKey(ConceptMention, on_delete=models.CASCADE, related_name="user_edits")
    custom_explanation = models.TextField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "concept_mention")

    def __str__(self):
        return f"{self.user_id}:{self.concept_mention_id}"


class BookSummary(models.Model):
    global_book = models.OneToOneField(GlobalBookCache, on_delete=models.CASCADE, related_name="summary")
    short_summary = models.TextField(blank=True)
    detailed_summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"summary:{self.global_book_id}"


class BookTheme(models.Model):
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="themes")
    chapter_title = models.CharField(max_length=512, blank=True)
    title = models.CharField(max_length=512)
    order_number = models.PositiveIntegerField()
    start_block_number = models.PositiveIntegerField(default=1)
    end_block_number = models.PositiveIntegerField(default=1)
    start_paragraph = models.PositiveIntegerField(default=0)
    end_paragraph = models.PositiveIntegerField(default=0)
    summary = models.TextField(blank=True)
    embedding_id = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("order_number", "id")
        unique_together = ("global_book", "order_number")
        indexes = [
            models.Index(fields=("global_book", "order_number")),
            models.Index(fields=("global_book", "chapter_title")),
        ]

    def __str__(self):
        return f"{self.global_book_id}:{self.order_number}:{self.title}"


class ThemeSubtopic(models.Model):
    theme = models.ForeignKey(BookTheme, on_delete=models.CASCADE, related_name="subtopics")
    name = models.CharField(max_length=255)
    normalized_name = models.CharField(max_length=255, db_index=True)
    summary = models.TextField(blank=True)
    source_quote = models.TextField(blank=True)
    importance_score = models.FloatField(default=0.5)
    start_paragraph = models.PositiveIntegerField(default=0)
    end_paragraph = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-importance_score", "id")
        unique_together = ("theme", "normalized_name")
        indexes = [
            models.Index(fields=("theme", "importance_score")),
        ]

    def __str__(self):
        return f"{self.theme_id}:{self.name}"


# Legacy glossary models are kept for backward compatibility with earlier migrations.
class TermDefinition(models.Model):
    global_cache = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="terms")
    term = models.CharField(max_length=255)
    normalized_term = models.CharField(max_length=255, db_index=True)
    definition = models.TextField()
    source_chapter = models.CharField(max_length=255, blank=True)
    source_paragraph_index = models.PositiveIntegerField(default=0)
    source_quote = models.TextField(blank=True)
    frequency = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("term",)
        unique_together = ("global_cache", "normalized_term")
        indexes = [
            models.Index(fields=("global_cache", "term")),
            models.Index(fields=("global_cache", "normalized_term")),
        ]

    def __str__(self):
        return self.term


class UserTermEdit(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="term_edits")
    user_book = models.ForeignKey(UserBook, on_delete=models.CASCADE, related_name="term_edits")
    term_definition = models.ForeignKey(TermDefinition, on_delete=models.CASCADE, related_name="user_edits")
    custom_definition = models.TextField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user_book", "term_definition")

    def __str__(self):
        return f"{self.user_id}:{self.term_definition.term}"
