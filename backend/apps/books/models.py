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
        QUEUED = "queued", "Queued"
        PROCESSING = "processing", "Processing"
        PARSING = "parsing", "Parsing"
        STRUCTURE_DETECTION = "structure_detection", "Structure Detection"
        FILTERING = "filtering", "Filtering"
        CHUNKING = "chunking", "Chunking"
        LLM_SECTION_ANALYSIS = "llm_section_analysis", "LLM Section Analysis"
        LLM_CHAPTER_ANALYSIS = "llm_chapter_analysis", "LLM Chapter Analysis"
        LLM_BOOK_ANALYSIS = "llm_book_analysis", "LLM Book Analysis"
        LLM_FAST_BATCHED_SECTION_ANALYSIS = (
            "llm_fast_batched_section_analysis",
            "LLM Fast Batched Section Analysis",
        )
        LLM_FAST_BATCHED_CHAPTER_ANALYSIS = (
            "llm_fast_batched_chapter_analysis",
            "LLM Fast Batched Chapter Analysis",
        )
        LLM_FAST_BATCHED_BOOK_ANALYSIS = (
            "llm_fast_batched_book_analysis",
            "LLM Fast Batched Book Analysis",
        )
        BUILDING_MAP = "building_map", "Building Map"
        SAVING_RESULTS = "saving_results", "Saving Results"
        READY = "ready", "Ready"
        READY_WITH_WARNINGS = "ready_with_warnings", "Ready with Warnings"
        PARTIAL_READY = "partial_ready", "Partial Ready"
        FAILED = "failed", "Failed"
        FAILED_TIMEOUT = "failed_timeout", "Failed Timeout"
        CANCELLED = "cancelled", "Cancelled"
        DEBUG_PREVIEW = "debug_preview", "Debug Preview"
        HEURISTIC_PREVIEW = "heuristic_preview", "Heuristic Preview"

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
    status = models.CharField(max_length=64, choices=Status.choices, default=Status.UPLOADED)
    current_stage = models.CharField(max_length=64, blank=True, default="uploaded")
    progress_percent = models.PositiveSmallIntegerField(default=0)
    error_message = models.TextField(blank=True)
    is_protected = models.BooleanField(default=False)
    views_count = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    last_heartbeat_at = models.DateTimeField(null=True, blank=True)
    llm_provider_used = models.CharField(max_length=64, blank=True, default="")
    llm_model_used = models.CharField(max_length=128, blank=True, default="")
    llm_calls_total = models.PositiveIntegerField(default=0)
    llm_failures_total = models.PositiveIntegerField(default=0)
    fallback_used_count = models.PositiveIntegerField(default=0)
    analysis_mode = models.CharField(max_length=32, blank=True, default="")
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-uploaded_at",)
        indexes = [
            models.Index(fields=("user", "uploaded_at")),
            models.Index(fields=("user", "status")),
        ]

    def mark_failed(self, message: str):
        self.status = UserBook.Status.FAILED
        self.current_stage = self.current_stage or "failed"
        self.progress_percent = min(self.progress_percent, 99)
        self.error_message = message[:2000]
        self.finished_at = timezone.now()
        self.last_heartbeat_at = timezone.now()
        self.processed_at = timezone.now()
        self.save(
            update_fields=[
                "status",
                "current_stage",
                "progress_percent",
                "error_message",
                "finished_at",
                "last_heartbeat_at",
                "processed_at",
                "updated_at",
            ]
        )

    def bump_heartbeat(self):
        self.last_heartbeat_at = timezone.now()
        self.save(update_fields=["last_heartbeat_at", "updated_at"])

    def __str__(self):
        return f"{self.user_id}: {self.title or self.original_filename}"


class LLMAnalysisRun(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        READY = "ready", "Ready"
        PARTIAL_READY = "partial_ready", "Partial Ready"
        FAILED = "failed", "Failed"
        FAILED_TIMEOUT = "failed_timeout", "Failed Timeout"
        DRY_RUN = "dry_run", "Dry Run"

    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="llm_analysis_runs")
    user_book = models.ForeignKey(
        UserBook,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="llm_analysis_runs",
    )
    analysis_run_id = models.CharField(max_length=64, unique=True, db_index=True)
    mode = models.CharField(max_length=64, default="llm_fast_batched")
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.QUEUED)
    current_batch_index = models.PositiveIntegerField(default=0)
    current_offset = models.PositiveIntegerField(default=0)
    batch_size = models.PositiveIntegerField(default=10)
    sections_total = models.PositiveIntegerField(default=0)
    sections_processed = models.PositiveIntegerField(default=0)
    chapters_processed = models.PositiveIntegerField(default=0)
    llm_calls_actual = models.PositiveIntegerField(default=0)
    cache_hits = models.PositiveIntegerField(default=0)
    fallback_count = models.PositiveIntegerField(default=0)
    timeout_count = models.PositiveIntegerField(default=0)
    valid_json_units = models.PositiveIntegerField(default=0)
    expected_json_units = models.PositiveIntegerField(default=0)
    progress_percent = models.PositiveSmallIntegerField(default=0)
    output_dir = models.CharField(max_length=1024, blank=True)
    final_report = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    last_heartbeat_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [
            models.Index(fields=("global_book", "mode", "status")),
            models.Index(fields=("user_book", "status")),
            models.Index(fields=("last_heartbeat_at",)),
        ]

    def __str__(self):
        return f"{self.mode}:{self.global_book_id}:{self.status}:{self.analysis_run_id}"


class LLMSectionAnalysis(models.Model):
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="llm_section_analyses")
    analysis_run = models.ForeignKey(
        LLMAnalysisRun,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="section_results",
    )
    mode = models.CharField(max_length=64, default="llm_fast_batched")
    chapter_title = models.CharField(max_length=512, blank=True)
    section_title = models.CharField(max_length=512)
    section_index = models.PositiveIntegerField()
    start_paragraph = models.PositiveIntegerField(default=0)
    end_paragraph = models.PositiveIntegerField(default=0)
    word_count = models.PositiveIntegerField(default=0)
    summary = models.TextField(blank=True)
    terms = models.JSONField(default=list, blank=True)
    subtopics = models.JSONField(default=list, blank=True)
    model_used = models.CharField(max_length=128, blank=True)
    prompt_version = models.CharField(max_length=64, default="v1")
    content_hash = models.CharField(max_length=64, db_index=True)
    json_valid = models.BooleanField(default=False)
    fallback_used = models.BooleanField(default=False)
    timeout = models.BooleanField(default=False)
    quality_flags = models.JSONField(default=list, blank=True)
    cache_hit = models.BooleanField(default=False)
    actual_llm_call = models.BooleanField(default=False)
    duration_seconds = models.FloatField(default=0.0)
    input_chars = models.PositiveIntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("section_index",)
        constraints = [
            models.UniqueConstraint(
                fields=("global_book", "mode", "content_hash", "prompt_version", "model_used"),
                name="uniq_llm_section_content_model",
            )
        ]
        indexes = [
            models.Index(fields=("global_book", "section_index")),
            models.Index(fields=("global_book", "mode", "json_valid")),
        ]

    def __str__(self):
        return f"{self.global_book_id}:{self.section_index}:{self.section_title}"


class LLMChapterAnalysis(models.Model):
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="llm_chapter_analyses")
    analysis_run = models.ForeignKey(
        LLMAnalysisRun,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chapter_results",
    )
    mode = models.CharField(max_length=64, default="llm_fast_batched")
    chapter_title = models.CharField(max_length=512)
    chapter_index = models.PositiveIntegerField(default=0)
    chapter_summary = models.TextField(blank=True)
    main_topics = models.JSONField(default=list, blank=True)
    key_terms = models.JSONField(default=list, blank=True)
    sections_count = models.PositiveIntegerField(default=0)
    source_section_ids = models.JSONField(default=list, blank=True)
    model_used = models.CharField(max_length=128, blank=True)
    prompt_version = models.CharField(max_length=64, default="v1")
    content_hash = models.CharField(max_length=64, db_index=True)
    json_valid = models.BooleanField(default=False)
    fallback_used = models.BooleanField(default=False)
    timeout = models.BooleanField(default=False)
    quality_flags = models.JSONField(default=list, blank=True)
    cache_hit = models.BooleanField(default=False)
    actual_llm_call = models.BooleanField(default=False)
    duration_seconds = models.FloatField(default=0.0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("chapter_index", "id")
        constraints = [
            models.UniqueConstraint(
                fields=("global_book", "mode", "content_hash", "prompt_version", "model_used"),
                name="uniq_llm_chapter_content_model",
            )
        ]
        indexes = [
            models.Index(fields=("global_book", "chapter_index")),
        ]

    def __str__(self):
        return f"{self.global_book_id}:{self.chapter_index}:{self.chapter_title}"


class LLMBookAnalysis(models.Model):
    global_book = models.OneToOneField(GlobalBookCache, on_delete=models.CASCADE, related_name="llm_book_analysis")
    analysis_run = models.ForeignKey(
        LLMAnalysisRun,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="book_results",
    )
    mode = models.CharField(max_length=64, default="llm_fast_batched")
    book_summary = models.TextField(blank=True)
    global_themes = models.JSONField(default=list, blank=True)
    learning_path = models.JSONField(default=list, blank=True)
    model_used = models.CharField(max_length=128, blank=True)
    prompt_version = models.CharField(max_length=64, default="v1")
    chapters_count = models.PositiveIntegerField(default=0)
    sections_count = models.PositiveIntegerField(default=0)
    content_hash = models.CharField(max_length=64, db_index=True)
    json_valid = models.BooleanField(default=False)
    fallback_used = models.BooleanField(default=False)
    timeout = models.BooleanField(default=False)
    quality_flags = models.JSONField(default=list, blank=True)
    cache_hit = models.BooleanField(default=False)
    actual_llm_call = models.BooleanField(default=False)
    duration_seconds = models.FloatField(default=0.0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.global_book_id}:{self.mode}:{self.json_valid}"


class BookSentence(models.Model):
    book = models.ForeignKey(UserBook, on_delete=models.CASCADE, related_name="sentences")
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="sentences")
    index = models.PositiveIntegerField()
    text = models.TextField()
    source_start = models.PositiveIntegerField(null=True, blank=True)
    source_end = models.PositiveIntegerField(null=True, blank=True)
    chapter_title = models.CharField(max_length=512, blank=True)
    section_title = models.CharField(max_length=512, blank=True)
    paragraph_index = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("index",)
        unique_together = ("global_book", "index")
        indexes = [
            models.Index(fields=("book", "index")),
            models.Index(fields=("global_book", "chapter_title")),
        ]

    def __str__(self):
        return f"{self.global_book_id}:s{self.index}"


class SentenceThought(models.Model):
    sentence = models.OneToOneField(BookSentence, on_delete=models.CASCADE, related_name="thought")
    book = models.ForeignKey(UserBook, on_delete=models.CASCADE, related_name="sentence_thoughts")
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="sentence_thoughts")
    index = models.PositiveIntegerField()
    thought_text = models.TextField()
    normalized_thought = models.TextField(blank=True)
    terms = models.JSONField(default=list, blank=True)
    is_meaningful = models.BooleanField(default=True)
    noise = models.BooleanField(default=False)
    skip_reason = models.TextField(blank=True)
    quality_flags = models.JSONField(default=list, blank=True)
    llm_raw_response = models.JSONField(default=dict, blank=True)
    json_valid = models.BooleanField(default=True)
    fallback_used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("index",)
        unique_together = ("global_book", "index")
        indexes = [
            models.Index(fields=("book", "is_meaningful")),
            models.Index(fields=("book", "noise")),
            models.Index(fields=("global_book", "index")),
        ]

    def __str__(self):
        return f"{self.global_book_id}:t{self.index}"


class SequentialThoughtGroup(models.Model):
    book = models.ForeignKey(UserBook, on_delete=models.CASCADE, related_name="sequential_thought_groups")
    global_book = models.ForeignKey(GlobalBookCache, on_delete=models.CASCADE, related_name="sequential_thought_groups")
    index = models.PositiveIntegerField()
    start_sentence_index = models.PositiveIntegerField()
    end_sentence_index = models.PositiveIntegerField()
    main_thought = models.TextField()
    sentence_indexes = models.JSONField(default=list)
    thought_ids = models.JSONField(default=list)
    llm_raw_response = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("index",)
        unique_together = ("global_book", "index")
        indexes = [
            models.Index(fields=("book", "index")),
            models.Index(fields=("global_book", "start_sentence_index")),
        ]

    def __str__(self):
        return f"{self.global_book_id}:group{self.index}"


class ThoughtRelation(models.Model):
    RELATION_SAME = "same"
    RELATION_RELATED = "related"
    RELATION_DIFFERENT = "different"
    RELATION_CHOICES = (
        (RELATION_SAME, "Same"),
        (RELATION_RELATED, "Related"),
        (RELATION_DIFFERENT, "Different"),
    )

    source_thought = models.ForeignKey(SentenceThought, on_delete=models.CASCADE, related_name="outgoing_relations")
    target_thought = models.ForeignKey(SentenceThought, on_delete=models.CASCADE, related_name="incoming_relations")
    relation = models.CharField(max_length=32, choices=RELATION_CHOICES)
    score = models.FloatField(default=0.0)
    explanation = models.TextField(blank=True)
    llm_raw_response = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("source_thought", "target_thought")
        indexes = [
            models.Index(fields=("relation", "score")),
            models.Index(fields=("source_thought", "target_thought")),
        ]

    def __str__(self):
        return f"{self.source_thought_id}->{self.target_thought_id}:{self.relation}:{self.score:.2f}"


class GlobalLogicalThoughtBlock(models.Model):
    title = models.CharField(max_length=512)
    main_idea = models.TextField()
    summary = models.TextField(blank=True)
    keywords = models.JSONField(default=list, blank=True)
    is_merged = models.BooleanField(default=False)
    merged_into = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="merged_sources",
    )
    source_books = models.ManyToManyField(UserBook, related_name="global_thought_blocks", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("title", "id")

    def __str__(self):
        return self.title


class ThoughtBlockMembership(models.Model):
    thought = models.ForeignKey(SentenceThought, on_delete=models.CASCADE, related_name="block_memberships")
    block = models.ForeignKey(GlobalLogicalThoughtBlock, on_delete=models.CASCADE, related_name="memberships")
    relevance_score = models.FloatField(default=0.0)
    reason = models.TextField(blank=True)
    llm_raw_response = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("thought", "block")
        indexes = [
            models.Index(fields=("block", "relevance_score")),
            models.Index(fields=("thought",)),
        ]

    def __str__(self):
        return f"{self.thought_id}->{self.block_id}:{self.relevance_score:.2f}"


class ThoughtChainAnalysisRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        READY = "ready", "Ready"
        PARTIAL_READY = "partial_ready", "Partial Ready"
        FAILED = "failed", "Failed"
        DRY_RUN = "dry_run", "Dry Run"

    book = models.ForeignKey(UserBook, on_delete=models.CASCADE, related_name="thought_chain_runs")
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.PENDING)
    model_name = models.CharField(max_length=100, blank=True)
    total_sentences = models.PositiveIntegerField(default=0)
    processed_sentences = models.PositiveIntegerField(default=0)
    total_thoughts = models.PositiveIntegerField(default=0)
    total_relations_checked = models.PositiveIntegerField(default=0)
    total_relations_created = models.PositiveIntegerField(default=0)
    total_blocks_created = models.PositiveIntegerField(default=0)
    checkpoint = models.JSONField(default=dict, blank=True)
    report = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [
            models.Index(fields=("book", "status")),
            models.Index(fields=("created_at",)),
        ]

    def __str__(self):
        return f"thought_chain:{self.book_id}:{self.status}:{self.id}"


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
    semantic_data = models.JSONField(null=True, blank=True)
    source_sentence_ids = models.JSONField(null=True, blank=True)
    concept_candidates = models.JSONField(null=True, blank=True)
    thought_cluster_ids = models.JSONField(null=True, blank=True)
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


class BookStudyNotes(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        GENERATING = "generating", "Generating"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    book = models.OneToOneField(UserBook, on_delete=models.CASCADE, related_name="study_notes")
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.PENDING)
    content_markdown = models.TextField(blank=True)
    model_name = models.CharField(max_length=100, blank=True)
    source_run = models.ForeignKey(LLMAnalysisRun, null=True, blank=True, on_delete=models.SET_NULL)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=("book", "status")),
        ]

    def __str__(self):
        return f"notes:{self.book_id}:{self.status}"


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
