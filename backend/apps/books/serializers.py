from rest_framework import serializers

from apps.books.models import (
    Concept,
    ConceptMention,
    LLMAnalysisRun,
    LogicalBlock,
    UserBook,
    UserConceptEdit,
)


class UserBookSerializer(serializers.ModelSerializer):
    logical_blocks_count = serializers.SerializerMethodField()
    concepts_count = serializers.SerializerMethodField()
    thought_sentences_count = serializers.SerializerMethodField()
    sentence_thoughts_count = serializers.SerializerMethodField()
    sequential_groups_count = serializers.SerializerMethodField()
    thought_relations_count = serializers.SerializerMethodField()
    global_thought_blocks_count = serializers.SerializerMethodField()
    current_batch_index = serializers.SerializerMethodField()
    sections_processed = serializers.SerializerMethodField()
    sections_total = serializers.SerializerMethodField()

    class Meta:
        model = UserBook
        fields = (
            "id",
            "title",
            "authors",
            "status",
            "current_stage",
            "progress_percent",
            "analysis_mode",
            "is_protected",
            "uploaded_at",
            "started_at",
            "updated_at",
            "finished_at",
            "last_heartbeat_at",
            "processed_at",
            "views_count",
            "llm_provider_used",
            "llm_model_used",
            "llm_calls_total",
            "llm_failures_total",
            "fallback_used_count",
            "current_batch_index",
            "sections_processed",
            "sections_total",
            "logical_blocks_count",
            "concepts_count",
            "thought_sentences_count",
            "sentence_thoughts_count",
            "sequential_groups_count",
            "thought_relations_count",
            "global_thought_blocks_count",
        )

    def get_logical_blocks_count(self, obj):
        if not obj.global_cache_id:
            return 0
        return obj.global_cache.logical_blocks.count()

    def get_concepts_count(self, obj):
        if not obj.global_cache_id:
            return 0
        return obj.global_cache.concept_mentions.count()

    def get_thought_sentences_count(self, obj):
        if not obj.global_cache_id:
            return 0
        return obj.global_cache.sentences.count()

    def get_sentence_thoughts_count(self, obj):
        if not obj.global_cache_id:
            return 0
        return obj.global_cache.sentence_thoughts.count()

    def get_sequential_groups_count(self, obj):
        if not obj.global_cache_id:
            return 0
        return obj.global_cache.sequential_thought_groups.count()

    def get_thought_relations_count(self, obj):
        if not obj.global_cache_id:
            return 0
        from apps.books.models import ThoughtRelation

        return ThoughtRelation.objects.filter(source_thought__global_book_id=obj.global_cache_id).count()

    def get_global_thought_blocks_count(self, obj):
        from apps.books.models import GlobalLogicalThoughtBlock

        return GlobalLogicalThoughtBlock.objects.filter(source_books=obj).distinct().count()

    def _latest_run(self, obj):
        if not obj.global_cache_id:
            return None
        cached: dict[int, LLMAnalysisRun] | None = self.context.get("latest_runs")
        if cached is not None:
            return cached.get(obj.id)
        return obj.llm_analysis_runs.order_by("-created_at").first()

    def get_current_batch_index(self, obj):
        run = self._latest_run(obj)
        return run.current_batch_index if run else None

    def get_sections_processed(self, obj):
        run = self._latest_run(obj)
        return run.sections_processed if run else None

    def get_sections_total(self, obj):
        run = self._latest_run(obj)
        return run.sections_total if run else None


class LogicalBlockSerializer(serializers.ModelSerializer):
    concepts_count = serializers.SerializerMethodField()

    class Meta:
        model = LogicalBlock
        fields = (
            "id",
            "title",
            "order_number",
            "chapter_title",
            "short_summary",
            "start_paragraph",
            "end_paragraph",
            "concepts_count",
        )

    def get_concepts_count(self, obj):
        return obj.concept_mentions.count()


class ConceptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Concept
        fields = ("id", "name", "normalized_name", "description")


class ConceptMentionSerializer(serializers.ModelSerializer):
    concept = ConceptSerializer(read_only=True)
    logical_block = serializers.PrimaryKeyRelatedField(read_only=True)
    custom_explanation = serializers.SerializerMethodField()
    book_id = serializers.SerializerMethodField()
    book_title = serializers.SerializerMethodField()
    chapter_title = serializers.SerializerMethodField()

    class Meta:
        model = ConceptMention
        fields = (
            "id",
            "concept",
            "logical_block",
            "short_explanation",
            "custom_explanation",
            "source_quote",
            "importance_score",
            "book_id",
            "book_title",
            "chapter_title",
        )

    def get_custom_explanation(self, obj):
        edits_map: dict[int, UserConceptEdit] = self.context.get("edits_map", {})
        edit = edits_map.get(obj.id)
        return edit.custom_explanation if edit else None

    def get_book_title(self, obj):
        return obj.global_book.title

    def get_book_id(self, obj):
        book_map: dict[int, int] = self.context.get("book_map", {})
        return book_map.get(obj.global_book_id)

    def get_chapter_title(self, obj):
        return obj.logical_block.chapter_title


class LogicalBlockDetailSerializer(serializers.ModelSerializer):
    concepts = serializers.SerializerMethodField()

    class Meta:
        model = LogicalBlock
        fields = (
            "id",
            "title",
            "order_number",
            "chapter_title",
            "short_summary",
            "source_text",
            "source_sentence_ids",
            "concept_candidates",
            "thought_cluster_ids",
            "semantic_data",
            "concepts",
        )

    def get_concepts(self, obj):
        edits_map = self.context.get("edits_map", {})
        book_map = self.context.get("book_map", {})
        mentions = obj.concept_mentions.select_related("concept", "global_book").order_by("-importance_score")
        return ConceptMentionSerializer(mentions, many=True, context={"edits_map": edits_map, "book_map": book_map}).data


class BookSummarySerializer(serializers.Serializer):
    book_id = serializers.IntegerField()
    title = serializers.CharField()
    authors = serializers.CharField(allow_blank=True)
    full_summary = serializers.CharField(allow_blank=True)
    blocks_count = serializers.IntegerField()
    concepts_count = serializers.IntegerField()


class UserConceptEditSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserConceptEdit
        fields = ("id", "custom_explanation", "updated_at")
        read_only_fields = ("id", "updated_at")

    def validate_custom_explanation(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError("custom_explanation cannot be empty")
        return value
