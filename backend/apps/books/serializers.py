from rest_framework import serializers

from apps.books.models import (
    Concept,
    ConceptMention,
    LogicalBlock,
    UserBook,
    UserConceptEdit,
)


class UserBookSerializer(serializers.ModelSerializer):
    logical_blocks_count = serializers.SerializerMethodField()
    concepts_count = serializers.SerializerMethodField()

    class Meta:
        model = UserBook
        fields = (
            "id",
            "title",
            "authors",
            "status",
            "is_protected",
            "uploaded_at",
            "processed_at",
            "views_count",
            "logical_blocks_count",
            "concepts_count",
        )

    def get_logical_blocks_count(self, obj):
        if not obj.global_cache_id:
            return 0
        return obj.global_cache.logical_blocks.count()

    def get_concepts_count(self, obj):
        if not obj.global_cache_id:
            return 0
        return obj.global_cache.concept_mentions.count()


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
