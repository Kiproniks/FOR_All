from django.contrib import admin

from .models import (
    BookSummary,
    BookSentence,
    Concept,
    ConceptMention,
    GlobalBookCache,
    GlobalLogicalThoughtBlock,
    SentenceThought,
    SequentialThoughtGroup,
    LogicalBlock,
    TermDefinition,
    ThoughtBlockMembership,
    ThoughtChainAnalysisRun,
    ThoughtRelation,
    UserBook,
    UserConceptEdit,
    UserTermEdit,
)


@admin.register(GlobalBookCache)
class GlobalBookCacheAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "file_hash", "analysis_version", "updated_at")
    search_fields = ("title", "authors", "file_hash")


@admin.register(UserBook)
class UserBookAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title", "status", "is_protected", "uploaded_at")
    list_filter = ("status", "is_protected")
    search_fields = ("title", "authors", "file_hash", "original_filename")


@admin.register(LogicalBlock)
class LogicalBlockAdmin(admin.ModelAdmin):
    list_display = ("id", "global_book", "order_number", "title", "chapter_title", "token_count")
    search_fields = ("title", "chapter_title")


@admin.register(Concept)
class ConceptAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "normalized_name", "updated_at")
    search_fields = ("name", "normalized_name")


@admin.register(ConceptMention)
class ConceptMentionAdmin(admin.ModelAdmin):
    list_display = ("id", "concept", "global_book", "logical_block", "importance_score")
    search_fields = ("concept__name", "logical_block__title", "source_quote")


@admin.register(UserConceptEdit)
class UserConceptEditAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "concept_mention", "updated_at")


@admin.register(BookSummary)
class BookSummaryAdmin(admin.ModelAdmin):
    list_display = ("id", "global_book", "updated_at")


@admin.register(BookSentence)
class BookSentenceAdmin(admin.ModelAdmin):
    list_display = ("id", "global_book", "book", "index", "chapter_title")
    search_fields = ("text", "chapter_title", "section_title")
    list_filter = ("global_book",)


@admin.register(SentenceThought)
class SentenceThoughtAdmin(admin.ModelAdmin):
    list_display = ("id", "global_book", "book", "index", "is_meaningful", "fallback_used")
    search_fields = ("thought_text", "normalized_thought")
    list_filter = ("is_meaningful", "fallback_used", "json_valid")


@admin.register(SequentialThoughtGroup)
class SequentialThoughtGroupAdmin(admin.ModelAdmin):
    list_display = ("id", "global_book", "book", "index", "start_sentence_index", "end_sentence_index")
    search_fields = ("main_thought",)


@admin.register(ThoughtRelation)
class ThoughtRelationAdmin(admin.ModelAdmin):
    list_display = ("id", "source_thought", "target_thought", "relation", "score")
    list_filter = ("relation",)


@admin.register(GlobalLogicalThoughtBlock)
class GlobalLogicalThoughtBlockAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "updated_at")
    search_fields = ("title", "main_idea", "summary")


@admin.register(ThoughtBlockMembership)
class ThoughtBlockMembershipAdmin(admin.ModelAdmin):
    list_display = ("id", "thought", "block", "relevance_score")
    list_filter = ("block",)


@admin.register(ThoughtChainAnalysisRun)
class ThoughtChainAnalysisRunAdmin(admin.ModelAdmin):
    list_display = ("id", "book", "status", "model_name", "processed_sentences", "total_sentences")
    list_filter = ("status",)


# Legacy glossary admin sections.
@admin.register(TermDefinition)
class TermDefinitionAdmin(admin.ModelAdmin):
    list_display = ("id", "term", "global_cache", "frequency")
    search_fields = ("term", "normalized_term")


@admin.register(UserTermEdit)
class UserTermEditAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "user_book", "term_definition", "updated_at")
