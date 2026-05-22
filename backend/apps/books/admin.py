from django.contrib import admin

from .models import (
    BookSummary,
    Concept,
    ConceptMention,
    GlobalBookCache,
    LogicalBlock,
    TermDefinition,
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


# Legacy glossary admin sections.
@admin.register(TermDefinition)
class TermDefinitionAdmin(admin.ModelAdmin):
    list_display = ("id", "term", "global_cache", "frequency")
    search_fields = ("term", "normalized_term")


@admin.register(UserTermEdit)
class UserTermEditAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "user_book", "term_definition", "updated_at")
