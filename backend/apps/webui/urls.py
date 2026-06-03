from django.urls import path

from . import views

app_name = "webui"

urlpatterns = [
    path("", views.root_redirect, name="root"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("library/", views.library_view, name="library"),
    path("library/upload/", views.upload_books_view, name="upload-books"),
    path("library/books/<int:book_id>/", views.book_summary_view, name="book-summary"),
    path("library/books/<int:book_id>/notes/", views.book_notes_view, name="book-notes"),
    path("library/books/<int:book_id>/notes/generate/", views.generate_book_notes_view, name="generate-book-notes"),
    path("library/books/<int:book_id>/delete/", views.delete_book_view, name="delete-book"),
    path("library/books/<int:book_id>/protect/", views.protect_book_view, name="protect-book"),
    path("library/books/<int:book_id>/reanalyze/", views.reanalyze_book_view, name="reanalyze-book"),
    path("library/books/<int:book_id>/blocks/<int:block_id>/", views.block_detail_view, name="block-detail"),
    path("library/books/<int:book_id>/export/<str:fmt>/", views.export_book_web_view, name="export-book"),
    path("library/concepts/", views.all_concepts_view, name="concepts"),
    path("library/map/", views.concept_map_view, name="concept-map"),
    path("library/concepts/<int:concept_id>/", views.concept_detail_view, name="concept-detail"),
    path("library/concepts/<int:concept_id>/compare/", views.concept_compare_view, name="concept-compare"),
    path("library/mentions/<int:mention_id>/edit/", views.edit_mention_view, name="edit-mention"),
    path("library/mentions/<int:mention_id>/reset/", views.reset_mention_view, name="reset-mention"),
]
