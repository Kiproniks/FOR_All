from __future__ import annotations

from dataclasses import dataclass

from apps.books.models import UserBook

MAX_BOOKS_PER_USER = 50


@dataclass
class RotationResult:
    can_upload: bool
    need_confirmation: bool = False
    reason: str = ""
    book_to_delete: UserBook | None = None


def get_user_books_count(user) -> int:
    return UserBook.objects.filter(user=user).count()


def can_upload_book(user) -> bool:
    return get_user_books_count(user) < MAX_BOOKS_PER_USER


def get_oldest_unprotected_book(user) -> UserBook | None:
    return (
        UserBook.objects.filter(user=user, is_protected=False)
        .order_by("uploaded_at")
        .first()
    )


def rotate_books_if_needed(user, *, confirmed: bool = False) -> RotationResult:
    count = get_user_books_count(user)
    if count < MAX_BOOKS_PER_USER:
        return RotationResult(can_upload=True)

    candidate = get_oldest_unprotected_book(user)
    if candidate is None:
        return RotationResult(
            can_upload=False,
            reason="Достигнут лимит 50 книг, и все книги защищены.",
        )

    if not confirmed:
        return RotationResult(
            can_upload=False,
            need_confirmation=True,
            reason="Требуется подтверждение удаления самой старой незащищенной книги.",
            book_to_delete=candidate,
        )

    candidate.delete()
    return RotationResult(can_upload=True, reason="Старая книга удалена в рамках ротации.")
