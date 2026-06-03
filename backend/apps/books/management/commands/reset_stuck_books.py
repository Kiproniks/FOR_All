from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.books.models import UserBook


STUCK_STATUSES = {
    UserBook.Status.PROCESSING,
    UserBook.Status.PARSING,
    UserBook.Status.STRUCTURE_DETECTION,
    UserBook.Status.FILTERING,
    UserBook.Status.CHUNKING,
    UserBook.Status.LLM_SECTION_ANALYSIS,
    UserBook.Status.LLM_CHAPTER_ANALYSIS,
    UserBook.Status.LLM_BOOK_ANALYSIS,
    UserBook.Status.LLM_FAST_BATCHED_SECTION_ANALYSIS,
    UserBook.Status.LLM_FAST_BATCHED_CHAPTER_ANALYSIS,
    UserBook.Status.LLM_FAST_BATCHED_BOOK_ANALYSIS,
    UserBook.Status.BUILDING_MAP,
    UserBook.Status.SAVING_RESULTS,
}


class Command(BaseCommand):
    help = "Reset stuck books that were not updated for too long."

    def add_arguments(self, parser):
        parser.add_argument("--minutes", type=int, default=60, help="Stuck threshold in minutes")
        parser.add_argument(
            "--action",
            choices=["failed_timeout", "queued"],
            default="failed_timeout",
            help="What status to apply to stuck books",
        )
        parser.add_argument("--dry-run", action="store_true", help="Only print candidates without updating")

    def handle(self, *args, **options):
        minutes = max(5, int(options["minutes"]))
        action = options["action"]
        dry_run = bool(options["dry_run"])
        cutoff = timezone.now() - timedelta(minutes=minutes)

        qs = UserBook.objects.filter(status__in=STUCK_STATUSES).filter(
            last_heartbeat_at__lt=cutoff
        ) | UserBook.objects.filter(status__in=STUCK_STATUSES, last_heartbeat_at__isnull=True, updated_at__lt=cutoff)
        qs = qs.distinct().order_by("id")

        total = qs.count()
        self.stdout.write(f"Found {total} stuck books (threshold={minutes} min).")
        for item in qs:
            self.stdout.write(
                f"- id={item.id} status={item.status} stage={item.current_stage} "
                f"updated_at={item.updated_at} heartbeat={item.last_heartbeat_at}"
            )

        if dry_run or total == 0:
            return

        now = timezone.now()
        if action == "queued":
            for item in qs:
                item.status = UserBook.Status.QUEUED
                item.current_stage = "queued"
                item.progress_percent = min(item.progress_percent, 5)
                item.error_message = "Requeued by reset_stuck_books due to stale heartbeat."
                item.last_heartbeat_at = now
                item.finished_at = None
                item.save(
                    update_fields=[
                        "status",
                        "current_stage",
                        "progress_percent",
                        "error_message",
                        "last_heartbeat_at",
                        "finished_at",
                        "updated_at",
                    ]
                )
        else:
            for item in qs:
                item.status = UserBook.Status.FAILED_TIMEOUT
                item.current_stage = "failed_timeout"
                item.progress_percent = min(item.progress_percent, 99)
                item.error_message = "Analysis timed out or heartbeat was stale."
                item.last_heartbeat_at = now
                item.finished_at = now
                item.processed_at = now
                item.save(
                    update_fields=[
                        "status",
                        "current_stage",
                        "progress_percent",
                        "error_message",
                        "last_heartbeat_at",
                        "finished_at",
                        "processed_at",
                        "updated_at",
                    ]
                )

        self.stdout.write(self.style.SUCCESS(f"Updated {total} books with action={action}."))
