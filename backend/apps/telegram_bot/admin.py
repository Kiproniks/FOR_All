from django.contrib import admin

from apps.telegram_bot.models import TelegramProfile


@admin.register(TelegramProfile)
class TelegramProfileAdmin(admin.ModelAdmin):
    list_display = ("telegram_id", "username", "user", "updated_at")
    search_fields = ("telegram_id", "username", "user__email")
