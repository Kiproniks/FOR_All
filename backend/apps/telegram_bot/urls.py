from django.urls import path
from rest_framework.response import Response
from rest_framework.views import APIView


class TelegramHealthView(APIView):
    permission_classes = []

    def get(self, request):
        return Response({"status": "ok"})


urlpatterns = [
    path("telegram/health/", TelegramHealthView.as_view(), name="telegram-health"),
]
