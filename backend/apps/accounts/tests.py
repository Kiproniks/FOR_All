from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.test import APITestCase


class AuthApiTests(APITestCase):
    def test_register_user(self):
        payload = {
            "email": "user@example.com",
            "password": "StrongPass123",
            "password_repeat": "StrongPass123",
        }
        response = self.client.post("/api/auth/register/", payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("token", response.data)
        self.assertEqual(get_user_model().objects.count(), 1)
