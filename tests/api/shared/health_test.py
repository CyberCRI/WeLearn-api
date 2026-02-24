from unittest import TestCase

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class SearchTests(TestCase):
    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
