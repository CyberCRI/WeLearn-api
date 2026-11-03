import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.app.shared.utils.dependencies import get_settings
from src.main import app

FAKE_SETTINGS = SimpleNamespace(
    DATABASE_HEALTHCHECK_URL="http://db-health.local", HTTPX_TIMEOUT=3
)


def _client_returning(response):
    """Build an httpx.AsyncClient replacement whose get() returns `response`."""
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=http_client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx), http_client


class DbHealthTests(TestCase):
    """/health/db reads DATABASE_HEALTHCHECK_URL and HTTPX_TIMEOUT from settings.

    Those keys are not declared on `Settings`, so the settings dependency is
    overridden here to exercise the endpoint logic itself.
    """

    def setUp(self):
        app.dependency_overrides[get_settings] = lambda: FAKE_SETTINGS
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_settings, None)

    def test_db_health_returns_remote_status(self):
        rep = MagicMock()
        rep.json.return_value = {"status": "OK"}
        async_client_cls, http_client = _client_returning(rep)

        with patch("src.app.shared.api.health.httpx.AsyncClient", async_client_cls):
            response = self.client.get("/health/db")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})
        http_client.get.assert_awaited_once_with(
            "http://db-health.local/health", timeout=3
        )

    def test_db_health_forwards_degraded_status(self):
        rep = MagicMock()
        rep.json.return_value = {"status": "DEGRADED"}
        async_client_cls, _ = _client_returning(rep)

        with patch("src.app.shared.api.health.httpx.AsyncClient", async_client_cls):
            response = self.client.get("/health/db")

        self.assertEqual(response.json(), {"status": "DEGRADED"})

    def test_db_health_invalid_json_returns_403(self):
        rep = MagicMock()
        rep.json.side_effect = json.JSONDecodeError("bad", "doc", 0)
        async_client_cls, _ = _client_returning(rep)

        with patch("src.app.shared.api.health.httpx.AsyncClient", async_client_cls):
            response = self.client.get("/health/db")

        self.assertEqual(response.status_code, 403)
