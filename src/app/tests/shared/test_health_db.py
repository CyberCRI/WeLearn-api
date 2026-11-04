from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.main import app


def _session_maker_with(session):
    """Build a `session_maker` replacement whose context manager yields `session`."""
    ctx = MagicMock()
    ctx.__enter__.return_value = session
    ctx.__exit__.return_value = False
    return MagicMock(return_value=ctx)


class DbHealthTests(TestCase):
    """/health/db runs `SELECT 1` against Postgres through `session_maker`."""

    def setUp(self):
        self.client = TestClient(app)

    def test_db_health_ok_when_select_succeeds(self):
        session = MagicMock()
        maker = _session_maker_with(session)

        with patch("src.app.shared.api.health.session_maker", maker):
            response = self.client.get("/health/db")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})
        maker.assert_called_once_with()
        session.execute.assert_called_once()
        self.assertEqual(str(session.execute.call_args.args[0]), "SELECT 1")

    def test_db_health_needs_no_api_key(self):
        maker = _session_maker_with(MagicMock())

        with patch("src.app.shared.api.health.session_maker", maker):
            response = self.client.get("/health/db", headers={})

        self.assertEqual(response.status_code, 200)

    def test_db_health_503_when_database_unreachable(self):
        session = MagicMock()
        session.execute.side_effect = OperationalError(
            "SELECT 1", {}, Exception("password=secret host unreachable")
        )
        maker = _session_maker_with(session)

        with patch("src.app.shared.api.health.session_maker", maker):
            response = self.client.get("/health/db")

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "DATABASE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)

    def test_db_health_503_when_connection_fails_with_os_error(self):
        maker = MagicMock(side_effect=OSError("connection refused"))

        with patch("src.app.shared.api.health.session_maker", maker):
            response = self.client.get("/health/db")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "DATABASE_UNAVAILABLE")
        self.assertNotIn("connection refused", response.text)
