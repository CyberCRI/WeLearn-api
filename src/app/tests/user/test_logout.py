"""POST /user/logout ends the session and drops its cookie."""

import unittest
import uuid
from datetime import datetime, timedelta
from unittest import mock

from fastapi.testclient import TestClient
from welearn_database.data.models import InferredUser, Session

from src.app.core.config import settings
from src.app.shared.domain.constants import SESSION_COOKIE_NAME
from src.app.tests.data_collection.sqlite_db import make_sqlite_session_maker
from src.main import app

BASE = f"{settings.API_V1_STR}/user"
HEADERS = {"X-API-Key": "test"}


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
@mock.patch("src.app.user.api.router.end_session_sync")
class LogoutApiTests(unittest.TestCase):
    def assert_cookie_deleted(self, response):
        set_cookie = response.headers["set-cookie"]
        self.assertTrue(set_cookie.startswith(f'{SESSION_COOKIE_NAME}="";'))
        lowered = set_cookie.lower()
        self.assertIn("max-age=0", lowered)
        self.assertIn("expires=", lowered)
        self.assertIn(f"domain={settings.SESSION_COOKIE_DOMAIN}", lowered)
        self.assertIn("path=/", lowered)
        self.assertIn("secure", lowered)
        self.assertIn("httponly", lowered)
        # same attributes as the cookie set by /user_and_session
        self.assertNotIn("samesite", lowered)

    def test_logout_with_session_cookie(self, end_session_mock):
        session_id = uuid.uuid4()
        client = TestClient(app, cookies={SESSION_COOKIE_NAME: str(session_id)})

        response = client.post(f"{BASE}/logout", headers=HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "logged out"})
        end_session_mock.assert_called_once_with(session_id)
        self.assert_cookie_deleted(response)

    def test_logout_with_session_header(self, end_session_mock):
        session_id = uuid.uuid4()

        response = TestClient(app).post(
            f"{BASE}/logout", headers={**HEADERS, "X-Session-Id": str(session_id)}
        )

        self.assertEqual(response.status_code, 200)
        end_session_mock.assert_called_once_with(session_id)

    def test_logout_without_session(self, end_session_mock):
        response = TestClient(app).post(f"{BASE}/logout", headers=HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "logged out"})
        end_session_mock.assert_not_called()
        self.assert_cookie_deleted(response)

    def test_logout_with_invalid_session_value(self, end_session_mock):
        response = TestClient(app).post(
            f"{BASE}/logout", headers={**HEADERS, "X-Session-Id": "not-a-uuid"}
        )

        self.assertEqual(response.status_code, 200)
        end_session_mock.assert_not_called()


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
@mock.patch(
    "src.app.shared.api.monitor_requests.register_endpoint", new=mock.MagicMock()
)
class LogoutFlowTests(unittest.TestCase):
    """Against sqlite: after logout, the account's session no longer resolves."""

    def setUp(self):
        self.engine, self.s_maker = make_sqlite_session_maker()
        patcher = mock.patch("src.app.user.infra.queries.session_maker", self.s_maker)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(app)

    def test_account_session_gives_a_new_guest_after_logout(self):
        account, session_id = uuid.uuid4(), uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=account, keycloak_id=uuid.uuid4()))
            s.add(
                Session(
                    id=session_id,
                    inferred_user_id=account,
                    end_at=datetime.now() + timedelta(hours=1),
                )
            )
            s.commit()
        session_headers = {**HEADERS, "X-Session-Id": str(session_id)}

        before = self.client.post(f"{BASE}/user_and_session", headers=session_headers)
        self.assertEqual(before.json()["user_id"], str(account))

        logout = self.client.post(f"{BASE}/logout", headers=session_headers)
        self.assertEqual(logout.status_code, 200)

        after = self.client.post(f"{BASE}/user_and_session", headers=session_headers)
        self.assertEqual(after.status_code, 200)
        self.assertNotEqual(after.json()["user_id"], str(account))
        self.assertNotEqual(after.json()["session_id"], str(session_id))
