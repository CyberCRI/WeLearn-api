"""
Session expiry through the API, against an in-memory sqlite database: the
cookie lives longer than the DB session, only a guest's expired session is
renewed, and only by /user/user_and_session.
"""

import unittest
import uuid
from datetime import datetime, timedelta
from unittest import mock

from fastapi.testclient import TestClient
from welearn_database.data.models import Bookmark, InferredUser, Session

from src.app.core.config import settings
from src.app.tests.data_collection.sqlite_db import make_sqlite_session_maker
from src.main import app

BASE = f"{settings.API_V1_STR}/user"
HEADERS = {"X-API-Key": "test"}


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
@mock.patch(
    "src.app.shared.api.monitor_requests.register_endpoint", new=mock.MagicMock()
)
class SessionExpiryApiTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.s_maker = make_sqlite_session_maker()
        patcher = mock.patch("src.app.user.infra.queries.session_maker", self.s_maker)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(app)

    def _add_user(self, keycloak_id: uuid.UUID | None = None) -> uuid.UUID:
        user_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=user_id, keycloak_id=keycloak_id))
            s.commit()
        return user_id

    def _add_session(self, user_id: uuid.UUID, end_at: datetime) -> uuid.UUID:
        session_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(Session(id=session_id, inferred_user_id=user_id, end_at=end_at))
            s.commit()
        return session_id

    def _expired_session(self, user_id: uuid.UUID) -> uuid.UUID:
        return self._add_session(user_id, datetime.now() - timedelta(days=3))

    def _user_and_session(self, session_id: uuid.UUID):
        return self.client.post(
            f"{BASE}/user_and_session",
            headers={**HEADERS, "X-Session-Id": str(session_id)},
        )

    def test_expired_guest_session_is_renewed_for_the_same_user(self):
        guest = self._add_user()
        expired = self._expired_session(guest)

        response = self._user_and_session(expired)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user_id"], str(guest))
        self.assertNotEqual(body["session_id"], str(expired))
        self.assertIn(body["session_id"], response.headers["set-cookie"])
        with self.s_maker() as s:
            new_session = s.get(Session, uuid.UUID(body["session_id"]))
            self.assertEqual(new_session.inferred_user_id, guest)
            self.assertGreater(new_session.end_at, datetime.now())

    def test_valid_session_is_kept(self):
        guest = self._add_user()
        valid = self._add_session(guest, datetime.now() + timedelta(hours=1))

        body = self._user_and_session(valid).json()

        self.assertEqual(body["user_id"], str(guest))
        self.assertEqual(body["session_id"], str(valid))

    def test_expired_keycloak_session_is_not_renewed_by_cookie(self):
        account = self._add_user(keycloak_id=uuid.uuid4())
        expired = self._expired_session(account)

        response = self._user_and_session(expired)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        # a new guest, the account needs its token
        self.assertNotEqual(body["user_id"], str(account))
        self.assertNotEqual(body["session_id"], str(expired))
        with self.s_maker() as s:
            new_user = s.get(InferredUser, uuid.UUID(body["user_id"]))
            self.assertIsNone(new_user.keycloak_id)

    def test_bookmarks_do_not_accept_an_expired_session(self):
        guest = self._add_user()
        expired = self._expired_session(guest)
        with self.s_maker() as s:
            s.add(Bookmark(document_id=uuid.uuid4(), inferred_user_id=guest))
            s.commit()

        response = self.client.get(
            f"{BASE}/bookmarks", headers={**HEADERS, "X-Session-Id": str(expired)}
        )

        # the expired cookie no longer reaches the guest's bookmarks
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"bookmarks": []})

    def test_bookmarks_with_a_valid_session(self):
        guest = self._add_user()
        valid = self._add_session(guest, datetime.now() + timedelta(hours=1))
        document_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(Bookmark(document_id=document_id, inferred_user_id=guest))
            s.commit()

        response = self.client.get(
            f"{BASE}/bookmarks", headers={**HEADERS, "X-Session-Id": str(valid)}
        )

        bookmarks = response.json()["bookmarks"]
        self.assertEqual([b["document_id"] for b in bookmarks], [str(document_id)])


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class RemovedSessionMintingRoutesTests(unittest.TestCase):
    """POST /user/user and POST /user/session minted sessions for any user id."""

    def test_routes_are_gone(self):
        client = TestClient(app)
        user_id = str(uuid.uuid4())
        for path, params in (
            ("user", {"user_id": user_id}),
            ("session", {"user_id": user_id}),
        ):
            with self.subTest(path=path):
                response = client.post(f"{BASE}/{path}", params=params, headers=HEADERS)
                self.assertIn(response.status_code, (404, 405))

    def test_routes_are_not_in_the_schema(self):
        paths = app.openapi()["paths"]
        self.assertNotIn(f"{BASE}/user", paths)
        self.assertNotIn(f"{BASE}/session", paths)
