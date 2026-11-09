import unittest
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.app.core.config import settings
from src.app.shared.domain.exceptions import UserNotFoundError
from src.app.tests.shared.log_helpers import format_records
from src.app.user.infra.keycloak import KeycloakUnavailableError
from src.main import app

ROUTER = "src.app.user.api.router"
BASE = f"{settings.API_V1_STR}/user"
SECRET = "S3cr3t%#"

client = TestClient(app)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class UserApiExtraTests(unittest.TestCase):
    def setUp(self):
        self.user_id = uuid.uuid4()
        self.session_id = uuid.uuid4()
        self.document_id = uuid.uuid4()
        self.headers = {"X-API-Key": "test", "X-Session-Id": str(self.session_id)}

        patcher = mock.patch(f"{ROUTER}.resolve_user_and_session")
        self.resolve_user = patcher.start()
        self.addCleanup(patcher.stop)
        self.resolve_user.return_value = (self.user_id, self.session_id)

    def _patch(self, name, **kwargs):
        patcher = mock.patch(f"{ROUTER}.{name}", **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    # ---- /user_and_session ----

    def test_user_and_session_user_not_found_returns_404(self):
        self.resolve_user.side_effect = UserNotFoundError("no such user")

        response = client.post(f"{BASE}/user_and_session", headers=self.headers)

        self.assertEqual(response.status_code, 404)
        self.assertIn("no such user", response.json()["detail"])

    def test_user_and_session_unexpected_error_returns_500(self):
        self.resolve_user.side_effect = RuntimeError("db down")

        response = client.post(f"{BASE}/user_and_session", headers=self.headers)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "db down")

    def test_user_and_session_sets_session_cookie(self):
        response = client.post(f"{BASE}/user_and_session", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], str(self.session_id))
        self.assertIn(str(self.session_id), response.headers["set-cookie"])
        self.assertEqual(
            self.resolve_user.call_args.kwargs["session_uuid"], self.session_id
        )
        # the only path allowed to renew a guest's expired session
        self.assertTrue(self.resolve_user.call_args.kwargs["renew_expired_guest"])

    def test_other_routes_do_not_renew_expired_sessions(self):
        self._patch("get_user_bookmarks_sync", return_value=[])
        self._patch("add_institution_data_to_user_sync")

        client.get(f"{BASE}/bookmarks", headers=self.headers)
        self._institution(self.headers)

        self.assertEqual(self.resolve_user.call_count, 2)
        for call in self.resolve_user.call_args_list:
            self.assertFalse(call.kwargs.get("renew_expired_guest", False))

    # ---- resolve_bookmark_user ----

    def test_bookmarks_keycloak_unavailable_returns_503(self):
        resolve_kc = self._patch(
            "resolve_keycloak_user_and_session",
            side_effect=KeycloakUnavailableError("down"),
        )
        get_bookmarks = self._patch("get_user_bookmarks_sync")

        response = client.get(
            f"{BASE}/bookmarks",
            headers={"X-API-Key": "test", "Authorization": "Bearer a.b.c"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Keycloak unavailable")
        resolve_kc.assert_called_once()
        get_bookmarks.assert_not_called()

    # ---- GET /bookmarks ----

    def test_get_bookmarks_returns_bookmarks(self):
        bookmark = SimpleNamespace(
            id=uuid.uuid4(),
            document_id=self.document_id,
            inferred_user_id=self.user_id,
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        get_bookmarks = self._patch("get_user_bookmarks_sync", return_value=[bookmark])

        response = client.get(f"{BASE}/bookmarks", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        body = response.json()["bookmarks"]
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["document_id"], str(self.document_id))
        get_bookmarks.assert_called_once_with(self.user_id)

    def test_get_bookmarks_value_error_returns_404(self):
        self._patch("get_user_bookmarks_sync", side_effect=ValueError("no user"))

        response = client.get(f"{BASE}/bookmarks", headers=self.headers)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "no user")

    def test_get_bookmarks_unexpected_error_returns_500(self):
        self._patch("get_user_bookmarks_sync", side_effect=RuntimeError("boom"))

        response = client.get(f"{BASE}/bookmarks", headers=self.headers)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "boom")

    def test_bookmarks_database_error_returns_generic_503(self):
        # the global handler answers, not the route's str(e) 500 (C-9)
        self._patch(
            "get_user_bookmarks_sync",
            side_effect=OperationalError("SELECT", {}, Exception(f"password={SECRET}")),
        )

        with self.assertLogs(level="DEBUG") as logs:
            response = client.get(f"{BASE}/bookmarks", headers=self.headers)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "DATABASE_UNAVAILABLE")
        self.assertNotIn(SECRET, response.text)
        self.assertNotIn(SECRET, format_records(logs.records))

    # ---- DELETE /bookmarks ----

    def test_delete_bookmarks_returns_count(self):
        delete_all = self._patch("delete_user_bookmarks_sync", return_value=3)

        response = client.delete(f"{BASE}/bookmarks", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted": 3})
        delete_all.assert_called_once_with(self.user_id)

    def test_delete_bookmarks_value_error_returns_404(self):
        self._patch("delete_user_bookmarks_sync", side_effect=ValueError("no user"))

        response = client.delete(f"{BASE}/bookmarks", headers=self.headers)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "no user")

    def test_delete_bookmarks_unexpected_error_returns_500(self):
        self._patch("delete_user_bookmarks_sync", side_effect=RuntimeError("boom"))

        response = client.delete(f"{BASE}/bookmarks", headers=self.headers)

        self.assertEqual(response.status_code, 500)

    # ---- DELETE /bookmarks/{document_id} ----

    def _bookmark_url(self):
        return f"{BASE}/bookmarks/{self.document_id}"

    def test_delete_bookmark_returns_document_id(self):
        delete_one = self._patch(
            "delete_user_bookmark_sync", return_value=self.document_id
        )

        response = client.delete(
            self._bookmark_url(),
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted": str(self.document_id)})
        delete_one.assert_called_once_with(self.user_id, self.document_id)

    def test_delete_bookmark_value_error_returns_404(self):
        self._patch("delete_user_bookmark_sync", side_effect=ValueError("missing"))

        response = client.delete(
            self._bookmark_url(),
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "missing")

    def test_delete_bookmark_unexpected_error_returns_500(self):
        self._patch("delete_user_bookmark_sync", side_effect=RuntimeError("boom"))

        response = client.delete(
            self._bookmark_url(),
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 500)

    # ---- POST /bookmarks/{document_id} ----

    def test_add_bookmark_value_error_returns_400(self):
        self._patch("add_user_bookmark_sync", side_effect=ValueError("exists"))

        response = client.post(
            self._bookmark_url(),
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "exists")

    def test_add_bookmark_unexpected_error_returns_500(self):
        self._patch("add_user_bookmark_sync", side_effect=RuntimeError("boom"))

        response = client.post(
            self._bookmark_url(),
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 500)

    def test_add_bookmark_returns_document_id(self):
        add_one = self._patch("add_user_bookmark_sync", return_value=self.document_id)

        response = client.post(self._bookmark_url(), headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"added": str(self.document_id)})
        add_one.assert_called_once_with(self.user_id, self.document_id)

    def test_bookmark_non_uuid_path_returns_422(self):
        add_one = self._patch("add_user_bookmark_sync")
        delete_one = self._patch("delete_user_bookmark_sync")

        for method in (client.post, client.delete):
            with self.subTest(method=method.__name__):
                response = method(f"{BASE}/bookmarks/not-a-uuid", headers=self.headers)

                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"][0]["loc"][0], "path")
        add_one.assert_not_called()
        delete_one.assert_not_called()

    # ---- legacy /bookmarks/:document_id?document_id=... (deprecated) ----

    def test_legacy_add_bookmark_reads_document_id_from_query(self):
        add_one = self._patch("add_user_bookmark_sync", return_value=self.document_id)

        response = client.post(
            f"{BASE}/bookmarks/:document_id",
            params={"document_id": str(self.document_id)},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"added": str(self.document_id)})
        add_one.assert_called_once_with(self.user_id, self.document_id)

    def test_legacy_delete_bookmark_reads_document_id_from_query(self):
        delete_one = self._patch(
            "delete_user_bookmark_sync", return_value=self.document_id
        )

        response = client.delete(
            f"{BASE}/bookmarks/:document_id",
            params={"document_id": str(self.document_id)},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted": str(self.document_id)})
        delete_one.assert_called_once_with(self.user_id, self.document_id)

    # ---- POST /institution-data ----

    def _institution(self, headers):
        return client.post(
            f"{BASE}/institution-data",
            json={"institution": "LPI", "role": "teacher"},
            headers=headers,
        )

    def test_institution_data_without_session_returns_401(self):
        add_data = self._patch("add_institution_data_to_user_sync")

        response = self._institution({"X-API-Key": "test"})

        self.assertEqual(response.status_code, 401)
        self.resolve_user.assert_not_called()
        add_data.assert_not_called()

    def test_institution_data_value_error_returns_404(self):
        self._patch("add_institution_data_to_user_sync", side_effect=ValueError("no"))

        response = self._institution(self.headers)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "no")

    def test_institution_data_unexpected_error_returns_500(self):
        add_data = self._patch(
            "add_institution_data_to_user_sync", side_effect=RuntimeError("boom")
        )

        response = self._institution(self.headers)

        self.assertEqual(response.status_code, 500)
        add_data.assert_called_once_with(self.user_id, "LPI", "teacher")
