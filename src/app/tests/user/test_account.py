import unittest
import uuid
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.app.core.config import settings
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakTokenError,
    KeycloakUnavailableError,
    KeycloakUser,
)
from src.app.user.services import account
from src.main import app

ROUTER = "src.app.user.api.router"
BASE = f"{settings.API_V1_STR}/user"
BEARER = {"Authorization": "Bearer a.b.c"}

client = TestClient(app)


def make_request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


def restrictions(enabled: bool):
    return mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", enabled)


class RequireAccountTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.account = KeycloakUser(keycloak_id=uuid.uuid4())
        patcher = mock.patch.object(
            account, "decode_keycloak_token", return_value=self.account
        )
        self.decode = patcher.start()
        self.addCleanup(patcher.stop)

    async def assert_http_error(self, headers, status_code, detail):
        with self.assertRaises(HTTPException) as ctx:
            await account.require_account(make_request(headers))
        self.assertEqual(ctx.exception.status_code, status_code)
        self.assertEqual(ctx.exception.detail, detail)

    async def test_restrictions_off_checks_nothing(self):
        with restrictions(False):
            self.assertIsNone(await account.require_account(make_request({})))
            self.assertIsNone(await account.require_account(make_request(BEARER)))
        self.decode.assert_not_called()

    async def test_restrictions_on_without_token_requires_an_account(self):
        with restrictions(True):
            await self.assert_http_error(
                {"X-Session-Id": str(uuid.uuid4())},
                401,
                {
                    "message": "An account is required for this feature",
                    "code": "ACCOUNT_REQUIRED",
                },
            )
        self.decode.assert_not_called()

    async def test_restrictions_on_with_token_returns_the_account(self):
        with restrictions(True):
            result = await account.require_account(make_request(BEARER))
        self.assertEqual(result, self.account)
        self.decode.assert_called_once_with("a.b.c")

    async def test_invalid_token_is_a_plain_401_without_account_required(self):
        # the client must tell "sign in again" from "create an account"
        self.decode.side_effect = KeycloakTokenError("expired")
        with restrictions(True):
            await self.assert_http_error(BEARER, 401, "Invalid or expired token")

    async def test_keycloak_unavailable_is_503(self):
        self.decode.side_effect = KeycloakUnavailableError("down")
        with restrictions(True):
            await self.assert_http_error(BEARER, 503, "Keycloak unavailable")

    async def test_keycloak_not_configured_is_503_auth_not_configured(self):
        self.decode.side_effect = KeycloakConfigError("missing")
        with restrictions(True):
            with self.assertRaises(HTTPException) as ctx:
                await account.require_account(make_request(BEARER))
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail["code"], "AUTH_NOT_CONFIGURED")

    async def test_verify_account_token_without_token_returns_none(self):
        self.assertIsNone(await account.verify_account_token(make_request({})))
        self.decode.assert_not_called()


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class BookmarksGuestRestrictionsTests(unittest.TestCase):
    def setUp(self):
        self.user_id = uuid.uuid4()
        self.document_id = uuid.uuid4()
        self.cookie_headers = {"X-API-Key": "test", "X-Session-Id": str(uuid.uuid4())}
        self.resolve_guest = self._patch(
            "resolve_user_and_session", return_value=(self.user_id, uuid.uuid4())
        )
        self.resolve_account = self._patch(
            "resolve_keycloak_user_and_session",
            return_value=(self.user_id, uuid.uuid4()),
        )
        self.get_bookmarks = self._patch("get_user_bookmarks_sync", return_value=[])
        self.add_bookmark = self._patch(
            "add_user_bookmark_sync", return_value=self.document_id
        )
        self.delete_bookmark = self._patch(
            "delete_user_bookmark_sync", return_value=self.document_id
        )
        self.delete_bookmarks = self._patch(
            "delete_user_bookmarks_sync", return_value=0
        )

    def _patch(self, name, **kwargs):
        patcher = mock.patch(f"{ROUTER}.{name}", **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _calls(self, headers):
        url = f"{BASE}/bookmarks/{self.document_id}"
        legacy = f"{BASE}/bookmarks/:document_id?document_id={self.document_id}"
        return [
            client.get(f"{BASE}/bookmarks", headers=headers),
            client.delete(f"{BASE}/bookmarks", headers=headers),
            client.post(url, headers=headers),
            client.delete(url, headers=headers),
            client.post(legacy, headers=headers),
            client.delete(legacy, headers=headers),
        ]

    def test_restrictions_on_guest_cookie_gets_account_required(self):
        with restrictions(True):
            responses = self._calls(self.cookie_headers)

        for response in responses:
            self.assertEqual(response.status_code, 401, response.request.url)
            self.assertEqual(response.json()["detail"]["code"], "ACCOUNT_REQUIRED")
        self.resolve_guest.assert_not_called()
        for query in (
            self.get_bookmarks,
            self.add_bookmark,
            self.delete_bookmark,
            self.delete_bookmarks,
        ):
            query.assert_not_called()

    def test_restrictions_on_account_token_is_served(self):
        with restrictions(True):
            responses = self._calls({**self.cookie_headers, **BEARER})

        for response in responses:
            self.assertEqual(response.status_code, 200, response.request.url)
        self.get_bookmarks.assert_called_once_with(self.user_id)
        self.resolve_guest.assert_not_called()

    def test_restrictions_off_guest_cookie_is_served(self):
        with restrictions(False):
            responses = self._calls(self.cookie_headers)

        for response in responses:
            self.assertEqual(response.status_code, 200, response.request.url)
        self.resolve_account.assert_not_called()

    def test_restrictions_on_invalid_token_is_a_plain_401(self):
        self.resolve_account.side_effect = KeycloakTokenError("expired")
        with restrictions(True):
            response = client.get(
                f"{BASE}/bookmarks", headers={**self.cookie_headers, **BEARER}
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid or expired token")
        self.get_bookmarks.assert_not_called()


class AccountRequiredOpenApiTests(unittest.TestCase):
    def test_account_only_routes_document_account_required(self):
        paths = app.openapi()["paths"]
        api = settings.API_V1_STR
        operations = [
            paths[f"{api}/user/bookmarks"]["get"],
            paths[f"{api}/user/bookmarks/{{document_id}}"]["post"],
            paths[f"{api}/bibliography/export_bibliography"]["post"],
            paths[f"{api}/metric/syllabus_downloaded"]["post"],
        ]
        for operation in operations:
            self.assertIn(
                "ACCOUNT_REQUIRED", operation["responses"]["401"]["description"]
            )
        # the session route only reads an optional token: no ACCOUNT_REQUIRED
        session = paths[f"{api}/user/user_and_session"]["post"]
        self.assertNotIn("ACCOUNT_REQUIRED", session["responses"]["401"]["description"])
