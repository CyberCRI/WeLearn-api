import json
import unittest
import uuid
from unittest.mock import patch

import httpx

from src.app.user.infra import keycloak, keycloak_admin
from src.app.user.infra.keycloak import KeycloakConfigError, KeycloakUnavailableError
from src.app.user.infra.keycloak_admin import (
    KeycloakUserExistsError,
    create_user,
    find_user_id_by_email,
    send_signup_email,
)

KEYCLOAK_URL = "https://auth.example.org"
KEYCLOAK_REALM = "welearn"
TOKEN_URL = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
ADMIN_URL = f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}"


class FakeKeycloak:
    """Answers the token endpoint, then hands admin calls to `handler`."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "admin-token"})
        self.requests.append(request)
        return self.handler(request)


@patch.object(keycloak_admin.settings, "KEYCLOAK_SIGNUP_REDIRECT_URI", None)
@patch.object(keycloak_admin.settings, "KEYCLOAK_ADMIN_CLIENT_SECRET", "secret")
@patch.object(keycloak_admin.settings, "KEYCLOAK_ADMIN_CLIENT_ID", "welearn-api")
@patch.object(keycloak.settings, "KEYCLOAK_REALM", KEYCLOAK_REALM)
@patch.object(keycloak.settings, "KEYCLOAK_URL", KEYCLOAK_URL)
class TestKeycloakAdmin(unittest.IsolatedAsyncioTestCase):
    def _fake(self, handler) -> FakeKeycloak:
        fake = FakeKeycloak(handler)
        patcher = patch.object(
            keycloak_admin,
            "_http_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    async def test_find_user_by_email_found(self):
        user_id = uuid.uuid4()
        fake = self._fake(lambda r: httpx.Response(200, json=[{"id": str(user_id)}]))

        result = await find_user_id_by_email("jane@example.org")

        self.assertEqual(result, user_id)
        request = fake.requests[0]
        self.assertEqual(request.url.path, f"/admin/realms/{KEYCLOAK_REALM}/users")
        self.assertEqual(request.url.params["email"], "jane@example.org")
        self.assertEqual(request.url.params["exact"], "true")
        self.assertEqual(request.headers["Authorization"], "Bearer admin-token")

    async def test_find_user_by_email_not_found(self):
        self._fake(lambda r: httpx.Response(200, json=[]))
        self.assertIsNone(await find_user_id_by_email("jane@example.org"))

    async def test_create_user(self):
        user_id = uuid.uuid4()
        fake = self._fake(
            lambda r: httpx.Response(
                201, headers={"Location": f"{ADMIN_URL}/users/{user_id}"}
            )
        )

        result = await create_user("jane@example.org", "Jane", None)

        self.assertEqual(result, user_id)
        payload = json.loads(fake.requests[0].content)
        self.assertEqual(payload["username"], "jane@example.org")
        self.assertEqual(payload["email"], "jane@example.org")
        self.assertEqual(payload["firstName"], "Jane")
        self.assertNotIn("lastName", payload)
        self.assertNotIn("credentials", payload)
        self.assertEqual(
            payload["requiredActions"], ["VERIFY_EMAIL", "UPDATE_PASSWORD"]
        )

    async def test_create_user_conflict(self):
        self._fake(lambda r: httpx.Response(409, json={"errorMessage": "exists"}))
        with self.assertRaises(KeycloakUserExistsError):
            await create_user("jane@example.org")

    async def test_create_user_bad_location(self):
        self._fake(lambda r: httpx.Response(201))
        with self.assertRaises(KeycloakUnavailableError):
            await create_user("jane@example.org")

    async def test_server_error(self):
        self._fake(lambda r: httpx.Response(500))
        with self.assertRaises(KeycloakUnavailableError):
            await find_user_id_by_email("jane@example.org")

    async def test_connection_error(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        self._fake(handler)
        with self.assertRaises(KeycloakUnavailableError):
            await find_user_id_by_email("jane@example.org")

    async def test_token_refused(self):
        def handler(request):
            return httpx.Response(401)

        patcher = patch.object(
            keycloak_admin,
            "_http_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        with self.assertRaises(KeycloakUnavailableError):
            await find_user_id_by_email("jane@example.org")

    async def test_missing_admin_config(self):
        fake = self._fake(lambda r: httpx.Response(200, json=[]))
        with patch.object(
            keycloak_admin.settings, "KEYCLOAK_ADMIN_CLIENT_SECRET", None
        ):
            with self.assertRaises(KeycloakConfigError):
                await find_user_id_by_email("jane@example.org")
        self.assertEqual(fake.requests, [])

    async def test_send_signup_email(self):
        keycloak_id = uuid.uuid4()
        fake = self._fake(lambda r: httpx.Response(204))

        await send_signup_email(keycloak_id)

        request = fake.requests[0]
        self.assertEqual(request.method, "PUT")
        self.assertEqual(
            request.url.path,
            f"/admin/realms/{KEYCLOAK_REALM}/users/{keycloak_id}/execute-actions-email",
        )
        self.assertEqual(
            json.loads(request.content), ["VERIFY_EMAIL", "UPDATE_PASSWORD"]
        )
        self.assertNotIn("redirect_uri", request.url.params)

    async def test_send_signup_email_with_redirect(self):
        fake = self._fake(lambda r: httpx.Response(204))
        with (
            patch.object(
                keycloak_admin.settings, "KEYCLOAK_CLIENT_ID", "welearn-front"
            ),
            patch.object(
                keycloak_admin.settings,
                "KEYCLOAK_SIGNUP_REDIRECT_URI",
                "https://front.example.org",
            ),
        ):
            await send_signup_email(uuid.uuid4())

        params = fake.requests[0].url.params
        self.assertEqual(params["client_id"], "welearn-front")
        self.assertEqual(params["redirect_uri"], "https://front.example.org")


if __name__ == "__main__":
    unittest.main()
