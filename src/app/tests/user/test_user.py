import unittest
import uuid
from unittest import mock
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.app.shared.domain.exceptions import SessionNotFoundError, UserNotFoundError
from src.app.tests.user.test_keycloak import (
    CLIENT_ID,
    KEYCLOAK_REALM,
    KEYCLOAK_URL,
    PRIVATE_KEY,
    make_token,
)
from src.app.user.api.router import (
    SIGNUP_ACCEPTED_MESSAGE,
    SIGNUP_RATE_LIMIT_PER_EMAIL,
    SIGNUP_RATE_LIMIT_PER_IP,
)
from src.app.user.infra import keycloak
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakTokenError,
    KeycloakUnavailableError,
)
from src.main import app

client = TestClient(app)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class UserApiTests(unittest.IsolatedAsyncioTestCase):
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    @mock.patch("src.app.user.api.router.resolve_keycloak_user_and_session")
    async def test_user_and_session_with_keycloak_token(
        self, resolve_keycloak_mock, resolve_anonymous_mock, *mocks
    ):
        """Avec un token Keycloak, l'utilisateur est résolu via Keycloak"""
        user_id, session_id = uuid.uuid4(), uuid.uuid4()
        resolve_keycloak_mock.return_value = (user_id, session_id)

        response = client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={"X-API-Key": "test", "Authorization": "Bearer a.b.c"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], str(session_id))
        self.assertEqual(response.json()["user_id"], str(user_id))
        self.assertEqual(resolve_keycloak_mock.call_args.kwargs["token"], "a.b.c")
        resolve_anonymous_mock.assert_not_called()

    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    @mock.patch("src.app.user.api.router.resolve_keycloak_user_and_session")
    async def test_user_and_session_without_token(
        self, resolve_keycloak_mock, resolve_anonymous_mock, *mocks
    ):
        """Sans token, le flux anonyme est conservé"""
        user_id = uuid.uuid4()
        resolve_anonymous_mock.return_value = (user_id, uuid.uuid4())

        response = client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], str(user_id))
        resolve_anonymous_mock.assert_called_once()
        resolve_keycloak_mock.assert_not_called()

    @mock.patch("src.app.user.services.session.get_or_create_session_sync")
    @mock.patch("src.app.user.services.session.get_or_create_user_sync")
    @mock.patch("src.app.user.services.session.get_user_for_session_renewal")
    async def test_user_and_session_unknown_session_creates_new_guest(
        self, get_user_mock, get_or_create_user_mock, get_or_create_session_mock, *mocks
    ):
        """Unknown session id without a token -> new guest, not an error"""
        new_user_id, new_session_id = uuid.uuid4(), uuid.uuid4()
        get_user_mock.side_effect = SessionNotFoundError("Session not found")
        get_or_create_user_mock.return_value = new_user_id
        get_or_create_session_mock.return_value = new_session_id

        response = client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={"X-API-Key": "test", "X-Session-Id": str(uuid.uuid4())},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], str(new_user_id))
        self.assertEqual(response.json()["session_id"], str(new_session_id))
        self.assertIn(str(new_session_id), response.headers["set-cookie"])

    @mock.patch("src.app.user.api.router.resolve_keycloak_user_and_session")
    async def test_user_and_session_invalid_token(self, resolve_keycloak_mock, *mocks):
        """Token invalide ou expiré -> 401"""
        resolve_keycloak_mock.side_effect = KeycloakTokenError("expired")

        response = client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={"X-API-Key": "test", "Authorization": "Bearer a.b.c"},
        )

        self.assertEqual(response.status_code, 401)

    @mock.patch("src.app.user.api.router.resolve_keycloak_user_and_session")
    async def test_user_and_session_keycloak_unavailable(
        self, resolve_keycloak_mock, *mocks
    ):
        """Keycloak injoignable -> 503"""
        resolve_keycloak_mock.side_effect = KeycloakUnavailableError("timeout")

        response = client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={"X-API-Key": "test", "Authorization": "Bearer a.b.c"},
        )

        self.assertEqual(response.status_code, 503)

    @mock.patch.object(keycloak.settings, "KEYCLOAK_AUDIENCE", None)
    @mock.patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", CLIENT_ID)
    @mock.patch.object(keycloak.settings, "KEYCLOAK_REALM", KEYCLOAK_REALM)
    @mock.patch.object(keycloak.settings, "KEYCLOAK_URL", KEYCLOAK_URL)
    @mock.patch.object(keycloak, "get_or_create_user_from_keycloak_sync")
    @mock.patch.object(keycloak, "get_jwks_client")
    async def test_bearer_token_for_another_client(
        self, get_jwks_client_mock, get_or_create_mock, *mocks
    ):
        """Token émis pour un autre client du realm (azp) -> 401"""
        get_jwks_client_mock.return_value.get_signing_key_from_jwt.return_value = (
            MagicMock(key=PRIVATE_KEY.public_key())
        )

        response = client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={
                "X-API-Key": "test",
                "Authorization": f"Bearer {make_token(azp='another-client')}",
            },
        )

        self.assertEqual(response.status_code, 401)
        get_or_create_mock.assert_not_called()

    @mock.patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", None)
    @mock.patch.object(keycloak.settings, "KEYCLOAK_REALM", "welearn")
    @mock.patch.object(keycloak.settings, "KEYCLOAK_URL", "https://auth.example.org")
    @mock.patch.object(keycloak, "_config_error_logged", False)
    @mock.patch.object(keycloak, "get_jwks_client")
    @mock.patch.object(keycloak, "logger")
    async def test_bearer_token_keycloak_not_configured(
        self, logger_mock, get_jwks_client_mock, *mocks
    ):
        """Token sans KEYCLOAK_CLIENT_ID configuré -> 503 AUTH_NOT_CONFIGURED, loggé une fois"""
        headers = {"X-API-Key": "test", "Authorization": "Bearer a.b.c"}

        responses = [
            client.post(
                f"{settings.API_V1_STR}/user/user_and_session", headers=headers
            ),
            client.get(f"{settings.API_V1_STR}/user/bookmarks", headers=headers),
            client.post(
                f"{settings.API_V1_STR}/user/bookmarks/{uuid.uuid4()}", headers=headers
            ),
        ]

        for response in responses:
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"]["code"], "AUTH_NOT_CONFIGURED")
        logger_mock.error.assert_called_once()
        get_jwks_client_mock.assert_not_called()

    @mock.patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", None)
    @mock.patch.object(keycloak.settings, "KEYCLOAK_URL", None)
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    async def test_no_bearer_token_keycloak_not_configured(
        self, resolve_anonymous_mock, *mocks
    ):
        """Sans token, l'absence de config Keycloak n'a pas d'effet"""
        user_id = uuid.uuid4()
        resolve_anonymous_mock.return_value = (user_id, uuid.uuid4())

        response = client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], str(user_id))

    @mock.patch("src.app.user.infra.queries.session_maker")
    async def test_get_user_bookmarks_user_not_found(self, session_maker_mock, *mocks):
        """Bookmarks pour user inexistant -> 404"""
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "11111111-1111-1111-1111-111111111111"
        response = client.get(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks",
            params={"user_id": user_id},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.user.infra.queries.session_maker")
    async def test_get_user_bookmarks_success_empty(self, session_maker_mock, *mocks):
        """Bookmarks existants -> liste vide si pas de bookmarks"""
        session = MagicMock()
        session.execute.return_value.first.return_value = MagicMock(id="user-1")
        session.execute.return_value.all.return_value = []
        session_maker_mock.return_value.__enter__.return_value = session

        cookie_client = TestClient(
            app, cookies={"x-session-id": "bdb62bb2-1fe5-4d14-92fd-60a041355aea"}
        )
        response = cookie_client.get(
            f"{settings.API_V1_STR}/user/bookmarks",
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"bookmarks": []})

    @mock.patch("src.app.user.api.router.run_in_threadpool")
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    async def test_add_user_bookmark_success(
        self, resolve_user_and_session_mock, run_in_threadpool_mock, *mocks
    ):
        """Ajout d'un bookmark - mocks only what is needed"""
        # Mock resolve_user_and_session to return user_id and session_id
        user_id = uuid.UUID("cfc8072c-a055-442a-9878-b5a73d9141b2")
        session_id = uuid.UUID("bdb62bb2-1fe5-4d14-92fd-60a041355aea")
        resolve_user_and_session_mock.return_value = (user_id, session_id)

        # Mock run_in_threadpool to simulate DB add_user_bookmark_sync
        document_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        run_in_threadpool_mock.return_value = document_id

        response = client.post(
            f"{settings.API_V1_STR}/user/bookmarks/{document_id}",
            headers={"X-API-Key": "test", "X-Session-Id": str(session_id)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"added": document_id})
        resolve_user_and_session_mock.assert_called_once()
        run_in_threadpool_mock.assert_called_once()

    @mock.patch("src.app.user.api.router.run_in_threadpool")
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    @mock.patch("src.app.user.api.router.resolve_keycloak_user_and_session")
    async def test_get_user_bookmarks_with_keycloak_token(
        self,
        resolve_keycloak_mock,
        resolve_anonymous_mock,
        run_in_threadpool_mock,
        *mocks,
    ):
        """Avec un token Keycloak, les bookmarks sont ceux du compte Keycloak"""
        user_id = uuid.uuid4()
        resolve_keycloak_mock.return_value = (user_id, uuid.uuid4())
        run_in_threadpool_mock.return_value = []

        response = client.get(
            f"{settings.API_V1_STR}/user/bookmarks",
            headers={"X-API-Key": "test", "Authorization": "Bearer a.b.c"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(resolve_keycloak_mock.call_args.kwargs["token"], "a.b.c")
        self.assertEqual(run_in_threadpool_mock.call_args.args[1], user_id)
        resolve_anonymous_mock.assert_not_called()

    @mock.patch("src.app.user.api.router.resolve_keycloak_user_and_session")
    async def test_add_user_bookmark_invalid_token(self, resolve_keycloak_mock, *mocks):
        """Token invalide -> 401, même avec un cookie de session"""
        resolve_keycloak_mock.side_effect = KeycloakTokenError("expired")

        response = client.post(
            f"{settings.API_V1_STR}/user/bookmarks/{uuid.uuid4()}",
            headers={
                "X-API-Key": "test",
                "Authorization": "Bearer a.b.c",
                "X-Session-Id": str(uuid.uuid4()),
            },
        )

        self.assertEqual(response.status_code, 401)

    async def test_delete_user_bookmarks_without_token_nor_session(self, *mocks):
        """Ni token ni cookie de session -> 401"""
        response = client.delete(
            f"{settings.API_V1_STR}/user/bookmarks",
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 401)

    @mock.patch("src.app.user.api.router.run_in_threadpool")
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    async def test_add_user_bookmark_user_not_found(
        self, resolve_user_and_session_mock, run_in_threadpool_mock, *mocks
    ):
        """Ajout d'un bookmark - mocks only what is needed"""
        # Mock resolve_user_and_session to return user_id and session_id
        user_id = uuid.UUID("cfc8072c-a055-442a-9878-b5a73d9141b2")
        session_id = uuid.UUID("bdb62bb2-1fe5-4d14-92fd-60a041355aea")

        # mock error for user not found
        resolve_user_and_session_mock.side_effect = UserNotFoundError("User not found")
        resolve_user_and_session_mock.return_value = (user_id, session_id)

        # Mock run_in_threadpool to simulate DB add_user_bookmark_sync
        document_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        run_in_threadpool_mock.return_value = document_id

        response = client.post(
            f"{settings.API_V1_STR}/user/bookmarks/{document_id}",
            headers={"X-API-Key": "test", "X-Session-Id": str(session_id)},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {"detail": {"message": "User not found", "code": "USER_NOT_FOUND"}},
        )
        resolve_user_and_session_mock.assert_called_once()
        run_in_threadpool_mock.assert_not_called()

    @mock.patch("src.app.user.api.router.run_in_threadpool")
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    async def test_add_user_institution_data_user_not_found(
        self, resolve_user_and_session_mock, run_in_threadpool_mock, *mocks
    ):
        """Ajout d'un bookmark - mocks only what is needed"""
        # Mock resolve_user_and_session to return user_id and session_id
        user_id = uuid.UUID("cfc8072c-a055-442a-9878-b5a73d9141b2")
        session_id = uuid.UUID("bdb62bb2-1fe5-4d14-92fd-60a041355aea")

        # mock error for user not found
        resolve_user_and_session_mock.side_effect = UserNotFoundError("User not found")
        resolve_user_and_session_mock.return_value = (user_id, session_id)

        # Mock run_in_threadpool to simulate DB add_user_bookmark_sync
        document_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        run_in_threadpool_mock.return_value = document_id

        response = client.post(
            f"{settings.API_V1_STR}/user/institution-data",
            json={"institution": "Test University", "role": "Student"},
            headers={"X-API-Key": "test", "X-Session-Id": str(session_id)},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {"detail": {"message": "User not found", "code": "USER_NOT_FOUND"}},
        )
        resolve_user_and_session_mock.assert_called_once()
        run_in_threadpool_mock.assert_not_called()

    @mock.patch("src.app.user.api.router.run_in_threadpool")
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    async def test_add_user_institution_data_success(
        self, resolve_user_and_session_mock, run_in_threadpool_mock, *mocks
    ):
        """Ajout d'un institution data - mocks only what is needed"""
        # Mock resolve_user_and_session to return user_id and session_id
        user_id = uuid.UUID("cfc8072c-a055-442a-9878-b5a73d9141b2")
        session_id = uuid.UUID("bdb62bb2-1fe5-4d14-92fd-60a041355aea")
        resolve_user_and_session_mock.return_value = (user_id, session_id)

        # Mock run_in_threadpool to simulate DB add_institution_data_to_user_sync
        document_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        run_in_threadpool_mock.return_value = document_id

        response = client.post(
            f"{settings.API_V1_STR}/user/institution-data",
            json={"institution": "Test University", "role": "Student"},
            headers={"X-API-Key": "test", "X-Session-Id": str(session_id)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "message": "Institution data added to user",
                "institution": "Test University",
                "role": "Student",
            },
        )
        resolve_user_and_session_mock.assert_called_once()
        run_in_threadpool_mock.assert_called_once()


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
@mock.patch("src.app.user.api.router.sign_up_user")
class SignUpApiTests(unittest.TestCase):
    url = f"{settings.API_V1_STR}/user/signup"

    def test_signup_new_email_is_accepted(self, sign_up_mock):
        """Nouvel email -> 202 neutre, l'email est normalisé et la session transmise"""
        session_id = uuid.uuid4()
        sign_up_mock.return_value = uuid.uuid4()

        response = client.post(
            f"{self.url}?referer=welearn",
            json={"email": "  Jane@Example.org ", "first_name": "Jane"},
            headers={"X-API-Key": "test"},
            cookies={"x-session-id": str(session_id)},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"message": SIGNUP_ACCEPTED_MESSAGE})
        sign_up_mock.assert_called_once_with(
            email="jane@example.org",
            first_name="Jane",
            last_name=None,
            session_uuid=session_id,
            referer="welearn",
            locale=None,
        )

    def test_signup_locale_is_normalized(self, sign_up_mock):
        """Locale -> langue supportée par le realm, sinon ignorée (pas de 422)"""
        cases = [
            ("fr", "fr"),
            ("FR-fr", "fr"),
            ("en_GB", "en"),
            ("de", None),
            ("", None),
        ]
        for given, expected in cases:
            sign_up_mock.reset_mock()
            response = client.post(
                self.url,
                json={
                    "email": f"locale-{given or 'empty'}@example.org",
                    "locale": given,
                },
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 202, given)
            self.assertEqual(sign_up_mock.call_args.kwargs["locale"], expected, given)

    def test_signup_existing_email_gets_the_same_answer(self, sign_up_mock):
        """Email déjà inscrit -> même 202 et même corps qu'un nouvel email"""
        sign_up_mock.return_value = uuid.uuid4()
        new = client.post(
            self.url, json={"email": "new@example.org"}, headers={"X-API-Key": "test"}
        )
        sign_up_mock.return_value = None
        existing = client.post(
            self.url, json={"email": "old@example.org"}, headers={"X-API-Key": "test"}
        )

        self.assertEqual(existing.status_code, new.status_code)
        self.assertEqual(existing.status_code, 202)
        self.assertEqual(existing.json(), new.json())
        self.assertNotIn("user_id", existing.json())

    def test_signup_rate_limited_per_email(self, sign_up_mock):
        """Trop de tentatives pour un même email -> 429 RATE_LIMITED + Retry-After"""
        for _ in range(SIGNUP_RATE_LIMIT_PER_EMAIL):
            response = client.post(
                self.url,
                json={"email": "Jane@example.org"},
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 202)

        # same address once normalised
        response = client.post(
            self.url, json={"email": " jane@EXAMPLE.org"}, headers={"X-API-Key": "test"}
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "RATE_LIMITED")
        self.assertGreater(int(response.headers["retry-after"]), 0)
        self.assertEqual(sign_up_mock.call_count, SIGNUP_RATE_LIMIT_PER_EMAIL)

        other = client.post(
            self.url, json={"email": "john@example.org"}, headers={"X-API-Key": "test"}
        )
        self.assertEqual(other.status_code, 202)

    def test_signup_rate_limited_per_ip(self, sign_up_mock):
        """Trop de tentatives depuis une même IP -> 429, quel que soit l'email"""
        for i in range(SIGNUP_RATE_LIMIT_PER_IP):
            response = client.post(
                self.url,
                json={"email": f"user{i}@example.org"},
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 202)

        response = client.post(
            self.url, json={"email": "last@example.org"}, headers={"X-API-Key": "test"}
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "RATE_LIMITED")
        self.assertIn("retry-after", response.headers)
        self.assertEqual(sign_up_mock.call_count, SIGNUP_RATE_LIMIT_PER_IP)

    def test_signup_keycloak_unavailable(self, sign_up_mock):
        """Keycloak injoignable -> 503"""
        sign_up_mock.side_effect = KeycloakUnavailableError("timeout")

        response = client.post(
            self.url, json={"email": "jane@example.org"}, headers={"X-API-Key": "test"}
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "KEYCLOAK_UNAVAILABLE")

    def test_signup_keycloak_not_configured(self, sign_up_mock):
        """Client admin Keycloak non configuré -> 500"""
        sign_up_mock.side_effect = KeycloakConfigError("missing")

        response = client.post(
            self.url, json={"email": "jane@example.org"}, headers={"X-API-Key": "test"}
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "SIGNUP_DISABLED")

    def test_signup_invalid_email(self, sign_up_mock):
        """Email invalide -> 422, Keycloak n'est pas appelé"""
        response = client.post(
            self.url, json={"email": "not-an-email"}, headers={"X-API-Key": "test"}
        )

        self.assertEqual(response.status_code, 422)
        sign_up_mock.assert_not_called()
