import time
import unittest
import uuid
from unittest.mock import MagicMock, patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy.exc import IntegrityError
from welearn_database.data.models import InferredUser

from src.app.user.infra import keycloak
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakTokenError,
    KeycloakUnavailableError,
    decode_keycloak_token,
    extract_bearer_token,
)
from src.app.user.infra.queries import get_or_create_user_from_keycloak_sync

KEYCLOAK_URL = "https://auth.example.org"
KEYCLOAK_REALM = "welearn"
ISSUER = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"

PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def make_token(key=PRIVATE_KEY, algorithm="RS256", **overrides) -> str:
    now = int(time.time())
    payload = {
        "sub": str(uuid.uuid4()),
        "iss": ISSUER,
        "iat": now,
        "exp": now + 300,
        "azp": "welearn-front",
    }
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, key, algorithm=algorithm, headers={"kid": "k1"})


@patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", None)
@patch.object(keycloak.settings, "KEYCLOAK_REALM", KEYCLOAK_REALM)
@patch.object(keycloak.settings, "KEYCLOAK_URL", KEYCLOAK_URL)
@patch("src.app.user.infra.keycloak.get_jwks_client")
class TestDecodeKeycloakToken(unittest.TestCase):
    def _use_public_key(self, get_jwks_client_mock, key=PRIVATE_KEY):
        signing_key = MagicMock(key=key.public_key())
        get_jwks_client_mock.return_value.get_signing_key_from_jwt.return_value = (
            signing_key
        )

    def test_decode_valid_token(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        sub = uuid.uuid4()
        token = make_token(
            sub=str(sub),
            email="jane@example.org",
            preferred_username="jane",
            given_name="Jane",
            family_name="Doe",
            name="Jane Doe",
        )

        user = decode_keycloak_token(token)

        self.assertEqual(user.keycloak_id, sub)
        self.assertEqual(user.email, "jane@example.org")
        self.assertEqual(user.username, "jane")
        self.assertEqual(user.first_name, "Jane")
        self.assertEqual(user.last_name, "Doe")
        self.assertEqual(user.name, "Jane Doe")
        get_jwks_client_mock.assert_called_once_with(ISSUER)

    def test_wrong_signature(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token(key=OTHER_PRIVATE_KEY))

    def test_expired_token(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token(exp=int(time.time()) - 10))

    def test_wrong_issuer(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token(iss="https://evil.example.org/realms/x"))

    def test_missing_sub(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token(sub=None))

    def test_sub_not_uuid(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token(sub="not-a-uuid"))

    def test_symmetric_algorithm_rejected(self, get_jwks_client_mock):
        signing_key = MagicMock(key="a-shared-secret-of-at-least-32-bytes!!")
        get_jwks_client_mock.return_value.get_signing_key_from_jwt.return_value = (
            signing_key
        )
        token = make_token(
            key="a-shared-secret-of-at-least-32-bytes!!", algorithm="HS256"
        )
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(token)

    def test_client_id_mismatch(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", "welearn-front"):
            with self.assertRaises(KeycloakTokenError):
                decode_keycloak_token(make_token(azp="another-client"))
            # matching client is accepted
            decode_keycloak_token(make_token())

    def test_signing_key_not_found(self, get_jwks_client_mock):
        get_jwks_client_mock.return_value.get_signing_key_from_jwt.side_effect = (
            jwt.PyJWKClientError("Unable to find a signing key")
        )
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token())

    def test_malformed_token(self, get_jwks_client_mock):
        get_jwks_client_mock.return_value.get_signing_key_from_jwt.side_effect = (
            jwt.DecodeError("Not enough segments")
        )
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token("not-a-jwt")

    def test_keycloak_unreachable(self, get_jwks_client_mock):
        get_jwks_client_mock.return_value.get_signing_key_from_jwt.side_effect = (
            jwt.PyJWKClientConnectionError("timeout")
        )
        with self.assertRaises(KeycloakUnavailableError):
            decode_keycloak_token(make_token())

    def test_missing_config(self, get_jwks_client_mock):
        with patch.object(keycloak.settings, "KEYCLOAK_URL", None):
            with self.assertRaises(KeycloakConfigError):
                decode_keycloak_token(make_token())
        get_jwks_client_mock.assert_not_called()


class TestExtractBearerToken(unittest.TestCase):
    def _request(self, headers):
        request = MagicMock()
        request.headers = headers
        return request

    def test_bearer_token(self):
        request = self._request({"Authorization": "Bearer abc.def.ghi"})
        self.assertEqual(extract_bearer_token(request), "abc.def.ghi")

    def test_no_header(self):
        self.assertIsNone(extract_bearer_token(self._request({})))

    def test_other_scheme(self):
        request = self._request({"Authorization": "Basic dXNlcjpwYXNz"})
        self.assertIsNone(extract_bearer_token(request))

    def test_empty_token(self):
        request = self._request({"Authorization": "Bearer "})
        self.assertIsNone(extract_bearer_token(request))


class TestGetOrCreateKeycloakUser(unittest.IsolatedAsyncioTestCase):
    @patch("src.app.user.infra.keycloak.get_or_create_user_from_keycloak_sync")
    @patch("src.app.user.infra.keycloak.decode_keycloak_token")
    async def test_returns_user_id_and_claims(self, decode_mock, get_or_create_mock):
        keycloak_user = keycloak.KeycloakUser(keycloak_id=uuid.uuid4())
        decode_mock.return_value = keycloak_user
        user_id = uuid.uuid4()
        anonymous_user_id = uuid.uuid4()
        get_or_create_mock.return_value = user_id

        result_user_id, result_user = await keycloak.get_or_create_keycloak_user(
            "token", anonymous_user_id, "welearn"
        )

        self.assertEqual(result_user_id, user_id)
        self.assertEqual(result_user, keycloak_user)
        decode_mock.assert_called_once_with("token")
        get_or_create_mock.assert_called_once_with(
            keycloak_user.keycloak_id, anonymous_user_id, "welearn"
        )

    @patch("src.app.user.infra.keycloak.get_or_create_user_from_keycloak_sync")
    @patch("src.app.user.infra.keycloak.decode_keycloak_token")
    async def test_invalid_token_does_not_hit_db(self, decode_mock, get_or_create_mock):
        decode_mock.side_effect = KeycloakTokenError("bad token")
        with self.assertRaises(KeycloakTokenError):
            await keycloak.get_or_create_keycloak_user("bad-token")
        get_or_create_mock.assert_not_called()


@patch("src.app.user.infra.queries.session_maker")
class TestGetOrCreateUserFromKeycloakSync(unittest.TestCase):
    def _session(self, session_maker_mock, *results):
        session = MagicMock()
        session_maker_mock.return_value.__enter__.return_value = session
        session.execute.side_effect = list(results)
        return session

    def _result(self, first=None, scalar=None):
        result = MagicMock()
        result.first.return_value = first
        result.scalar_one_or_none.return_value = scalar
        result.scalar_one.return_value = scalar
        return result

    def test_existing_keycloak_user(self, session_maker_mock):
        user_id = uuid.uuid4()
        session = self._session(
            session_maker_mock, self._result(first=MagicMock(id=user_id))
        )

        result = get_or_create_user_from_keycloak_sync(uuid.uuid4(), uuid.uuid4())

        self.assertEqual(result, user_id)
        session.add.assert_not_called()

    def test_links_anonymous_user(self, session_maker_mock):
        keycloak_id = uuid.uuid4()
        anonymous_user = InferredUser(id=uuid.uuid4())
        session = self._session(
            session_maker_mock,
            self._result(first=None),
            self._result(scalar=anonymous_user),
        )

        result = get_or_create_user_from_keycloak_sync(keycloak_id, anonymous_user.id)

        self.assertEqual(result, anonymous_user.id)
        self.assertEqual(anonymous_user.keycloak_id, keycloak_id)
        session.add.assert_called_once_with(anonymous_user)
        session.commit.assert_called_once()

    def test_creates_new_user(self, session_maker_mock):
        keycloak_id = uuid.uuid4()
        session = self._session(session_maker_mock, self._result(first=None))

        get_or_create_user_from_keycloak_sync(keycloak_id, None, "welearn")

        new_user: InferredUser = session.add.call_args[0][0]
        self.assertEqual(new_user.keycloak_id, keycloak_id)
        self.assertEqual(new_user.origin_referrer, "welearn")
        session.commit.assert_called_once()

    def test_creates_new_user_when_anonymous_already_linked(self, session_maker_mock):
        keycloak_id = uuid.uuid4()
        session = self._session(
            session_maker_mock,
            self._result(first=None),
            self._result(scalar=None),
        )

        get_or_create_user_from_keycloak_sync(keycloak_id, uuid.uuid4())

        new_user: InferredUser = session.add.call_args[0][0]
        self.assertEqual(new_user.keycloak_id, keycloak_id)

    def test_concurrent_creation(self, session_maker_mock):
        existing_id = uuid.uuid4()
        session = self._session(
            session_maker_mock,
            self._result(first=None),
            self._result(scalar=existing_id),
        )
        session.commit.side_effect = IntegrityError("insert", {}, Exception())

        result = get_or_create_user_from_keycloak_sync(uuid.uuid4())

        self.assertEqual(result, existing_id)
        session.rollback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
