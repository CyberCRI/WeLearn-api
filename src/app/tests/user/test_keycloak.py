import threading
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
    JWKS_REFETCH_INTERVAL_SECONDS,
    KeycloakConfigError,
    KeycloakTokenError,
    KeycloakUnavailableError,
    RateLimitedJWKClient,
    decode_keycloak_token,
    extract_bearer_token,
)
from src.app.user.infra.queries import get_or_create_user_from_keycloak_sync

KEYCLOAK_URL = "https://auth.example.org"
KEYCLOAK_REALM = "welearn"
ISSUER = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"
CLIENT_ID = "welearn-front"
JWKS_URI = f"{ISSUER}/protocol/openid-connect/certs"

PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def make_token(key=PRIVATE_KEY, algorithm="RS256", kid="k1", **overrides) -> str:
    now = int(time.time())
    payload = {
        "sub": str(uuid.uuid4()),
        "iss": ISSUER,
        "iat": now,
        "exp": now + 300,
        "azp": CLIENT_ID,
    }
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, key, algorithm=algorithm, headers={"kid": kid})


def make_jwks(*kids: str) -> dict:
    """A JWKS publishing PRIVATE_KEY's public key under each kid."""
    keys = []
    for kid in kids:
        jwk = jwt.algorithms.RSAAlgorithm.to_jwk(PRIVATE_KEY.public_key(), as_dict=True)
        keys.append({**jwk, "kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": keys}


@patch.object(keycloak.settings, "KEYCLOAK_AUDIENCE", None)
@patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", CLIENT_ID)
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
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token(azp="another-client"))
        # matching client is accepted
        decode_keycloak_token(make_token())

    def test_missing_azp(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with self.assertRaises(KeycloakTokenError):
            decode_keycloak_token(make_token(azp=None))

    def test_audience_not_checked_when_unset(self, get_jwks_client_mock):
        """Keycloak access tokens' default "aud" ("account") is accepted"""
        self._use_public_key(get_jwks_client_mock)
        decode_keycloak_token(make_token(aud="account"))

    def test_audience_checked_when_set(self, get_jwks_client_mock):
        self._use_public_key(get_jwks_client_mock)
        with patch.object(keycloak.settings, "KEYCLOAK_AUDIENCE", "welearn-api"):
            with self.assertRaises(KeycloakTokenError):
                decode_keycloak_token(make_token(aud="account"))
            with self.assertRaises(KeycloakTokenError):
                decode_keycloak_token(make_token())  # no "aud" claim
            # "aud" may be a list containing the audience
            decode_keycloak_token(make_token(aud=["account", "welearn-api"]))
            decode_keycloak_token(make_token(aud="welearn-api"))

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
        for setting in ["KEYCLOAK_URL", "KEYCLOAK_REALM", "KEYCLOAK_CLIENT_ID"]:
            with self.subTest(setting=setting):
                with patch.object(keycloak.settings, setting, None):
                    with self.assertRaises(KeycloakConfigError):
                        decode_keycloak_token(make_token())
        get_jwks_client_mock.assert_not_called()

    def test_missing_config_logged_once(self, get_jwks_client_mock):
        with (
            patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", None),
            patch.object(keycloak, "_config_error_logged", False),
            patch.object(keycloak, "logger") as logger_mock,
        ):
            for _ in range(3):
                with self.assertRaises(KeycloakConfigError):
                    decode_keycloak_token(make_token())

        logger_mock.error.assert_called_once()
        self.assertIn("KEYCLOAK_CLIENT_ID", str(logger_mock.error.call_args))


@patch.object(jwt.PyJWKClient, "fetch_data")
@patch("src.app.user.infra.keycloak.time.monotonic")
class TestRateLimitedJWKClient(unittest.TestCase):
    """The JWKS endpoint is mocked by PyJWKClient.fetch_data."""

    def _client(self, monotonic_mock, fetch_data_mock, *kids):
        monotonic_mock.return_value = 1000.0
        fetch_data_mock.return_value = make_jwks(*kids)
        return RateLimitedJWKClient(
            JWKS_URI, refetch_interval=JWKS_REFETCH_INTERVAL_SECONDS, cache_keys=True
        )

    def test_known_kid_fetched_once(self, monotonic_mock, fetch_data_mock):
        jwks_client = self._client(monotonic_mock, fetch_data_mock, "k1")

        for _ in range(5):
            key = jwks_client.get_signing_key_from_jwt(make_token(kid="k1"))
            self.assertEqual(key.key_id, "k1")

        fetch_data_mock.assert_called_once()

    def test_burst_of_unknown_kids_fetches_once(self, monotonic_mock, fetch_data_mock):
        jwks_client = self._client(monotonic_mock, fetch_data_mock, "k1")
        jwks_client.get_signing_key("k1")
        self.assertEqual(fetch_data_mock.call_count, 1)

        for i in range(20):
            with self.assertRaises(jwt.PyJWKClientError):
                jwks_client.get_signing_key(f"forged-{i}")
        # one refetch for the first unknown kid, none for the others
        self.assertEqual(fetch_data_mock.call_count, 2)

        # a known kid is still served from the cache
        self.assertEqual(jwks_client.get_signing_key("k1").key_id, "k1")
        self.assertEqual(fetch_data_mock.call_count, 2)

    def test_unknown_kid_on_cold_cache_fetches_once(
        self, monotonic_mock, fetch_data_mock
    ):
        """The first fetch already has the latest keys: no immediate refetch"""
        jwks_client = self._client(monotonic_mock, fetch_data_mock, "k1")

        with self.assertRaises(jwt.PyJWKClientError):
            jwks_client.get_signing_key("forged")
        fetch_data_mock.assert_called_once()

    def test_refetch_allowed_after_interval(self, monotonic_mock, fetch_data_mock):
        """A rotated key is picked up once the interval has passed"""
        jwks_client = self._client(monotonic_mock, fetch_data_mock, "k1")
        jwks_client.get_signing_key("k1")
        with self.assertRaises(jwt.PyJWKClientError):
            jwks_client.get_signing_key("k2")  # refetch, k2 not published yet
        self.assertEqual(fetch_data_mock.call_count, 2)

        fetch_data_mock.return_value = make_jwks("k1", "k2")
        monotonic_mock.return_value += JWKS_REFETCH_INTERVAL_SECONDS - 1
        with self.assertRaises(jwt.PyJWKClientError):
            jwks_client.get_signing_key("k2")  # still within the interval
        self.assertEqual(fetch_data_mock.call_count, 2)

        monotonic_mock.return_value += 1
        self.assertEqual(jwks_client.get_signing_key("k2").key_id, "k2")
        self.assertEqual(fetch_data_mock.call_count, 3)

    def test_concurrent_unknown_kids_fetch_once(self, monotonic_mock, fetch_data_mock):
        jwks_client = self._client(monotonic_mock, fetch_data_mock, "k1")
        jwks_client.get_signing_key("k1")
        errors = []

        def lookup(kid):
            try:
                jwks_client.get_signing_key(kid)
            except jwt.PyJWKClientError as e:
                errors.append(e)

        threads = [
            threading.Thread(target=lookup, args=(f"forged-{i}",)) for i in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(errors), 20)
        self.assertEqual(fetch_data_mock.call_count, 2)

    def test_failed_refetch_counts(self, monotonic_mock, fetch_data_mock):
        """Keycloak down: unknown kids do not retry the fetch on every request"""
        jwks_client = self._client(monotonic_mock, fetch_data_mock, "k1")
        jwks_client.get_signing_key("k1")
        fetch_data_mock.side_effect = jwt.PyJWKClientConnectionError("timeout")

        with self.assertRaises(jwt.PyJWKClientConnectionError):
            jwks_client.get_signing_key("k2")
        with self.assertRaises(jwt.PyJWKClientError):
            jwks_client.get_signing_key("k2")
        self.assertEqual(fetch_data_mock.call_count, 2)

    @patch.object(keycloak.settings, "KEYCLOAK_AUDIENCE", None)
    @patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", CLIENT_ID)
    @patch.object(keycloak.settings, "KEYCLOAK_REALM", KEYCLOAK_REALM)
    @patch.object(keycloak.settings, "KEYCLOAK_URL", KEYCLOAK_URL)
    def test_decode_with_jwks(self, monotonic_mock, fetch_data_mock):
        jwks_client = self._client(monotonic_mock, fetch_data_mock, "k1")
        with patch.object(keycloak, "get_jwks_client", return_value=jwks_client):
            sub = uuid.uuid4()
            self.assertEqual(
                decode_keycloak_token(make_token(sub=str(sub))).keycloak_id, sub
            )
            # unknown kids -> invalid token (401), one refetch for the lot
            for kid in ["forged-1", "forged-2", "forged-3"]:
                with self.assertRaises(KeycloakTokenError):
                    decode_keycloak_token(make_token(kid=kid))
        self.assertEqual(fetch_data_mock.call_count, 2)

    def test_get_jwks_client(self, monotonic_mock, fetch_data_mock):
        keycloak.get_jwks_client.cache_clear()
        try:
            jwks_client = keycloak.get_jwks_client(ISSUER)
            self.assertIsInstance(jwks_client, RateLimitedJWKClient)
            self.assertEqual(jwks_client.uri, JWKS_URI)
            self.assertIs(keycloak.get_jwks_client(ISSUER), jwks_client)
        finally:
            keycloak.get_jwks_client.cache_clear()


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
        """
        An anonymous user already attached to another Keycloak account (e.g. a
        different email) is not taken over: a new user is created.
        """
        keycloak_id = uuid.uuid4()
        anonymous_user_id = uuid.uuid4()
        session = self._session(
            session_maker_mock,
            self._result(first=None),
            self._result(scalar=None),
        )

        get_or_create_user_from_keycloak_sync(keycloak_id, anonymous_user_id, "web")

        # the anonymous user is only looked up among users without a keycloak_id
        anonymous_query = str(session.execute.call_args_list[1][0][0])
        self.assertIn("inferred_user.keycloak_id IS NULL", anonymous_query)
        new_user: InferredUser = session.add.call_args[0][0]
        self.assertEqual(new_user.keycloak_id, keycloak_id)
        self.assertNotEqual(new_user.id, anonymous_user_id)
        self.assertEqual(new_user.origin_referrer, "web")

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
