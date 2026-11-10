# src/app/user/infra/keycloak.py
"""
Keycloak JWT verification.

The front end logs users in with Keycloak and forwards the access token as
`Authorization: Bearer <token>`. The token is verified against the realm's
public keys (JWKS), then used to read the user's data and register them in
the WeLearn DB.
"""

import threading
import time
import uuid
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError

from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.infra.queries import get_or_create_user_from_keycloak_sync

logger = logger_utils(__name__)
settings = get_settings()

# asymmetric algorithms only: never accept HS* tokens signed with a shared secret
ALLOWED_ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "ES256", "ES384", "ES512"]

# a token signed with an unknown "kid" refetches the JWKS at most this often, so
# forged kids cannot make every request hit Keycloak
JWKS_REFETCH_INTERVAL_SECONDS = 60

# the missing-settings error is logged once per process, not on every request
_config_error_logged = False


class KeycloakTokenError(ValueError):
    """Raised when a Keycloak token is invalid, expired or malformed"""


class KeycloakUnavailableError(RuntimeError):
    """Raised when Keycloak's public keys cannot be fetched"""


class KeycloakConfigError(RuntimeError):
    """Raised when Keycloak settings are missing"""


class KeycloakUser(BaseModel):
    keycloak_id: uuid.UUID
    email: str | None = None
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    name: str | None = None


def get_issuer() -> str:
    if not settings.KEYCLOAK_URL or not settings.KEYCLOAK_REALM:
        raise KeycloakConfigError("KEYCLOAK_URL and KEYCLOAK_REALM must be set")
    return f"{settings.KEYCLOAK_URL.rstrip('/')}/realms/{settings.KEYCLOAK_REALM}"


def get_token_settings() -> tuple[str, str]:
    """
    Return the issuer and client id tokens are checked against. Raises
    KeycloakConfigError, logged once, when a setting is missing: the app still
    starts without them, but cannot accept bearer tokens.
    """
    global _config_error_logged
    try:
        issuer = get_issuer()
        if not settings.KEYCLOAK_CLIENT_ID:
            raise KeycloakConfigError("KEYCLOAK_CLIENT_ID must be set")
    except KeycloakConfigError as e:
        if not _config_error_logged:
            _config_error_logged = True
            logger.error(
                "Bearer tokens are rejected, Keycloak is not configured: %s", e
            )
        raise
    return issuer, settings.KEYCLOAK_CLIENT_ID


class RateLimitedJWKClient(jwt.PyJWKClient):
    """
    PyJWKClient whose refetches for unknown "kid"s are rate limited.

    PyJWKClient refetches the JWKS whenever a token's kid is not in its cache
    (key rotation). Here, an unknown kid refetches at most once every
    `refetch_interval` seconds; in between it fails without a network call.
    Lookups run under a lock (decoding runs in a threadpool): requests that
    waited on a refetch then find the new key in the cache.
    """

    def __init__(self, uri: str, refetch_interval: float, **kwargs: Any):
        self.refetch_interval = refetch_interval
        self._lookup_lock = threading.Lock()
        self._last_refetch: float | None = None
        self._fetches = 0
        super().__init__(uri, **kwargs)

    def fetch_data(self) -> Any:
        self._fetches += 1
        return super().fetch_data()

    def _may_refetch(self) -> bool:
        now = time.monotonic()
        if (
            self._last_refetch is not None
            and now - self._last_refetch < self.refetch_interval
        ):
            return False
        self._last_refetch = now
        return True

    def get_signing_key(self, kid: str) -> jwt.PyJWK:
        with self._lookup_lock:
            fetches = self._fetches
            signing_key = self.match_kid(self.get_signing_keys(), kid)
            # no refetch when the lookup above just fetched the JWKS (cold or
            # expired cache): the kid is not in the current key set
            if signing_key is None and self._fetches == fetches and self._may_refetch():
                signing_key = self.match_kid(self.get_signing_keys(refresh=True), kid)
            if signing_key is None:
                raise jwt.PyJWKClientError(
                    f'Unable to find a signing key that matches: "{kid}"'
                )
            return signing_key


@lru_cache
def get_jwks_client(issuer: str) -> RateLimitedJWKClient:
    # the key set is cached for 5 minutes and known kids for good; an unknown
    # kid (key rotation) refetches it, at most once per JWKS_REFETCH_INTERVAL_SECONDS
    return RateLimitedJWKClient(
        f"{issuer}/protocol/openid-connect/certs",
        refetch_interval=JWKS_REFETCH_INTERVAL_SECONDS,
        cache_keys=True,
        timeout=10,
    )


def decode_keycloak_token(token: str) -> KeycloakUser:
    """
    Verify a Keycloak JWT (signature, issuer, expiry, authorized party and, if
    KEYCLOAK_AUDIENCE is set, audience) and return its user.
    """
    issuer, client_id = get_token_settings()

    try:
        signing_key = get_jwks_client(issuer).get_signing_key_from_jwt(token)
    except jwt.PyJWKClientConnectionError as e:
        raise KeycloakUnavailableError(f"Cannot fetch Keycloak keys: {e}") from e
    except (jwt.PyJWKClientError, jwt.DecodeError) as e:
        raise KeycloakTokenError(f"Cannot find signing key for token: {e}") from e

    # Keycloak access tokens' "aud" is usually "account" (the client is in
    # "azp"), so it is only checked when an audience mapper sets one
    audience = settings.KEYCLOAK_AUDIENCE
    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALLOWED_ALGORITHMS,
            issuer=issuer,
            audience=audience,
            options={
                "verify_aud": bool(audience),
                "require": ["exp", "iat", "iss", "sub"],
            },
        )
    except jwt.InvalidTokenError as e:
        raise KeycloakTokenError(f"Invalid token: {e}") from e

    # tokens from the realm's other clients are rejected
    if claims.get("azp") != client_id:
        raise KeycloakTokenError(
            f"Token issued for another client: {claims.get('azp')}"
        )

    try:
        return KeycloakUser(
            keycloak_id=claims["sub"],
            email=claims.get("email"),
            username=claims.get("preferred_username"),
            first_name=claims.get("given_name"),
            last_name=claims.get("family_name"),
            name=claims.get("name"),
        )
    except ValidationError as e:
        raise KeycloakTokenError(f"Invalid 'sub' claim: {e}") from e


def extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization")
    if not auth:
        return None
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def get_or_create_keycloak_user(
    token: str,
    anonymous_user_id: uuid.UUID | None = None,
    referer: str | None = None,
) -> tuple[uuid.UUID, KeycloakUser]:
    """
    Verify the token and return the WeLearn user id matching its Keycloak id.
    If none exists, the anonymous user (if any) is linked to it, otherwise a
    new user is created.
    """
    # JWKS fetch is blocking I/O
    keycloak_user = await run_in_threadpool(decode_keycloak_token, token)
    user_id = await run_in_threadpool(
        get_or_create_user_from_keycloak_sync,
        keycloak_user.keycloak_id,
        anonymous_user_id,
        referer,
    )
    logger.info(
        "Resolved keycloak user. keycloak_id=%s user_id=%s",
        keycloak_user.keycloak_id,
        user_id,
    )
    return user_id, keycloak_user
