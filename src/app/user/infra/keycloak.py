# src/app/user/infra/keycloak.py
"""
Keycloak JWT verification.

The front end logs users in with Keycloak and forwards the access token as
`Authorization: Bearer <token>`. The token is verified against the realm's
public keys (JWKS), then used to read the user's data and register them in
the WeLearn DB.
"""

import uuid
from functools import lru_cache

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


@lru_cache
def get_jwks_client(issuer: str) -> jwt.PyJWKClient:
    # keys are cached in memory; an unknown "kid" triggers a refetch (key rotation)
    return jwt.PyJWKClient(
        f"{issuer}/protocol/openid-connect/certs", cache_keys=True, timeout=10
    )


def decode_keycloak_token(token: str) -> KeycloakUser:
    """Verify a Keycloak JWT (signature, issuer, expiry) and return its user."""
    issuer = get_issuer()

    try:
        signing_key = get_jwks_client(issuer).get_signing_key_from_jwt(token)
    except jwt.PyJWKClientConnectionError as e:
        raise KeycloakUnavailableError(f"Cannot fetch Keycloak keys: {e}") from e
    except (jwt.PyJWKClientError, jwt.DecodeError) as e:
        raise KeycloakTokenError(f"Cannot find signing key for token: {e}") from e

    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALLOWED_ALGORITHMS,
            issuer=issuer,
            # Keycloak access tokens' "aud" is usually "account", the client is in "azp"
            options={"verify_aud": False, "require": ["exp", "iat", "iss", "sub"]},
        )
    except jwt.InvalidTokenError as e:
        raise KeycloakTokenError(f"Invalid token: {e}") from e

    if settings.KEYCLOAK_CLIENT_ID and claims.get("azp") != settings.KEYCLOAK_CLIENT_ID:
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
