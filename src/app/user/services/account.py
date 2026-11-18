"""
Account checks shared by the routes that read the Keycloak bearer token.

A guest has only a session cookie; an account sends `Authorization: Bearer
<Keycloak token>`. With GUEST_RESTRICTIONS_ENABLED, the account-only features
(bookmarks, bibliography export, syllabus download) answer 401 ACCOUNT_REQUIRED
to a guest, so the API enforces what the client only hides. The flag is off by
default: off, these routes behave as before (a cookie is enough).
"""

from contextlib import contextmanager
from typing import Any, Iterator

from fastapi import HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from src.app.shared.domain.responses import ErrorResponse
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakTokenError,
    KeycloakUnavailableError,
    KeycloakUser,
    decode_keycloak_token,
    extract_bearer_token,
)
from src.app.user.services.limited_access import check_limited_access

logger = logger_utils(__name__)
settings = get_settings()

ACCOUNT_REQUIRED_CODE = "ACCOUNT_REQUIRED"

# routes reading the optional `Authorization: Bearer <Keycloak token>` header
BEARER_TOKEN_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Invalid or expired token"},
    403: {
        "model": ErrorResponse,
        "description": (
            "The access of a time-limited account (Google or email/password "
            "login) has ended (code ACCESS_EXPIRED)"
        ),
    },
    503: {
        "model": ErrorResponse,
        "description": (
            "Keycloak unavailable, or bearer tokens not accepted because Keycloak "
            "is not configured (code AUTH_NOT_CONFIGURED)"
        ),
    },
}

# routes reserved to accounts when GUEST_RESTRICTIONS_ENABLED is on
ACCOUNT_REQUIRED_RESPONSES: dict[int | str, dict[str, Any]] = {
    **BEARER_TOKEN_RESPONSES,
    401: {
        "model": ErrorResponse,
        "description": (
            "No bearer token while guest restrictions are on (code "
            f"{ACCOUNT_REQUIRED_CODE}), or invalid or expired token"
        ),
    },
}


def guest_restrictions_enabled() -> bool:
    return settings.GUEST_RESTRICTIONS_ENABLED


def auth_not_configured() -> HTTPException:
    """503 for a bearer token while the KEYCLOAK_* settings are missing."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "message": "Authentication is not configured on this server",
            "code": "AUTH_NOT_CONFIGURED",
        },
    )


def account_required() -> HTTPException:
    """
    401 for a guest calling an account-only route. Its code tells the client
    to offer sign-up, unlike the bare 401 of an invalid token.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "message": "An account is required for this feature",
            "code": ACCOUNT_REQUIRED_CODE,
        },
    )


@contextmanager
def keycloak_errors_as_http() -> Iterator[None]:
    """Turn the Keycloak errors raised inside the block into HTTP errors."""
    try:
        yield
    except KeycloakTokenError as e:
        logger.warning("Invalid keycloak token: %s", e)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except KeycloakUnavailableError as e:
        logger.error("Keycloak unavailable: %s", e)
        raise HTTPException(status_code=503, detail="Keycloak unavailable")
    except KeycloakConfigError:
        # logged once by keycloak.get_token_settings
        raise auth_not_configured()


async def verify_account_token(request: Request) -> KeycloakUser | None:
    """
    The account of the request's bearer token, or None without a token. Only
    verifies the token: no user or session is created or linked. Raises
    AccessExpiredError (403) for a time-limited account past its end date.
    """
    token = extract_bearer_token(request)
    if not token:
        return None
    with keycloak_errors_as_http():
        # JWKS fetch is blocking I/O
        account = await run_in_threadpool(decode_keycloak_token, token)
    await check_limited_access(account)
    return account


async def require_account(request: Request) -> KeycloakUser | None:
    """
    Dependency of the account-only routes that do not need the user id.

    - guest restrictions off: no check, returns None;
    - on, without a bearer token: 401 ACCOUNT_REQUIRED;
    - on, with a token: the verified account (401 if the token is invalid).
    """
    if not guest_restrictions_enabled():
        return None
    account = await verify_account_token(request)
    if account is None:
        raise account_required()
    return account
