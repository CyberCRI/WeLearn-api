# src/app/user/infra/keycloak_admin.py
"""
Keycloak Admin REST API, used for sign-up.

The API authenticates as a confidential client (client credentials grant)
whose service account has the realm-management roles "view-users" and
"manage-users".
"""

import uuid

import httpx

from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakUnavailableError,
    get_issuer,
)

logger = logger_utils(__name__)
settings = get_settings()

# the user verifies their email and sets a password from the link Keycloak sends
SIGNUP_ACTIONS = ["VERIFY_EMAIL", "UPDATE_PASSWORD"]
SIGNUP_LINK_LIFESPAN_SECONDS = 24 * 3600


class KeycloakUserExistsError(ValueError):
    """Raised when a Keycloak account already exists for an email"""


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=10)


def get_admin_base_url() -> str:
    issuer = get_issuer()
    if (
        not settings.KEYCLOAK_ADMIN_CLIENT_ID
        or not settings.KEYCLOAK_ADMIN_CLIENT_SECRET
    ):
        raise KeycloakConfigError(
            "KEYCLOAK_ADMIN_CLIENT_ID and KEYCLOAK_ADMIN_CLIENT_SECRET must be set"
        )
    # issuer is "<url>/realms/<realm>", the admin API is "<url>/admin/realms/<realm>"
    return issuer.replace("/realms/", "/admin/realms/", 1)


def _raise_for_status(response: httpx.Response, action: str) -> None:
    if response.is_success:
        return
    logger.error(
        "Keycloak admin call failed. action=%s status=%s body=%s",
        action,
        response.status_code,
        response.text,
    )
    raise KeycloakUnavailableError(
        f"Keycloak {action} failed with status {response.status_code}"
    )


async def _request(client: httpx.AsyncClient, action: str, method: str, url: str, **kw):
    try:
        return await client.request(method, url, **kw)
    except httpx.RequestError as e:
        # timeouts have an empty message: keep the exception type
        raise KeycloakUnavailableError(
            f"Cannot reach Keycloak ({action}): {type(e).__name__} {e}"
        ) from e


async def get_admin_token(client: httpx.AsyncClient) -> str:
    response = await _request(
        client,
        "token",
        "POST",
        f"{get_issuer()}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": settings.KEYCLOAK_ADMIN_CLIENT_ID,
            "client_secret": settings.KEYCLOAK_ADMIN_CLIENT_SECRET,
        },
    )
    _raise_for_status(response, "token")
    return response.json()["access_token"]


async def find_user_id_by_email(email: str) -> uuid.UUID | None:
    base_url = get_admin_base_url()
    async with _http_client() as client:
        token = await get_admin_token(client)
        response = await _request(
            client,
            "find user",
            "GET",
            f"{base_url}/users",
            params={"email": email, "exact": "true"},
            headers={"Authorization": f"Bearer {token}"},
        )
    _raise_for_status(response, "find user")
    users = response.json()
    return uuid.UUID(users[0]["id"]) if users else None


async def create_user(
    email: str, first_name: str | None = None, last_name: str | None = None
) -> uuid.UUID:
    """Create an enabled user with no password; raises if the email is taken."""
    base_url = get_admin_base_url()
    payload = {
        "username": email,
        "email": email,
        "enabled": True,
        "emailVerified": False,
        "requiredActions": SIGNUP_ACTIONS,
    }
    if first_name:
        payload["firstName"] = first_name
    if last_name:
        payload["lastName"] = last_name

    async with _http_client() as client:
        token = await get_admin_token(client)
        response = await _request(
            client,
            "create user",
            "POST",
            f"{base_url}/users",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
    if response.status_code == 409:
        # another request created it since the lookup
        raise KeycloakUserExistsError(f"Keycloak user already exists for {email}")
    _raise_for_status(response, "create user")

    # Keycloak returns the new user's URL, ".../users/<id>", in Location
    location = response.headers.get("Location", "")
    try:
        return uuid.UUID(location.rstrip("/").rsplit("/", 1)[-1])
    except ValueError as e:
        raise KeycloakUnavailableError(
            f"Unexpected Location header from Keycloak: {location!r}"
        ) from e


async def send_signup_email(keycloak_id: uuid.UUID) -> None:
    """Ask Keycloak to email the link to verify the email and set a password."""
    base_url = get_admin_base_url()
    params: dict[str, str | int] = {"lifespan": SIGNUP_LINK_LIFESPAN_SECONDS}
    if settings.KEYCLOAK_CLIENT_ID and settings.KEYCLOAK_SIGNUP_REDIRECT_URI:
        params["client_id"] = settings.KEYCLOAK_CLIENT_ID
        params["redirect_uri"] = settings.KEYCLOAK_SIGNUP_REDIRECT_URI

    async with _http_client() as client:
        token = await get_admin_token(client)
        response = await _request(
            client,
            "send signup email",
            "PUT",
            f"{base_url}/users/{keycloak_id}/execute-actions-email",
            params=params,
            json=SIGNUP_ACTIONS,
            headers={"Authorization": f"Bearer {token}"},
        )
    _raise_for_status(response, "send signup email")
