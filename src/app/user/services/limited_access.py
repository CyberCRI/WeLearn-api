"""
Time-limited accounts.

Users who log in through a university (Renater) keep their access. Those who
log in with Google, or with an email and password, get LIMITED_ACCESS_DAYS
from their first login. The end date is the Keycloak user attribute
"welearn_access_until" (an ISO date, last day included): an admin ends the
access early by setting a past date, or extends it with a later one, from the
Keycloak admin console. Once it is past, account tokens answer 403
ACCESS_EXPIRED and the client goes back to guest mode.
"""

import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone

from src.app.shared.domain.exceptions import AccessExpiredError
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.domain.models import AccountAccessResponse
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakUnavailableError,
    KeycloakUser,
)
from src.app.user.infra.keycloak_admin import set_user_attribute_if_missing

logger = logger_utils(__name__)
settings = get_settings()

ACCESS_UNTIL_ATTRIBUTE = "welearn_access_until"

# The token only carries the end date once it is refreshed after the first
# login. Until then the date written to Keycloak is remembered here, so the
# requests in between do not each call the admin API.
FIRST_LOGIN_CACHE_SECONDS = 15 * 60
_first_login_dates: dict[uuid.UUID, tuple[str, float]] = {}
_first_login_lock = threading.Lock()


def today() -> date:
    return datetime.now(timezone.utc).date()


def is_limited(user: KeycloakUser) -> bool:
    """Google (or another listed IdP) and email/password logins are limited."""
    if user.identity_provider is None:
        return True
    return user.identity_provider in settings.LIMITED_ACCESS_IDENTITY_PROVIDERS


def is_time_limited(user: KeycloakUser) -> bool:
    """Whether the account's access is time-limited on this server."""
    return settings.LIMITED_ACCESS_ENABLED and is_limited(user)


def parse_last_day(access_until: str) -> date | None:
    try:
        return date.fromisoformat(access_until.strip()[:10])
    except ValueError:
        logger.error("Unreadable %s: %r", ACCESS_UNTIL_ATTRIBUTE, access_until)
        return None


def has_expired(access_until: str) -> bool:
    """
    Whether the last day of access is past. An unreadable date counts as
    expired: it was edited by hand and access is not given by mistake.
    """
    last_day = parse_last_day(access_until)
    return last_day is None or today() > last_day


def _cached_first_login_date(keycloak_id: uuid.UUID) -> str | None:
    with _first_login_lock:
        cached = _first_login_dates.get(keycloak_id)
        if cached is None:
            return None
        access_until, stored_at = cached
        if time.monotonic() - stored_at > FIRST_LOGIN_CACHE_SECONDS:
            del _first_login_dates[keycloak_id]
            return None
        return access_until


def _cache_first_login_date(keycloak_id: uuid.UUID, access_until: str) -> None:
    with _first_login_lock:
        _first_login_dates[keycloak_id] = (access_until, time.monotonic())


async def _start_access(user: KeycloakUser) -> str | None:
    """
    Store the end date of a first limited login in Keycloak, or return the
    one already stored. None when Keycloak cannot be reached: the request goes
    through and a later one stores the date.
    """
    cached = _cached_first_login_date(user.keycloak_id)
    if cached:
        return cached

    access_until = (today() + timedelta(days=settings.LIMITED_ACCESS_DAYS)).isoformat()
    try:
        stored = await set_user_attribute_if_missing(
            user.keycloak_id, ACCESS_UNTIL_ATTRIBUTE, access_until
        )
    except (KeycloakUnavailableError, KeycloakConfigError) as e:
        logger.error(
            "Cannot store %s. keycloak_id=%s error=%s",
            ACCESS_UNTIL_ATTRIBUTE,
            user.keycloak_id,
            e,
        )
        return None

    if stored == access_until:
        logger.info(
            "Started limited access. keycloak_id=%s identity_provider=%s until=%s",
            user.keycloak_id,
            user.identity_provider,
            access_until,
        )
    _cache_first_login_date(user.keycloak_id, stored)
    return stored


async def check_limited_access(user: KeycloakUser) -> str | None:
    """
    Raise AccessExpiredError for a limited account past its end date. On the
    first limited login (no date in the token yet), the date is stored.
    Returns the end date of a limited account, None when the account is not
    limited or the date could not be stored.
    """
    if not is_time_limited(user):
        return None

    access_until = user.access_until or await _start_access(user)
    if access_until and has_expired(access_until):
        logger.info(
            "Limited access ended. keycloak_id=%s until=%s",
            user.keycloak_id,
            access_until,
        )
        raise AccessExpiredError(f"The access of this account ended on {access_until}")
    return access_until


async def get_account_access(user: KeycloakUser) -> AccountAccessResponse:
    """The account's access, for the client to tell the user how long it lasts."""
    access_until = await check_limited_access(user)
    if not is_time_limited(user):
        return AccountAccessResponse(limited=False)

    last_day = parse_last_day(access_until) if access_until else None
    if last_day is None:
        return AccountAccessResponse(limited=True)
    return AccountAccessResponse(
        limited=True,
        access_until=last_day,
        days_left=(last_day - today()).days + 1,
    )
