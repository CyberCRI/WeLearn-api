"""
Authentication events (sign-ups, new accounts, logins, logouts), read by
Grafana. A user is only known by its WeLearn id: no email, name or username is
stored.

Recording never breaks authentication: a failure is logged and swallowed, and
the request goes on.
"""

from contextlib import contextmanager
from typing import Iterator
from uuid import UUID

from fastapi.concurrency import run_in_threadpool
from welearn_database.data.enumeration import AuthEventType

from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakTokenError,
    KeycloakUnavailableError,
    KeycloakUser,
    decode_keycloak_token,
)
from src.app.user.infra.queries import (
    add_auth_event_sync,
    get_keycloak_user_and_session_sync,
)
from src.app.user.services.limited_access import is_time_limited

logger = logger_utils(__name__)


@contextmanager
def _not_breaking_auth(event_type: AuthEventType) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # the error's type only: its message may carry ids or SQL parameters
        logger.error(
            "Auth event not recorded. event_type=%s error=%s",
            event_type,
            type(e).__name__,
        )


async def record_auth_event(
    event_type: AuthEventType,
    user_id: UUID,
    account: KeycloakUser,
    session_id: UUID | None = None,
) -> None:
    """
    Record an event of the account's WeLearn user. A login is recorded once
    per Keycloak session, and not at all without one.
    """
    keycloak_session_id = None
    if event_type == AuthEventType.LOGIN:
        keycloak_session_id = account.keycloak_session_id
        if not keycloak_session_id:
            # without it, every page load would count as a login
            logger.debug("Token without sid, login not recorded. user_id=%s", user_id)
            return

    with _not_breaking_auth(event_type):
        inserted = await run_in_threadpool(
            add_auth_event_sync,
            user_id,
            event_type,
            session_id,
            account.identity_provider,
            is_time_limited(account),
            keycloak_session_id,
        )
        if not inserted:
            logger.debug("Login already recorded. user_id=%s", user_id)


async def record_logout(token: str, session_id: UUID | None) -> None:
    """
    Record the logout of the token's account, with the session being ended
    when it is the account's.
    """
    with _not_breaking_auth(AuthEventType.LOGOUT):
        try:
            account = await run_in_threadpool(decode_keycloak_token, token)
        except (KeycloakTokenError, KeycloakUnavailableError, KeycloakConfigError):
            logger.info("Logout not recorded, the token could not be verified")
            return

        user_id, session_id = await run_in_threadpool(
            get_keycloak_user_and_session_sync, account.keycloak_id, session_id
        )
        if user_id is None:
            logger.info("Logout not recorded, WeLearn does not know this account")
            return
        await record_auth_event(AuthEventType.LOGOUT, user_id, account, session_id)
