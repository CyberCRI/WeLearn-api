from typing import Optional
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.infra.keycloak import get_or_create_keycloak_user
from src.app.user.infra.queries import (
    get_or_create_session_sync,
    get_or_create_user_sync,
    get_user_for_session_renewal,
    get_user_from_session_id,
)

logger = logger_utils(__name__)


async def resolve_user_and_session(
    session_uuid: Optional[UUID],
    host: str,
    referer: Optional[str],
    renew_expired_guest: bool = False,
) -> tuple[UUID, UUID]:
    """
    Resolve the user of the session cookie, or create a new guest.

    Only a valid session resolves, unless `renew_expired_guest` (the
    /user/user_and_session path, which sets the new cookie): an anonymous
    user's expired session then resolves to the same user and is rotated into
    a new session, so a guest keeps their user while the cookie lives.
    """
    lookup = (
        get_user_for_session_renewal
        if renew_expired_guest
        else get_user_from_session_id
    )
    try:
        user_id = await run_in_threadpool(lookup, session_uuid)
    except SessionNotFoundError:
        # unknown, purged or expired session cookie: treat the request as a new guest
        logger.info("Session %s not found or expired, ignoring it", session_uuid)
        user_id = None

    if not user_id:
        logger.info("No user found. Creating new user and session.")
        user_id = await run_in_threadpool(get_or_create_user_sync, None, referer)

        session_uuid = await run_in_threadpool(
            get_or_create_session_sync, user_id, None, host, referer
        )
    else:
        logger.info(
            "Existing user found. user_id=%s session_id=%s",
            user_id,
            session_uuid,
        )
        # a new session replaces an expired one
        session_uuid = await run_in_threadpool(
            get_or_create_session_sync, user_id, session_uuid, host, referer
        )

    return user_id, session_uuid


async def resolve_keycloak_user_and_session(
    token: str,
    session_uuid: Optional[UUID],
    host: str,
    referer: Optional[str],
) -> tuple[UUID, UUID]:
    """
    Resolve the user from a Keycloak token. An anonymous user found through the
    session cookie is linked to the Keycloak account instead of being dropped.
    """
    anonymous_user_id = None
    if session_uuid:
        try:
            anonymous_user_id = await run_in_threadpool(
                get_user_from_session_id, session_uuid
            )
        except SessionNotFoundError:
            logger.info("Session %s not found, ignoring it", session_uuid)
            session_uuid = None

    user_id, _ = await get_or_create_keycloak_user(token, anonymous_user_id, referer)

    # reuses the session only if it belongs to this user and is still valid
    session_uuid = await run_in_threadpool(
        get_or_create_session_sync, user_id, session_uuid, host, referer
    )
    return user_id, session_uuid
