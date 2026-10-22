from typing import Optional
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.infra.keycloak import get_or_create_keycloak_user
from src.app.user.infra.queries import (
    get_or_create_session_sync,
    get_or_create_user_sync,
    get_user_from_session_id,
)

logger = logger_utils(__name__)


async def resolve_user_and_session(
    session_uuid: Optional[UUID],
    host: str,
    referer: Optional[str],
) -> tuple[UUID, UUID]:
    user_id = await run_in_threadpool(get_user_from_session_id, session_uuid)

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
