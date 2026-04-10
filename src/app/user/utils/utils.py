from typing import Optional
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from src.app.services.sql_db.queries_user import (
    get_or_create_session_sync,
    get_or_create_user_sync,
    get_user_from_session_id,
)
from src.app.utils.logger import logger as logger_utils

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
