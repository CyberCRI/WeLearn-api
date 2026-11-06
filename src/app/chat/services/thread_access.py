"""
Ownership of chat threads.

A thread has no table of its own: its owner is written in the LangGraph
checkpoint metadata. `AbstractChat.agent_message` puts the caller's user id
under `OWNER_METADATA_KEY` in the run config metadata, and LangGraph copies it
into every checkpoint the run writes. Every thread access reads the latest
checkpoint and compares that owner with the caller.

The owner is a user id, not a session id, so a thread survives a session
renewal and follows a guest who signs in (their user is linked to the account).
"""

from typing import Optional
from uuid import UUID

import psycopg
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import AsyncRowFactory, DictRow

from src.app.shared.domain.exceptions import SessionNotFoundError, ThreadNotFoundError
from src.app.shared.utils.logger import logger as utils_logger
from src.app.shared.utils.requests import extract_session_cookie
from src.app.user.infra.queries import get_user_from_session_id

logger = utils_logger(__name__)

OWNER_METADATA_KEY = "owner_user_id"


async def resolve_caller_user_id(request: Request) -> Optional[UUID]:
    """
    The user of the request's session, or None when there is no session or it
    is unknown or expired. Nothing is created: a chat request only reads the
    session the client already has.
    """
    session_id = extract_session_cookie(request)
    if not session_id:
        return None
    try:
        return await run_in_threadpool(get_user_from_session_id, session_id)
    except SessionNotFoundError:
        logger.info("Chat request with unknown or expired session %s", session_id)
        return None


async def ensure_thread_access(
    memory: AsyncPostgresSaver,
    thread_id: UUID,
    user_id: Optional[UUID],
) -> None:
    """
    Raise ThreadNotFoundError unless the caller may use the thread.

    - no checkpoint: a new thread, allowed;
    - no owner in the metadata: a thread from before ownership was recorded,
      allowed (the caller's next message writes them as its owner);
    - an owner: allowed only for that user. A caller without a user cannot
      reach an owned thread.
    """
    checkpoint = await memory.aget_tuple(
        {"configurable": {"thread_id": str(thread_id)}}
    )
    if checkpoint is None:
        return

    owner = (checkpoint.metadata or {}).get(OWNER_METADATA_KEY)
    if owner is None:
        logger.info(
            "Thread without owner, access allowed. thread_id=%s user_id=%s",
            thread_id,
            user_id,
        )
        return

    if user_id is None or owner != str(user_id):
        logger.warning(
            "Thread owned by another user. thread_id=%s user_id=%s",
            thread_id,
            user_id,
        )
        raise ThreadNotFoundError()


async def check_thread_access(
    *,
    db_uri: str,
    async_dict_row_factory: AsyncRowFactory[DictRow],
    thread_id: UUID,
    user_id: Optional[UUID],
) -> None:
    """`ensure_thread_access` on its own connection, for callers without one."""
    async with await psycopg.AsyncConnection[DictRow].connect(
        db_uri,
        autocommit=True,
        prepare_threshold=0,
        row_factory=async_dict_row_factory,
    ) as conn:
        await conn.execute("SET SEARCH_PATH to agent_related")
        await ensure_thread_access(AsyncPostgresSaver(conn), thread_id, user_id)
