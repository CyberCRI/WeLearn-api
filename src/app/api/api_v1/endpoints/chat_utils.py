import json
import uuid
from typing import Any, AsyncGenerator, cast
from uuid import UUID

import psycopg
from fastapi import BackgroundTasks
from fastapi.encoders import jsonable_encoder
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import AsyncRowFactory, DictRow

from src.app.models import chat as models
from src.app.search.services.search import SearchService
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


def _format_sse_event(data: str) -> str:
    lines = data.splitlines()
    return "".join(f"data: {line}\n" for line in lines) + "\n"


async def _sse_wrap(stream: Any) -> AsyncGenerator[str, None]:
    async for chunk in stream:
        if isinstance(chunk, str):
            data = chunk
        elif isinstance(chunk, bytes):
            data = chunk.decode("utf-8", errors="replace")
        else:
            data = json.dumps(jsonable_encoder(chunk))
        yield _format_sse_event(data)


def _resolve_thread_id(thread_id: UUID | None) -> UUID:
    if thread_id:
        return thread_id

    logger.info("No thread_id provided. Generating new thread_id.")
    return uuid.uuid4()


def _update_agent_stream_state(
    chunk: dict[str, Any],
    current_final_content: str,
    current_docs: Any,
) -> tuple[str, Any]:
    status = chunk.get("status")
    docs = current_docs
    final_content = current_final_content

    if status == "processing" and chunk.get("docs"):
        docs = chunk["docs"]
    elif status == "streaming":
        final_content += cast(str, chunk.get("content", ""))
    elif status == "stop":
        stop_content = cast(str, chunk.get("content", ""))
        if stop_content:
            final_content = stop_content

    return final_content, docs


def _serialize_agent_stream_chunk(chunk: dict[str, Any]) -> str:
    payload = {
        "content": chunk.get("content"),
        "status": chunk.get("status"),
        "step": chunk.get("step"),
        "label": chunk.get("label"),
        "docs": chunk.get("docs"),
    }

    return json.dumps(jsonable_encoder(payload))


async def _stream_agent_with_memory(
    *,
    db_uri: str,
    async_dict_row_factory: AsyncRowFactory[DictRow],
    chatfactory: Any,
    body: models.AgentContext,
    sp: SearchService,
    background_tasks: BackgroundTasks,
    thread_id: UUID,
) -> AsyncGenerator[dict[str, Any], None]:
    async with await psycopg.AsyncConnection[DictRow].connect(
        db_uri,
        autocommit=True,
        prepare_threshold=0,
        row_factory=async_dict_row_factory,
    ) as conn:
        await conn.execute("SET SEARCH_PATH to agent_related")
        await conn.commit()

        memory = AsyncPostgresSaver(conn)
        stream = await chatfactory.agent_message(
            query=body.query,
            memory=memory,
            thread_id=thread_id,
            corpora=body.corpora,
            sdg_filter=body.sdg_filter,
            reasoning=body.reasoning,
            sp=sp,
            background_tasks=background_tasks,
            streamed_ans=True,
        )

        async for chunk in stream:
            yield chunk


def _build_final_stream_payload(
    *,
    final_content: str,
    docs: Any,
    thread_id: UUID,
) -> dict[str, Any]:
    return {
        "content": final_content,
        "status": "stop",
        "docs": docs,
        "thread_id": thread_id,
    }


async def _register_stream_chat_data(
    *,
    data_collection: Any,
    session_id: UUID | None,
    user_query: str,
    conversation_id: UUID,
    answer_content: str,
    sources: Any,
) -> Any:
    _, message_id = await data_collection.register_chat_data(
        session_id=session_id,
        user_query=user_query,
        conversation_id=conversation_id,
        answer_content=answer_content,
        sources=sources,
    )
    return message_id


async def _stream_agent_response(
    *,
    db_uri: str,
    async_dict_row_factory: AsyncRowFactory[DictRow],
    body: models.AgentContext,
    chatfactory: Any,
    sp: SearchService,
    background_tasks: BackgroundTasks,
    data_collection: Any,
    session_id: UUID | None,
    thread_id: UUID,
) -> AsyncGenerator[str, None]:
    final_content = ""
    docs = []
    has_streamed_content = False

    stream = _stream_agent_with_memory(
        db_uri=db_uri,
        async_dict_row_factory=async_dict_row_factory,
        chatfactory=chatfactory,
        body=body,
        sp=sp,
        background_tasks=background_tasks,
        thread_id=thread_id,
    )

    async for chunk in stream:
        final_content, docs = _update_agent_stream_state(chunk, final_content, docs)
        if chunk.get("status") == "streaming" and chunk.get("content"):
            has_streamed_content = True
        if chunk.get("status") == "stop":
            continue
        try:
            yield _format_sse_event(_serialize_agent_stream_chunk(chunk))
        except Exception as e:
            logger.error("Error while yielding chunk: %s", e)

    final_payload = _build_final_stream_payload(
        final_content=final_content,
        docs=docs,
        thread_id=thread_id,
    )

    if has_streamed_content:
        final_payload = {**final_payload, "content": ""}

    try:
        message_id = await _register_stream_chat_data(
            data_collection=data_collection,
            session_id=session_id,
            user_query=cast(str, body.query),
            conversation_id=thread_id,
            answer_content=final_content,
            sources=docs,
        )
        final_payload = {**final_payload, "message_id": message_id}
    except Exception as e:
        logger.error("Error while registering chat data: %s", e)

    yield _format_sse_event(json.dumps(jsonable_encoder(final_payload)))
