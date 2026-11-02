import uuid
from typing import Annotated, Dict, cast
from uuid import UUID

import backoff
import psycopg
from fastapi import APIRouter, BackgroundTasks, Body, Depends, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from openai import RateLimitError
from psycopg.rows import AsyncRowFactory, DictRow, dict_row

from src.app.chat.domain import models
from src.app.chat.services.abst_chat import get_chat_service
from src.app.chat.services.chat_utils import _resolve_thread_id, _stream_agent_response
from src.app.chat.services.helpers import latest_tool_docs, linkify_missing_citations
from src.app.data_collection.services.data_collection import get_data_collection_service
from src.app.search.services.search import SearchService, get_search_service
from src.app.shared.domain.exceptions import (
    EmptyQueryError,
    LanguageNotSupportedError,
    bad_request,
)
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as utils_logger
from src.app.shared.utils.requests import extract_session_cookie

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()


DB_URI = "postgresql://{user}:{password}@{host}:{port}/{database}".format(
    user=settings.PG_USER,
    password=settings.PG_PASSWORD,
    host=settings.PG_HOST,
    port=settings.PG_PORT,
    database=settings.PG_DATABASE,
)

# psycopg exposes dict_row with a BaseCursor annotation, while AsyncConnection.connect
# expects an async row factory type. Runtime is valid; cast keeps static typing happy.
ASYNC_DICT_ROW_FACTORY = cast(AsyncRowFactory[DictRow], dict_row)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _build_agent_trace_context(
    *,
    endpoint: str,
    session_id: UUID | None,
    thread_id: UUID,
    body: models.AgentContext,
) -> models.TraceContext:
    env = settings.ENV
    return {
        "endpoint": endpoint,
        "feature": "chat_agent",
        "environment": env,
        "session_id": str(session_id) if session_id else None,
        "thread_id": str(thread_id),
        "query_length": len(body.query or ""),
        "sdg_filter": body.sdg_filter,
        "corpora": list(body.corpora) if body.corpora else None,
    }


def get_agent_params(
    body: Annotated[
        models.AgentContext,
        Body(
            description="Agent request payload with query, optional thread, and filters."
        ),
    ],
) -> models.AgentContext:
    if not body.query or body.query == "":
        e = EmptyQueryError()
        return bad_request(message=e.message, msg_code=e.msg_code)

    return models.AgentContext(
        query=body.query,
        thread_id=body.thread_id,
        corpora=body.corpora,
        sdg_filter=body.sdg_filter,
    )


@router.get("/chat/history")
async def get_chat_history(
    thread_id: UUID,
    chatfactory=Depends(get_chat_service),
) -> list[Dict[str, str | list[Dict[str, str]] | None]]:
    if thread_id:
        async with await psycopg.AsyncConnection[DictRow].connect(
            DB_URI,
            autocommit=True,
            prepare_threshold=0,
            row_factory=ASYNC_DICT_ROW_FACTORY,
        ) as conn:
            await conn.execute("SET SEARCH_PATH to agent_related")
            await conn.commit()

            memory = AsyncPostgresSaver(conn)
            res = await chatfactory.agent_get_history(
                thread_id=thread_id, memory=memory
            )
            return res


@router.post(
    "/chat/agent_stream",
    summary="Agent Response Stream",
    description="This endpoint streams an agent response to the user's message and ends with the full response payload.",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=RateLimitError,
    logger=logger,
    max_tries=5,
    max_time=180,
    jitter=backoff.random_jitter,
    factor=2,
)
async def agent_stream_response(
    request: Request,
    background_tasks: BackgroundTasks,
    body: models.AgentContext = Depends(get_agent_params),
    chatfactory=Depends(get_chat_service),
    sp: SearchService = Depends(get_search_service),
    data_collection=Depends(get_data_collection_service),
) -> StreamingResponse:
    try:
        session_id = extract_session_cookie(request)
        thread_id = _resolve_thread_id(body.thread_id)
        trace_context = _build_agent_trace_context(
            endpoint="/api/v1/qna/chat/agent_stream",
            session_id=session_id,
            thread_id=thread_id,
            body=body,
        )

        if body.query is None:
            raise EmptyQueryError()

        return StreamingResponse(
            content=_stream_agent_response(
                db_uri=DB_URI,
                async_dict_row_factory=ASYNC_DICT_ROW_FACTORY,
                body=body,
                chatfactory=chatfactory,
                sp=sp,
                background_tasks=background_tasks,
                data_collection=data_collection,
                session_id=session_id,
                thread_id=thread_id,
                trace_context=trace_context,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    except LanguageNotSupportedError as e:
        bad_request(message=e.message, msg_code=e.msg_code)
        raise


@router.post(
    "/chat/agent",
    operation_id="chat_agent_response",
    summary="Agent Response",
    description="This endpoint is used to get an agent response to the user's message",
    response_model=models.AgentResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "new_thread": {
                            "summary": "Start a new agent thread",
                            "value": {
                                "query": "How can communities improve clean water access?",
                                "corpora": ["conversation"],
                                "sdg_filter": [6],
                            },
                        },
                        "existing_thread": {
                            "summary": "Continue an existing thread",
                            "value": {
                                "query": "Can you give 3 concrete actions?",
                                "thread_id": "6fca8d02-0bc2-48a5-9c95-4942ca57e651",
                                "corpora": ["conversation"],
                                "sdg_filter": [6],
                            },
                        },
                    }
                }
            }
        }
    },
)
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=RateLimitError,
    logger=logger,
    max_tries=5,
    max_time=180,
    jitter=backoff.random_jitter,
    factor=2,
)
async def agent_response(
    request: Request,
    background_tasks: BackgroundTasks,
    body: models.AgentContext = Depends(get_agent_params),
    chatfactory=Depends(get_chat_service),
    sp: SearchService = Depends(get_search_service),
    data_collection=Depends(get_data_collection_service),
):
    try:
        session_id = extract_session_cookie(request)

        thread_id = body.thread_id if body.thread_id else None

        if not thread_id:
            logger.info("No thread_id provided. Generating new thread_id.")
            thread_id = uuid.uuid4()

        trace_context = _build_agent_trace_context(
            endpoint="/api/v1/qna/chat/agent",
            session_id=session_id,
            thread_id=thread_id,
            body=body,
        )

        if body.query is None:
            raise EmptyQueryError()

        if thread_id:
            async with await psycopg.AsyncConnection[DictRow].connect(
                DB_URI,
                autocommit=True,
                prepare_threshold=0,
                row_factory=ASYNC_DICT_ROW_FACTORY,
            ) as conn:
                await conn.execute("SET SEARCH_PATH to agent_related")
                await conn.commit()

                memory = AsyncPostgresSaver(conn)

                res = await chatfactory.agent_message(
                    query=body.query,
                    memory=memory,
                    thread_id=thread_id,
                    corpora=body.corpora,
                    sdg_filter=body.sdg_filter,
                    sp=sp,
                    background_tasks=background_tasks,
                    trace_context=trace_context,
                )
        else:
            res = await chatfactory.agent_message(
                query=body.query,
                corpora=body.corpora,
                sdg_filter=body.sdg_filter,
                sp=sp,
                trace_context=trace_context,
            )

        docs = latest_tool_docs(res["messages"])
        content = linkify_missing_citations(
            cast(str, res["messages"][-1].content), docs or []
        )

        agent_ans = {
            "content": content,
            "docs": docs,
            "thread_id": thread_id,
        }

        try:
            _, message_id = await data_collection.register_chat_data(
                session_id=session_id,
                user_query=body.query,
                conversation_id=thread_id,
                answer_content=content,
                sources=docs,
            )

            agent_ans = {
                **agent_ans,
                "message_id": message_id,
                "thread_id": thread_id,
            }

            return agent_ans
        except Exception as e:
            logger.error("Error while registering chat data: %s", e)

        return agent_ans
    except LanguageNotSupportedError as e:
        bad_request(message=e.message, msg_code=e.msg_code)
