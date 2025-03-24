from typing import Optional, cast

import backoff
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import RateLimitError

from src.app.api.dependencies import get_settings
from src.app.models import chat as models
from src.app.services.abst_chat import AbstractChat, ChatFactory
from src.app.services.constants import subjects as subjectsDict
from src.app.services.exceptions import (
    EmptyQueryError,
    InvalidQuestionError,
    LanguageNotSupportedError,
    bad_request,
)
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()

chatfactory: AbstractChat = ChatFactory().create_chat("openai")
chatfactory.init_client()


def get_params(body: models.Context) -> models.ContextOut:
    body.sources = body.sources[:7]

    if not body.query or body.query == "":
        e = EmptyQueryError()
        return bad_request(message=e.message, msg_code=e.msg_code)

    return models.ContextOut(
        sources=body.sources,
        history=body.history or [],
        query=body.query,
        subject=body.subject,
    )


@router.post(
    "/reformulate/query",
    summary="Reformulate User Query",
    description="This endpoint reformulates the user's query in english and french based on the provided context and history.",
    response_model=models.ReformulatedQueryResponse,
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
async def q_and_a_reformulate(
    body: models.ContextOut = Depends(get_params),
):
    try:
        reformulated_query: models.ReformulatedQueryResponse = (
            await chatfactory.reformulate_user_query(
                query=body.query, history=body.history
            )
        )

        if reformulated_query.QUERY_STATUS == "INVALID":
            raise InvalidQuestionError()

        return reformulated_query

    except (LanguageNotSupportedError, InvalidQuestionError) as e:
        bad_request(message=e.message, msg_code=e.msg_code)
    except ValueError as e:
        logger.error("Error while reformulating the query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Something went wrong while reformulating the query",
                "code": "REFORMULATE_ERROR",
            },
        )


@router.post(
    "/reformulate/questions",
    summary="Formulates new questions",
    description="This endpoint formulates new questions based on the provided context and history. They can be use to suggest it to the user.",
    response_model=models.ReformulatedQuestionsResponse,
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
async def q_and_a_new_questions(body: models.ContextOut = Depends(get_params)):
    try:
        new_questions = await chatfactory.get_new_questions(
            query=body.query, history=body.history
        )

        return new_questions
    except LanguageNotSupportedError as e:
        bad_request(message=e.message, msg_code=e.msg_code)


@router.post(
    "/chat/rephrase",
    summary="Rephrases input query",
    description="this endpoint is used to rephrase based on the provided context and history. It is meant to be used for rephrasing the last chat answer",
    response_model=str,
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
async def q_and_a_rephrase(
    body: models.ContextOut = Depends(get_params),
) -> Optional[str]:
    try:
        content = await chatfactory.rephrase_message(
            docs=body.sources,
            message=body.query,
            history=body.history,
            subject=subjectsDict.get(body.subject, None),
        )

        return cast(str, content)
    except Exception as e:
        logger.error("Error while rephrasing the query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Something went wrong while rephrasing the query",
                "code": "REPHRASE_ERROR",
            },
        )


@router.post(
    "/chat/rephrase_stream",
    summary="Rephrases input query",
    description="this endpoint is used to rephrase based on the provided context and history. It is meant to be used for rephrasing the last chat answer. Streamed version",
    response_model=str,
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
async def q_and_a_rephrase_stream(
    body: models.ContextOut = Depends(get_params),
) -> StreamingResponse:
    try:
        content = await chatfactory.rephrase_message(
            docs=body.sources,
            message=body.query,
            history=body.history,
            subject=subjectsDict.get(body.subject, None),
            streamed_ans=True,
        )
    except Exception as e:
        logger.error("Error while rephrasing the query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Something went wrong while rephrasing the query",
                "code": "REPHRASE_ERROR",
            },
        )

    return StreamingResponse(
        content=content,
        media_type="text/event-stream",
    )


@router.post(
    "/chat/answer",
    summary="Chat Answer",
    description="This endpoint is used to get the answer to the user's query based on the provided context and history",
    response_model=str,
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
async def q_and_a_ans(
    body: models.ContextOut = Depends(get_params),
) -> Optional[str]:
    """_summary_

    Args:
        body (models.Context): list of sources used for completion
        query (str, optional): users prompt. Defaults to "How to promote sustainable
        agriculture?".

    Returns:
        str: openai chat completion content
    """

    try:
        content = await chatfactory.chat_message(
            query=body.query,
            history=body.history,
            docs=body.sources,
            subject=subjectsDict.get(body.subject, None),
        )
        return cast(str, content)
    except LanguageNotSupportedError as e:
        bad_request(message=e.message, msg_code=e.msg_code)


@router.post(
    "/stream",
    summary="Chat Answer",
    description="This endpoint is used to get the answer to the user's query based on the provided context and history. Streamed version",
    response_model=str,
    response_class=StreamingResponse,
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
async def q_and_a_stream(
    body: models.ContextOut = Depends(get_params),
) -> StreamingResponse:  # type: ignore
    try:
        logger.info("q_and_a_stream subject=%s", body.subject)

        content = await chatfactory.chat_message(
            query=body.query,
            history=body.history,
            docs=body.sources,
            subject=subjectsDict.get(body.subject, None),
            streamed_ans=True,
        )

        return StreamingResponse(
            content=content,
            media_type="text/event-stream",
        )
    except LanguageNotSupportedError as e:
        bad_request(message=e.message, msg_code=e.msg_code)
    except Exception as e:
        logger.error("Error while streaming the query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Something went wrong while streaming the query",
                "code": "STREAM_ERROR",
            },
        )
