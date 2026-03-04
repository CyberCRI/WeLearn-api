import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from qdrant_client.models import ScoredPoint

from src.app.shared.utils.dependencies import get_settings
from src.app.models.documents import Document
from src.app.services.sql_db.queries import (
    get_current_data_collection_campaign,
    get_last_syllabus_conversation_id,
    get_last_syllabus_id_for_user,
    update_returned_document_click,
    update_syllabus_retrieved_status,
    write_chat_answer,
    write_filters_search,
    write_returned_docs,
    write_user_query,
)
from src.app.services.sql_db.queries_user import get_user_from_session_id
from src.app.services.tutor.models import SyllabusFeedback, TutorSyllabusRequest
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

_cache: dict[str, Any] = {"is_campaign_active": None, "expires": None}

# get from setting the starts with string
settings = get_settings()


class DataCollection:
    def __init__(self, origin: str):
        is_campaign_active = self.get_campaign_state()
        origin_settings = settings.DATA_COLLECTION_ORIGIN_PREFIX.strip()

        self.should_collect = origin.startswith(origin_settings) and is_campaign_active
        logger.info(
            "data_collection: origin=%s, origin_settings=%s, is_campaign=%s, should_collect=%s",
            origin,
            origin_settings,
            is_campaign_active,
            self.should_collect,
        )

    def get_campaign_state(
        self,
    ):
        """Returns True if a campaign is active, False otherwise."""

        now = datetime.now()
        if _cache["expires"] and now < _cache["expires"]:
            return _cache["is_campaign_active"] is not None

        campaign = get_current_data_collection_campaign()

        _cache["is_campaign_active"] = campaign and campaign.is_active
        _cache["expires"] = now + timedelta(hours=6)

        return _cache["is_campaign_active"]

    async def register_search_data(
        self,
        session_id: str | None,
        query: str,
        search_results: list[Document | ScoredPoint],
        sdg_filter: list[int] | None = None,
        corpora: list[str] | None = None,
        feature: str | None = "search",
    ) -> uuid.UUID | None:

        if not self.should_collect:
            logger.info("data_collection is not enabled.")
            return None

        logger.info("data_collection is enabled. Registering search data.")

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Session ID not found",
                    "code": "SESSION_ID_NOT_FOUND",
                },
            )

        user_id = await run_in_threadpool(
            get_user_from_session_id, uuid.UUID(session_id)
        )

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "code": "USER_NOT_FOUND",
                },
            )

        conversation_id, chat_msg_id = await run_in_threadpool(
            write_user_query, user_id, query, None, feature
        )

        await run_in_threadpool(write_returned_docs, chat_msg_id, search_results)

        if sdg_filter or corpora:
            await run_in_threadpool(
                write_filters_search, chat_msg_id, sdg_filter, corpora
            )

        return chat_msg_id

    async def register_syllabus_data(
        self,
        session_id: uuid.UUID | None,
        input_data: TutorSyllabusRequest | SyllabusFeedback,
        agent_answer: str,
        feature: Literal["syllabus_creation", "syllabus_feedback"],
    ) -> uuid.UUID | None:

        if not self.should_collect:
            logger.info("data_collection is not enabled.")
            return None

        logger.info("data_collection is enabled. Registering syllabus search data.")

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Session ID not found",
                    "code": "SESSION_ID_NOT_FOUND",
                },
            )

        user_id = await run_in_threadpool(get_user_from_session_id, session_id)

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "code": "USER_NOT_FOUND",
                },
            )

        conversation_id = None

        if feature == "syllabus_feedback":
            # get conversation id from the last syllabus creation message of the user
            conversation_id = await run_in_threadpool(
                get_last_syllabus_conversation_id, user_id
            )

        user_data = None

        if isinstance(input_data, TutorSyllabusRequest):
            user_data = {
                "course_title": input_data.course_title,
                "level": input_data.level,
                "duration": input_data.duration,
                "description": input_data.description,
            }
            user_data = json.dumps(user_data)
        elif isinstance(input_data, SyllabusFeedback):
            user_data = input_data.feedback

        conversation_id, chat_msg_id = await run_in_threadpool(
            write_user_query, user_id, user_data, conversation_id, feature
        )

        chat_msg_id = await run_in_threadpool(
            write_chat_answer,
            user_id,
            agent_answer,
            None,
            conversation_id,
            feature,
        )

        if input_data.documents:
            await run_in_threadpool(
                write_returned_docs,
                chat_msg_id,
                [doc for doc in input_data.documents],
                True,
            )

        return chat_msg_id

    async def register_syllabus_update(
        self, session_id: uuid.UUID | None, syllabus_content: str
    ) -> None:
        if not self.should_collect:
            logger.info("data_collection is not enabled.")
            return None

        logger.info("data_collection is enabled. Registering syllabus update data.")

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Session ID not found",
                    "code": "SESSION_ID_NOT_FOUND",
                },
            )

        user_id = await run_in_threadpool(get_user_from_session_id, session_id)

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "code": "USER_NOT_FOUND",
                },
            )

        conversation_id = await run_in_threadpool(
            get_last_syllabus_conversation_id, user_id
        )

        if not conversation_id:
            logger.warning(
                f"No conversation found for user {user_id} when registering syllabus update"
            )
            return None

        await run_in_threadpool(
            write_user_query,
            user_id,
            syllabus_content,
            conversation_id,
            "syllabus_user_update",
        )

    async def register_chat_data(
        self,
        session_id: str | None,
        user_query: str,
        conversation_id: uuid.UUID | None,
        answer_content: str,
        sources: list[Document],
        feature: str | None = "chat",
    ) -> tuple[uuid.UUID | None, uuid.UUID | None]:

        if not self.should_collect:
            logger.info("data_collection is not enabled.")
            return None, None

        logger.info("data_collection is enabled. Registering chat data.")

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Session ID not found",
                    "code": "SESSION_ID_NOT_FOUND",
                },
            )

        user_id = await run_in_threadpool(
            get_user_from_session_id, uuid.UUID(session_id)
        )

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "code": "USER_NOT_FOUND",
                },
            )

        conversation_id, chat_msg_id = await run_in_threadpool(
            write_user_query, user_id, user_query, conversation_id, feature
        )

        message_id = await run_in_threadpool(
            write_chat_answer,
            user_id,
            answer_content,
            sources,  # type: ignore
            conversation_id,
            feature,
        )

        return conversation_id, message_id

    async def register_document_click(
        self,
        doc_id: uuid.UUID,
        message_id: uuid.UUID,
    ) -> None:
        if not self.should_collect:
            logger.info("data_collection is not enabled.")
            return

        logger.info("data_collection is enabled. Registering document click.")

        await run_in_threadpool(update_returned_document_click, doc_id, message_id)

    async def register_syllabus_download(self, session_id: uuid.UUID | None) -> None:
        if not self.should_collect:
            logger.info("data_collection is not enabled.")
            return

        logger.info("data_collection is enabled. Registering syllabus download.")

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Session ID not found",
                    "code": "SESSION_ID_NOT_FOUND",
                },
            )

        user_id = await run_in_threadpool(get_user_from_session_id, session_id)

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "code": "USER_NOT_FOUND",
                },
            )

        # get last syllabus from the user and update it with the is retrieved event
        syllabus_id = await run_in_threadpool(get_last_syllabus_id_for_user, user_id)

        if not syllabus_id:
            logger.warning(
                f"No syllabus found for user {user_id} when registering download event"
            )
            return

        await run_in_threadpool(update_syllabus_retrieved_status, syllabus_id)


def get_data_collection_service(request: Request) -> DataCollection:
    origin = request.headers["origin"]
    stripped_origin = re.sub(r"https?://www\.|https?://", "", origin).strip("/")

    if stripped_origin is None:
        return DataCollection(origin="")
    return DataCollection(origin=stripped_origin)
