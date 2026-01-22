import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from src.app.api.dependencies import get_settings
from src.app.models.documents import Document
from src.app.services.sql_db.queries import (
    get_current_data_collection_campaign,
    update_returned_document_click,
    write_chat_answer,
    write_user_query,
)
from src.app.services.sql_db.queries_user import get_user_from_session_id
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

    async def register_chat_data(
        self,
        session_id: str | None,
        user_query: str,
        conversation_id: uuid.UUID | None,
        answer_content: str,
        sources: list[Document],
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

        conversation_id = await run_in_threadpool(
            write_user_query, user_id, user_query, conversation_id
        )

        message_id = await run_in_threadpool(
            write_chat_answer, user_id, answer_content, sources, conversation_id
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


def get_data_collection_service(request: Request) -> DataCollection:
    origin = request.headers["origin"]
    stripped_origin = re.sub(r"https?://www\.|https?://", "", origin).strip("/")
    print(f"Request host: {stripped_origin}")
    if stripped_origin is None:
        return DataCollection(origin="")
    return DataCollection(origin=stripped_origin)
