import hashlib
import uuid
from typing import Annotated

from fastapi import Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.sql import select

from src.app.models.db_models import APIKeyManagement
from src.app.services.sql_db import register_endpoint, session_maker
from src.app.utils.logger import logger as logger_utils

api_key_header = APIKeyHeader(name="X-API-Key")
logger = logger_utils(__name__)


def check_api_key(api_key: str) -> bool:
    digest = hashlib.sha256(api_key.encode()).digest()
    statement = select(APIKeyManagement.digest, APIKeyManagement.is_active).where(
        APIKeyManagement.digest == digest
    )
    with session_maker() as s:
        keys = s.execute(statement).first()

    if not keys:
        return False

    return keys.is_active


def get_user(api_key_header: str = Security(api_key_header)):
    if check_api_key(api_key_header):

        return "ok"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
    )


def monitot_requests(
    request: Request, X_Session_id: Annotated[uuid.UUID | None, Header()] = None
):
    if not X_Session_id:
        logger.warning("No X-Session-ID header provided")

        return "OK"

    register_endpoint(endpoint=request.url.path, session_id=X_Session_id, http_code=200)
    return "OK"
