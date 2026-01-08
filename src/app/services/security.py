# src/app/services/security.py

import hashlib

from fastapi import HTTPException, Security, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from sqlalchemy.sql import select
from welearn_database.data.models import APIKeyManagement

from src.app.services.sql_service import session_maker
from src.app.utils.logger import logger as logger_utils

api_key_header = APIKeyHeader(name="X-API-Key")
logger = logger_utils(__name__)


def check_api_key_sync(api_key: str) -> bool:
    digest = hashlib.sha256(api_key.encode()).digest()
    statement = select(APIKeyManagement.digest, APIKeyManagement.is_active).where(
        APIKeyManagement.digest == digest
    )
    with session_maker() as s:
        keys = s.execute(statement).first()

    if not keys:
        return False

    return keys.is_active


async def get_user(api_key_header: str = Security(api_key_header)):
    is_valid = await run_in_threadpool(check_api_key_sync, api_key_header)
    if is_valid:
        return "ok"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
    )
