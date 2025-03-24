import hashlib

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.sql import select

from src.app.models.db_models import APIKeyManagement
from src.app.services.sql_db import session_maker

api_key_header = APIKeyHeader(name="X-API-Key")


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
