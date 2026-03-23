from typing import Optional
from uuid import UUID

from fastapi import Request

from src.app.shared.domain.constants import SESSION_COOKIE_NAME
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


def extract_session_cookie(request: Request) -> Optional[UUID]:
    cookie_value = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie_value:
        return None

    try:
        return UUID(cookie_value)
    except ValueError:
        logger.warning("Invalid session cookie format: %s", cookie_value)
        return None


def extract_origin_from_request(request: Request) -> str:
    return request.headers.get("origin", "unknown")
