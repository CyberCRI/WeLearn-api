from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware

from src.app.services.sql_service import register_endpoint
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class MonitorRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        session_id = request.headers.get("X-Session-ID")
        response = await call_next(request)

        if session_id and request.url.path.startswith("/api/v1/"):
            try:
                response.background = BackgroundTask(
                    register_endpoint, request.url.path, session_id, 200
                )
            except Exception as e:
                logger.error(f"Failed to register endpoint {request.url.path}: {e}")
        else:
            logger.warning(f"No X-Session-ID header provided for {request.url.path}")

        return response
