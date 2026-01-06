from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from src.app.services.sql_service import register_endpoint
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class MonitorRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/v1/"):
            session_id = request.headers.get("X-Session-ID")
            if session_id:
                try:
                    await run_in_threadpool(
                        register_endpoint,
                        endpoint=request.url.path,
                        session_id=session_id,
                        http_code=200,
                    )
                except Exception as e:
                    logger.error(f"Failed to register endpoint {request.url.path}: {e}")
            else:
                logger.warning(
                    f"No X-Session-ID header provided for {request.url.path}"
                )

        response = await call_next(request)
        return response
