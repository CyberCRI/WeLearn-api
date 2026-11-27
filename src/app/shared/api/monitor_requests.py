"""
Records each /api/v1/ request of a session in user_related.endpoint_request, with
the status it was answered with and, for an error, its code.
"""

from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import IntegrityError
from starlette.middleware.base import BaseHTTPMiddleware

from src.app.shared.api.errors import noted_error_code
from src.app.shared.infra.queries import register_endpoint
from src.app.shared.utils.db_errors import describe_error
from src.app.shared.utils.logger import logger as logger_utils
from src.app.shared.utils.requests import extract_session_cookie

logger = logger_utils(__name__)


class MonitorRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # call_next returns once the response starts: a stream is not waited for
        response = await call_next(request)
        if request.url.path.startswith("/api/v1/"):
            await _register(request, response.status_code)
        return response


async def _register(request: Request, http_code: int) -> None:
    """Record the request; a failure is only logged, the response goes on."""
    session_id = extract_session_cookie(request)
    if not session_id:
        logger.warning("No X-Session-ID cookie provided for %s", request.url.path)
        return

    try:
        await run_in_threadpool(
            register_endpoint,
            endpoint=request.url.path,
            session_id=session_id,
            http_code=http_code,
            message=noted_error_code(request) if http_code >= 400 else None,
        )
    except IntegrityError:
        # the session id matches no session (deleted, or made up): nothing to
        # attach the row to, and nothing wrong with the request
        logger.debug("Request not recorded, unknown session: %s", request.url.path)
    except Exception as e:
        logger.error(
            "Failed to register endpoint %s: %s",
            request.url.path,
            describe_error(e),
        )
