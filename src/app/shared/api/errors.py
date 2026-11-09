from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from src.app.shared.domain.exceptions import WeLearnError
from src.app.shared.utils.db_errors import (
    DB_ERRORS,
    describe_error,
    describe_traceback,
    is_db_connection_error,
)
from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


async def welearn_error_handler(request: Request, exc: Exception) -> Response:
    """Map a domain error to its status code and `{"detail": {"message", "code"}}`."""
    assert isinstance(exc, WeLearnError)

    log = (
        logger.error
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
        else logger.warning
    )
    log(
        "%s endpoint=%s status=%s code=%s message=%s",
        type(exc).__name__,
        request.url.path,
        exc.status_code,
        exc.msg_code,
        exc.message,
    )

    if exc.status_code == status.HTTP_204_NO_CONTENT:
        return Response(status_code=exc.status_code)

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


DATABASE_UNAVAILABLE_DETAIL = {
    "message": "Database unavailable",
    "code": "DATABASE_UNAVAILABLE",
}
DATABASE_ERROR_DETAIL = {"message": "Internal error", "code": "DATABASE_ERROR"}


async def database_error_handler(request: Request, exc: Exception) -> Response:
    """
    Uncaught psycopg or SQLAlchemy error: 503 when the database cannot be
    reached, 500 otherwise. Their message can quote the connection string,
    password included, so neither the log nor the body contains it (C-9); the
    log keeps the error type and the stack frames.
    """
    unavailable = is_db_connection_error(exc)
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if unavailable
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    logger.error(
        "Database error endpoint=%s status=%s error=%s\n%s",
        request.url.path,
        status_code,
        describe_error(exc),
        describe_traceback(exc),
    )
    detail = DATABASE_UNAVAILABLE_DETAIL if unavailable else DATABASE_ERROR_DETAIL
    return JSONResponse(status_code=status_code, content={"detail": detail})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(WeLearnError, welearn_error_handler)
    # handled here rather than by Starlette's last-resort handler, which
    # re-raises them for uvicorn to log with their full message
    for db_error in DB_ERRORS:
        app.add_exception_handler(db_error, database_error_handler)
