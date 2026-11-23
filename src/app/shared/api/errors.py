import math

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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


def retry_later(
    status_code: int,
    detail: dict,
    retry_after: float,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """
    An error the caller may retry after `retry_after` seconds (at least one): said in the
    Retry-After header and in the body's `retry_after`, since a browser on another origin
    cannot read that header.
    """
    seconds = max(math.ceil(retry_after), 1)
    return HTTPException(
        status_code=status_code,
        detail={**detail, "retry_after": seconds},
        headers={"Retry-After": str(seconds), **(headers or {})},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(WeLearnError, welearn_error_handler)
    # handled here rather than by Starlette's last-resort handler, which
    # re-raises them for uvicorn to log with their full message
    for db_error in DB_ERRORS:
        app.add_exception_handler(db_error, database_error_handler)


INTERNAL_ERROR_DETAIL = {"message": "Internal server error", "code": "INTERNAL_ERROR"}


def describe_chain(exc: BaseException) -> str:
    """
    Each error of the chain (`raise … from …`), outermost first, with its frames.
    Database errors are described without their message, as everywhere else.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(
            f"{type(current).__name__}: {describe_error(current)}\n"
            f"{describe_traceback(current)}"
        )
        current = current.__cause__ or current.__context__
    return "caused by ".join(parts)


class UnhandledErrorMiddleware:
    """
    Turns an exception no handler caught into a JSON 500 `{"detail": {"message",
    "code": "INTERNAL_ERROR"}}`.

    Starlette builds the response of such an exception in its outermost
    middleware, outside CORSMiddleware: it has no CORS headers, and the browser
    reports a CORS error ("Network Error") instead of the 500. Added just inside
    CORSMiddleware, this middleware gives the client a 500 it can read.

    A response that has already started (a stream) can't be replaced: the
    exception goes on as before.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def send_and_note(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_and_note)
        except Exception as exc:
            if started:
                raise
            logger.error(
                "unhandled_error method=%s endpoint=%s error=%s",
                scope.get("method"),
                scope.get("path"),
                describe_chain(exc),
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": INTERNAL_ERROR_DETAIL},
            )
            await response(scope, receive, send)
