from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from src.app.shared.domain.exceptions import WeLearnError
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


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(WeLearnError, welearn_error_handler)
