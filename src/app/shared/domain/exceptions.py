from typing import NoReturn

from fastapi import HTTPException, Response, status

from src.app.shared.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


def bad_request(message: str, msg_code: str) -> NoReturn:
    """400 Bad Request"""
    logger.error("400 Bad Request: %s, Code: %s", message, msg_code)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"message": message, "code": msg_code},
    )


def no_content(message: str, msg_code: str) -> NoReturn:
    """204 No Content"""
    logger.info("204 No Content: %s, Code: %s", message, msg_code)
    raise HTTPException(
        status_code=status.HTTP_204_NO_CONTENT,
        detail={"message": message, "code": msg_code},
    )


def not_found(message: str, msg_code: str) -> NoReturn:
    """404 Not Found"""
    logger.error("404 Not Found: %s, Code: %s", message, msg_code)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": message, "code": msg_code},
    )


class WeLearnError(Exception):
    """
    Base class of every domain error. Subclasses only set the class attributes;
    the handler registered by `shared/api/errors.py` maps them to an HTTP
    response with a `{"detail": {"message", "code"}}` body.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_message: str = "Internal error"
    default_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str | None = None, msg_code: str | None = None):
        self.message = message if message is not None else self.default_message
        self.msg_code = msg_code if msg_code is not None else self.default_code
        super().__init__(self.message, self.msg_code)

    @property
    def detail(self) -> dict[str, str]:
        return {"message": self.message, "code": self.msg_code}


class SessionNotFoundError(WeLearnError):
    """Raised when a session id does not match any session"""

    status_code = status.HTTP_401_UNAUTHORIZED
    default_message = "Session not found"
    default_code = "SESSION_NOT_FOUND"


class UserNotFoundError(WeLearnError):
    """Raised when a user is not found"""

    status_code = status.HTTP_404_NOT_FOUND
    default_message = "User not found"
    default_code = "USER_NOT_FOUND"


class ThreadNotFoundError(WeLearnError):
    """
    Raised when a chat thread is not accessible to the caller. A thread owned
    by another user gets the same answer as a missing one, so its existence
    does not leak.
    """

    status_code = status.HTTP_404_NOT_FOUND
    default_message = "Thread not found"
    default_code = "THREAD_NOT_FOUND"


class EmptyQueryError(WeLearnError):
    """Raised when the query is empty"""

    status_code = status.HTTP_400_BAD_REQUEST
    default_message = "Empty query"
    default_code = "EMPTY_QUERY"


class LanguageNotSupportedError(WeLearnError):
    """Raised when an invalid language code is used"""

    status_code = status.HTTP_400_BAD_REQUEST
    default_message = "Language not supported"
    default_code = "LANG_NOT_SUPPORTED"


class NoResultsError(WeLearnError):
    """Raised when no results are found"""

    status_code = status.HTTP_204_NO_CONTENT
    default_message = "No results found"
    default_code = "NO_RESULTS"


class BookmarkNotFoundError(WeLearnError):
    """Raised when a user has no bookmark for a document"""

    status_code = status.HTTP_404_NOT_FOUND
    default_message = "Bookmark not found"
    default_code = "BOOKMARK_NOT_FOUND"


class BookmarkAlreadyExistsError(WeLearnError):
    """Raised when a user bookmarks a document twice"""

    status_code = status.HTTP_400_BAD_REQUEST
    default_message = "Bookmark already exists"
    default_code = "BOOKMARK_ALREADY_EXISTS"


class CollectionNotFoundError(WeLearnError):
    """Raised when no Qdrant collection matches the request"""

    status_code = status.HTTP_404_NOT_FOUND
    default_message = "Collection not found"
    default_code = "COLL_NOT_FOUND"


class ModelNotFoundError(WeLearnError):
    """Raised when an embedding model cannot be loaded"""

    status_code = status.HTTP_404_NOT_FOUND
    default_message = "Model not found"
    default_code = "MODEL_NOT_FOUND"


class SubjectNotFoundError(WeLearnError):
    """Raised when a subject is not found"""

    status_code = status.HTTP_404_NOT_FOUND
    default_message = "Subject not found"
    default_code = "SUBJECT_NOT_FOUND"


class MetricsNotReadyError(WeLearnError):
    """
    Raised when a metric reads a materialized view that has never been
    refreshed (created `WITH NO DATA`, or after a migration or restore).
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_message = "Metrics are not available yet"
    default_code = "METRICS_NOT_READY"


class PartialResponseResultError(WeLearnError):
    """Raised when only part of the results could be computed"""

    status_code = status.HTTP_206_PARTIAL_CONTENT
    default_message = "Partial response result"
    default_code = "PARTIAL_RESULT"


def handle_error(exc: Exception, response: Response | None = None) -> None:
    """
    Turn a domain error into its HTTPException. A partial result only sets 206
    on `response`. Any other exception is re-raised unchanged.
    """
    if isinstance(exc, PartialResponseResultError):
        if response:
            response.status_code = exc.status_code
        logger.warning(
            "Partial response result: %s, Code: %s", exc.message, exc.msg_code
        )
        return
    if isinstance(exc, WeLearnError):
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    raise exc
