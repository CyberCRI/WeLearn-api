from typing import Optional

from fastapi import HTTPException, Response, status

from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


def bad_request(message: str, msg_code: str):
    """400 Bad Request"""
    logger.error("400 Bad Request: %s, Code: %s", message, msg_code)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"message": message, "code": msg_code},
    )


def no_content(message: str, msg_code: str):
    """204 No Content"""
    logger.info("204 No Content: %s, Code: %s", message, msg_code)
    raise HTTPException(
        status_code=status.HTTP_204_NO_CONTENT,
        detail={"message": message, "code": msg_code},
    )


def not_found(message: str, msg_code: str):
    """404 Not Found"""
    logger.error("404 Not Found: %s, Code: %s", message, msg_code)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": message, "code": msg_code},
    )


class EmptyQueryError(BaseException):
    """Raised when an invalid language code is used"""

    def __init__(
        self,
        message="Empty query",
        msg_code="EMPTY_QUERY",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.error("EmptyQueryError: %s, Code: %s", self.message, self.msg_code)
        super().__init__(self.message, self.msg_code)


class LanguageNotSupportedError(BaseException):
    """Raised when an invalid language code is used"""

    def __init__(
        self,
        message="Language not supported",
        msg_code="LANG_NOT_SUPPORTED",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.error(
            "LanguageNotSupportedError: %s, Code: %s", self.message, self.msg_code
        )
        super().__init__(self.message, self.msg_code)


class InvalidQuestionError(BaseException):
    """Raised when an invalid question is used"""

    def __init__(
        self,
        message="Please provide a valid question",
        msg_code="INVALID_QUESTION",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.error("InvalidQuestionError: %s, Code: %s", self.message, self.msg_code)
        super().__init__(self.message, self.msg_code)


class NoResultsError(BaseException):
    """Raised when no results are found"""

    def __init__(
        self,
        message="No results found",
        msg_code="NO_RESULTS",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.info("NoResultsError: %s, Code: %s", self.message, self.msg_code)
        super().__init__(self.message, self.msg_code)


class CollectionNotFoundError(BaseException):
    """Raised when no results are found"""

    def __init__(
        self,
        message="Collection not found",
        msg_code="COLL_NOT_FOUND",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.error(
            "CollectionNotFoundError: %s, Code: %s", self.message, self.msg_code
        )
        super().__init__(self.message, self.msg_code)


class ModelNotFoundError(BaseException):
    """Raised when no results are found"""

    def __init__(
        self,
        message="Model not found",
        msg_code="MODEL_NOT_FOUND",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.error("ModelNotFoundError: %s, Code: %s", self.message, self.msg_code)
        super().__init__(self.message, self.msg_code)


class SubjectNotFoundError(BaseException):
    """Raised when no results are found"""

    def __init__(
        self,
        message="Subject not found",
        msg_code="SUBJECT_NOT_FOUND",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.error("SubjectNotFoundError: %s, Code: %s", self.message, self.msg_code)
        super().__init__(self.message, self.msg_code)


class PartialResponseResultError(BaseException):
    """Raised when no results are found"""

    def __init__(
        self,
        message="Partial response result",
        msg_code="PARTIAL_RESULT",
    ):
        self.message = message
        self.msg_code = msg_code
        logger.warning(
            "PartialResponseResultError: %s, Code: %s", self.message, self.msg_code
        )
        super().__init__(self.message, self.msg_code)


def handle_error(response: Optional[Response], exc: Exception) -> None:
    if isinstance(exc, PartialResponseResultError):
        if response:
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
        logger.warning(
            "Partial response result: %s, Code: %s", exc.message, exc.msg_code
        )
    elif isinstance(exc, NoResultsError):
        no_content(message=exc.message, msg_code=exc.msg_code)
    elif isinstance(exc, LanguageNotSupportedError):
        bad_request(message=exc.message, msg_code=exc.msg_code)
    elif isinstance(exc, (CollectionNotFoundError, ModelNotFoundError)):
        not_found(message=exc.message, msg_code=exc.msg_code)
    elif isinstance(exc, EmptyQueryError):
        bad_request(message=exc.message, msg_code=exc.msg_code)
    elif isinstance(exc, SubjectNotFoundError):
        not_found(message=exc.message, msg_code=exc.msg_code)
    else:
        logger.error("Unhandled exception: %s", exc)
