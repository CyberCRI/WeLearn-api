"""
Syllabus v2: brakes on the LLM routes of /tutor/course/*, per session (or per client IP
for a request without one), whoever the caller is:

- at most TUTOR_COURSE_IN_FLIGHT_PER_SESSION requests running at once (429
  TOO_MANY_REQUESTS_IN_FLIGHT). Each one sends the course and up to a dozen resources to
  the model, on a per-minute quota of the provider that every user shares;
- at most TUTOR_COURSE_REQUESTS_PER_WINDOW requests per TUTOR_COURSE_WINDOW_SECONDS
  (429 RATE_LIMITED, with Retry-After).

Both are kept in the memory of the process (shared/infra/rate_limit.py): they apply per
pod, which is enough with the one replica of the API. Guests' daily quotas, kept in the
database, come on top (user/services/guest_quota.py).
"""

import math
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request, status

from src.app.shared.domain.exceptions import CourseTooLargeError
from src.app.shared.domain.responses import ErrorResponse
from src.app.shared.infra.rate_limit import (
    InFlightLimiter,
    SlidingWindowRateLimiter,
    client_ip,
)
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as utils_logger
from src.app.shared.utils.requests import extract_session_cookie
from src.app.tutor.domain.connections import ConfirmedCourse
from src.app.tutor.services.connections import course_block
from src.app.tutor.services.llm import MAX_DATA_CHARS

logger = utils_logger(__name__)
settings = get_settings()

IN_FLIGHT_CODE = "TOO_MANY_REQUESTS_IN_FLIGHT"
# a running request ends within a few LLM calls
IN_FLIGHT_RETRY_AFTER_SECONDS = 5

_in_flight = InFlightLimiter(settings.TUTOR_COURSE_IN_FLIGHT_PER_SESSION)
_requests = SlidingWindowRateLimiter(
    settings.TUTOR_COURSE_REQUESTS_PER_WINDOW,
    settings.TUTOR_COURSE_WINDOW_SECONDS,
    max_keys=100_000,
)

COURSE_LIMIT_RESPONSES: dict[int | str, dict] = {
    429: {
        "model": ErrorResponse,
        "description": (
            "The session already has its LLM requests running (code "
            "TOO_MANY_REQUESTS_IN_FLIGHT) or sent too many requests (code "
            "RATE_LIMITED); both come with a Retry-After header"
        ),
    },
    503: {
        "model": ErrorResponse,
        "description": "The model failed or did not answer in time (code LLM_UNAVAILABLE)",
    },
}
COURSE_TOO_LARGE_RESPONSE: dict[int | str, dict] = {
    413: {
        "model": ErrorResponse,
        "description": "The course does not fit a prompt (code COURSE_TOO_LARGE)",
    },
}


def limit_key(request: Request) -> str:
    session_id = extract_session_cookie(request)
    return f"session:{session_id}" if session_id else f"ip:{client_ip(request)}"


def _too_many(message: str, code: str, retry_after: float) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={"message": message, "code": code},
        headers={"Retry-After": str(max(math.ceil(retry_after), 1))},
    )


async def course_llm_request(request: Request) -> AsyncIterator[None]:
    """
    Dependency of the /tutor/course/* routes: counts the request against the session's
    window, holds one of its in-flight places while the route runs, and frees it after.
    """
    key = limit_key(request)
    if not _in_flight.acquire(key):
        logger.warning("tutor_course_in_flight_limited")
        raise _too_many(
            "Wait for the requests already running",
            IN_FLIGHT_CODE,
            IN_FLIGHT_RETRY_AFTER_SECONDS,
        )
    try:
        retry_after = _requests.hit(key)
        if retry_after:
            logger.warning(
                "tutor_course_rate_limited retry_after=%ss", math.ceil(retry_after)
            )
            raise _too_many(
                "Too many requests, try again later", "RATE_LIMITED", retry_after
            )
        yield
    finally:
        _in_flight.release(key)


def ensure_course_fits(course: ConfirmedCourse) -> None:
    """A course longer than one prompt's data block would be cut: refused instead."""
    if len(course_block(course)) > MAX_DATA_CHARS:
        raise CourseTooLargeError()
