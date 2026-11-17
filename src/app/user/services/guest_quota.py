"""
Daily quotas of guests (no Keycloak account) for the features that call an LLM.

While GUEST_RESTRICTIONS_ENABLED is on, a guest gets GUEST_CHAT_DAILY_LIMIT chat
questions and GUEST_SYLLABUS_DAILY_LIMIT syllabi per sliding 24h window. The key
is the anonymous user of the session cookie, so clearing the browser storage
does not reset it. Accounts, and every caller while the flag is off, are not
limited.

New guests are also limited per client IP (GUEST_SESSIONS_PER_IP_HOURLY), since
a new guest gets fresh quotas: only creations count, not the page loads that
renew an existing session.

Known limits (v0): the counters live in the memory of the process
(see shared/infra/rate_limit.py). They apply per pod, are lost on restart and
multiply by the number of replicas. A persistent table can replace
the in-memory counters if this proves too lenient.
"""

import math
from contextlib import contextmanager
from typing import Iterator, Literal
from uuid import UUID

from fastapi import HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.shared.domain.responses import ErrorResponse
from src.app.shared.infra.rate_limit import SlidingWindowRateLimiter, client_ip
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.shared.utils.requests import extract_session_cookie
from src.app.user.infra.queries import get_user_for_session_renewal
from src.app.user.services.account import (
    guest_restrictions_enabled,
    verify_account_token,
)

logger = logger_utils(__name__)
settings = get_settings()

GuestFeature = Literal["chat", "syllabus"]

GUEST_QUOTA_WINDOW_SECONDS = 24 * 60 * 60
GUEST_QUOTA_EXCEEDED_CODE = "GUEST_QUOTA_EXCEEDED"
GUEST_REMAINING_HEADER = "X-Guest-Remaining"
GUEST_SESSION_WINDOW_SECONDS = 60 * 60

# past max_keys the least recently seen guests are forgotten (their count
# restarts): sized well above the daily number of guests of one pod
_limiters: dict[GuestFeature, SlidingWindowRateLimiter] = {
    "chat": SlidingWindowRateLimiter(
        settings.GUEST_CHAT_DAILY_LIMIT, GUEST_QUOTA_WINDOW_SECONDS, max_keys=100_000
    ),
    "syllabus": SlidingWindowRateLimiter(
        settings.GUEST_SYLLABUS_DAILY_LIMIT,
        GUEST_QUOTA_WINDOW_SECONDS,
        max_keys=100_000,
    ),
}

# new guests per client IP; only used when GUEST_SESSIONS_PER_IP_HOURLY > 0
_new_guest_ip_limiter = SlidingWindowRateLimiter(
    max(settings.GUEST_SESSIONS_PER_IP_HOURLY, 1),
    GUEST_SESSION_WINDOW_SECONDS,
    max_keys=100_000,
)

NEW_GUEST_RATE_LIMITED_RESPONSES: dict[int | str, dict] = {
    429: {
        "model": ErrorResponse,
        "description": (
            "Too many new guests from this IP (code RATE_LIMITED, with a "
            "Retry-After header); a request with a known session is not counted"
        ),
    },
}

# routes counted against a guest quota
GUEST_QUOTA_RESPONSES: dict[int | str, dict] = {
    401: {
        "model": ErrorResponse,
        "description": (
            "Invalid or expired token, or, with guest restrictions on, a guest "
            "without a known session (code SESSION_NOT_FOUND)"
        ),
    },
    403: {
        "model": ErrorResponse,
        "description": (
            f"Guest daily quota used up (code {GUEST_QUOTA_EXCEEDED_CODE}, with "
            "`feature` and `retry_after` in seconds, and a Retry-After header)"
        ),
    },
}


def guest_limiter(feature: GuestFeature) -> SlidingWindowRateLimiter:
    return _limiters[feature]


async def resolve_quota_user(request: Request) -> UUID | None:
    """
    The anonymous user whose quota a request counts against, or None when it
    is not limited: guest restrictions off, or a valid account token. A guest
    without a session (or with an unknown one) is refused: it could not be
    counted. An expired guest session still resolves to its user, as the
    cookie outlives the 24h session in the database.
    """
    if not guest_restrictions_enabled():
        return None
    if await verify_account_token(request):
        return None

    session_id = extract_session_cookie(request)
    if not session_id:
        raise SessionNotFoundError("A session is required")
    return await run_in_threadpool(get_user_for_session_renewal, session_id)


class GuestQuota:
    """
    The quota of one request for one feature. `consume` once the request is
    accepted (after its cheap checks), `refund` if the work then fails, so a
    guest only pays for what they got. Both do nothing for an unlimited caller.
    """

    def __init__(
        self,
        feature: GuestFeature,
        user_id: UUID | None,
        response: Response | None = None,
    ):
        self.feature = feature
        self.user_id = user_id
        self._response = response
        self._consumed = False

    @property
    def remaining(self) -> int | None:
        if self.user_id is None:
            return None
        return guest_limiter(self.feature).peek(str(self.user_id))[0]

    @property
    def headers(self) -> dict[str, str]:
        """The X-Guest-Remaining header, for responses built by the route."""
        remaining = self.remaining
        return {} if remaining is None else {GUEST_REMAINING_HEADER: str(remaining)}

    def consume(self) -> None:
        """Count one use, or raise 403 GUEST_QUOTA_EXCEEDED when none is left."""
        if self.user_id is None:
            return
        retry_after = guest_limiter(self.feature).hit(str(self.user_id))
        if retry_after:
            seconds = math.ceil(retry_after)
            logger.info(
                "Guest quota exceeded. feature=%s user_id=%s retry_after=%ss",
                self.feature,
                self.user_id,
                seconds,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "message": "Daily guest quota used up, create an account to continue",
                    "code": GUEST_QUOTA_EXCEEDED_CODE,
                    "feature": self.feature,
                    "retry_after": seconds,
                },
                headers={"Retry-After": str(seconds), GUEST_REMAINING_HEADER: "0"},
            )
        self._consumed = True
        if self._response is not None:
            self._response.headers.update(self.headers)

    def refund(self) -> None:
        """Give back the use counted by `consume`, for work that failed."""
        if self.user_id is None or not self._consumed:
            return
        guest_limiter(self.feature).refund(str(self.user_id))
        self._consumed = False
        logger.info(
            "Guest quota refunded. feature=%s user_id=%s", self.feature, self.user_id
        )
        if self._response is not None:
            self._response.headers.update(self.headers)

    @contextmanager
    def spend(self) -> Iterator[None]:
        """`consume`, then `refund` if the block raises."""
        self.consume()
        try:
            yield
        except BaseException:
            self.refund()
            raise


def guest_quota(feature: GuestFeature):
    """
    Dependency giving the route the caller's GuestQuota for `feature`. It only
    resolves who is counted: the route calls `consume` itself, once the request
    passed its own checks, so a rejected request costs nothing.
    """

    async def dependency(request: Request, response: Response) -> GuestQuota:
        user_id = await resolve_quota_user(request)
        return GuestQuota(feature, user_id, response)

    return dependency


def quota_status(feature: GuestFeature, user_id: UUID) -> tuple[int, int, int]:
    """(limit, remaining, seconds until a use comes back) of a guest."""
    limiter = guest_limiter(feature)
    remaining, resets_in = limiter.peek(str(user_id))
    return limiter.limit, remaining, math.ceil(resets_in)


def enforce_new_guest_ip_limit(request: Request) -> None:
    """
    Count the creation of a guest against the caller's IP, or raise 429
    RATE_LIMITED past GUEST_SESSIONS_PER_IP_HOURLY. Does nothing while guest
    restrictions or the limit are off.
    """
    if not guest_restrictions_enabled() or settings.GUEST_SESSIONS_PER_IP_HOURLY < 1:
        return
    retry_after = _new_guest_ip_limiter.hit(client_ip(request))
    if retry_after:
        seconds = math.ceil(retry_after)
        logger.warning("New guests rate limited. retry_after=%ss", seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Too many new sessions, try again later",
                "code": "RATE_LIMITED",
            },
            headers={"Retry-After": str(seconds)},
        )
