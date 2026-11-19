"""
Daily quotas of guests (no Keycloak account) for the features that call an LLM.

While GUEST_RESTRICTIONS_ENABLED is on, a guest gets GUEST_CHAT_DAILY_LIMIT chat
questions, GUEST_SYLLABUS_DAILY_LIMIT syllabi, GUEST_FEEDBACK_DAILY_LIMIT changes
asked on a syllabus and GUEST_FILES_DAILY_LIMIT file summaries per sliding 24h
window. The key is the anonymous user of the session cookie, so clearing the
browser storage does not reset it. Accounts, and every caller while the flag is
off, are not limited.

Each use is a row of user_related.guest_feature_usage, so the quotas are shared
by every pod and survive restarts.

New guests are also limited per client IP (GUEST_SESSIONS_PER_IP_HOURLY), since
a new guest gets fresh quotas: only creations count, not the page loads that
renew an existing session. That limit is kept in the memory of the process
(see shared/infra/rate_limit.py).
"""

import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncIterator, Literal
from uuid import UUID

from fastapi import HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.shared.domain.responses import ErrorResponse
from src.app.shared.infra.rate_limit import SlidingWindowRateLimiter, client_ip
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.shared.utils.requests import extract_session_cookie
from src.app.user.infra.guest_usage import (
    UsageKey,
    add_guest_usage_sync,
    delete_guest_usage_sync,
    get_guest_usage_sync,
)
from src.app.user.infra.queries import get_user_for_session_renewal
from src.app.user.services.account import (
    guest_restrictions_enabled,
    verify_account_token,
)

logger = logger_utils(__name__)
settings = get_settings()

GuestFeature = Literal["chat", "syllabus", "feedback", "files"]

GUEST_QUOTA_WINDOW = timedelta(hours=24)
GUEST_QUOTA_EXCEEDED_CODE = "GUEST_QUOTA_EXCEEDED"
GUEST_REMAINING_HEADER = "X-Guest-Remaining"
GUEST_SESSION_WINDOW_SECONDS = 60 * 60

# the (feature title, action) of user_related.guest_feature_usage each quota counts
GUEST_USAGE_KEYS: dict[GuestFeature, UsageKey] = {
    "chat": ("chat", "question"),
    "syllabus": ("syllabus", "generate"),
    "feedback": ("syllabus", "feedback"),
    "files": ("syllabus", "files"),
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
            "`feature` and `retry_after` in seconds, and a Retry-After header), "
            "or the access of a time-limited account has ended (code "
            "ACCESS_EXPIRED)"
        ),
    },
}


def guest_quota_limit(feature: GuestFeature) -> int:
    return {
        "chat": settings.GUEST_CHAT_DAILY_LIMIT,
        "syllabus": settings.GUEST_SYLLABUS_DAILY_LIMIT,
        "feedback": settings.GUEST_FEEDBACK_DAILY_LIMIT,
        "files": settings.GUEST_FILES_DAILY_LIMIT,
    }[feature]


def _seconds_until_free(oldest: datetime | None, now: datetime) -> float:
    """Seconds until the oldest use of the window leaves it, 0 without any."""
    if oldest is None:
        return 0
    return max((oldest + GUEST_QUOTA_WINDOW - now).total_seconds(), 0)


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
        self._usage_id: UUID | None = None
        # known once consumed or refunded
        self._remaining: int | None = None

    @property
    def headers(self) -> dict[str, str]:
        """
        The X-Guest-Remaining header once the request was counted, for
        responses built by the route.
        """
        if self._remaining is None:
            return {}
        return {GUEST_REMAINING_HEADER: str(self._remaining)}

    async def consume(self) -> None:
        """Count one use, or raise 403 GUEST_QUOTA_EXCEEDED when none is left."""
        if self.user_id is None:
            return
        limit = guest_quota_limit(self.feature)
        now = datetime.now()
        usage_id, used, oldest = await run_in_threadpool(
            add_guest_usage_sync,
            self.user_id,
            GUEST_USAGE_KEYS[self.feature],
            limit,
            now - GUEST_QUOTA_WINDOW,
            now,
        )
        if usage_id is None:
            raise self._exceeded(self.feature, _seconds_until_free(oldest, now))
        self._usage_id = usage_id
        self._set_remaining(max(limit - used - 1, 0))

    async def require_left(self, feature: GuestFeature) -> None:
        """
        Raise 403 GUEST_QUOTA_EXCEEDED for `feature` when the guest has none of
        it left, without counting anything: for a step that is only worth
        running while a later one is still possible.
        """
        if self.user_id is None:
            return
        _, remaining, resets_in = await quota_status(feature, self.user_id)
        if not remaining:
            raise self._exceeded(feature, resets_in)

    def _exceeded(self, feature: GuestFeature, retry_after: float) -> HTTPException:
        seconds = max(math.ceil(retry_after), 1)
        logger.info(
            "Guest quota exceeded. feature=%s user_id=%s retry_after=%ss",
            feature,
            self.user_id,
            seconds,
        )
        headers = {"Retry-After": str(seconds)}
        if feature == self.feature:
            headers[GUEST_REMAINING_HEADER] = "0"
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Daily guest quota used up, create an account to continue",
                "code": GUEST_QUOTA_EXCEEDED_CODE,
                "feature": feature,
                "retry_after": seconds,
            },
            headers=headers,
        )

    async def refund(self) -> None:
        """
        Give back the use counted by `consume`, for work that failed. A refund
        that fails is only logged: the error of the work is the one to report.
        """
        if self._usage_id is None:
            return
        usage_id, self._usage_id = self._usage_id, None
        try:
            await run_in_threadpool(delete_guest_usage_sync, usage_id)
        except Exception:
            logger.exception(
                "Guest quota refund failed. feature=%s user_id=%s",
                self.feature,
                self.user_id,
            )
            return
        logger.info(
            "Guest quota refunded. feature=%s user_id=%s", self.feature, self.user_id
        )
        if self._remaining is not None:
            limit = guest_quota_limit(self.feature)
            self._set_remaining(min(self._remaining + 1, limit))

    def _set_remaining(self, remaining: int) -> None:
        self._remaining = remaining
        if self._response is not None:
            self._response.headers.update(self.headers)

    @asynccontextmanager
    async def spend(self) -> AsyncIterator[None]:
        """`consume`, then `refund` if the block raises."""
        await self.consume()
        try:
            yield
        except BaseException:
            await self.refund()
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


async def quota_status(feature: GuestFeature, user_id: UUID) -> tuple[int, int, int]:
    """(limit, remaining, seconds until a use comes back) of a guest."""
    limit = guest_quota_limit(feature)
    now = datetime.now()
    used, oldest = await run_in_threadpool(
        get_guest_usage_sync,
        user_id,
        GUEST_USAGE_KEYS[feature],
        now - GUEST_QUOTA_WINDOW,
    )
    return limit, max(limit - used, 0), math.ceil(_seconds_until_free(oldest, now))


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
