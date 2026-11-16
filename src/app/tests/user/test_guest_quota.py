import unittest
import uuid
from unittest import mock

from fastapi import HTTPException, Response
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.app.core.config import settings
from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.shared.infra.rate_limit import SlidingWindowRateLimiter
from src.app.user.infra.keycloak import KeycloakUser
from src.app.user.services import account, guest_quota
from src.app.user.services.guest_quota import GuestQuota, resolve_quota_user
from src.main import app

BASE = f"{settings.API_V1_STR}/user"
GUEST, SESSION = uuid.uuid4(), uuid.uuid4()

client = TestClient(app)


def make_request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


def restrictions(enabled: bool):
    return mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", enabled)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class LimiterRefundPeekTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.limiter = SlidingWindowRateLimiter(2, 100, clock=self.clock)

    def test_peek_does_not_count(self):
        self.assertEqual(self.limiter.peek("k"), (2, 0.0))
        self.limiter.hit("k")
        self.clock.now += 10
        self.assertEqual(self.limiter.peek("k"), (1, 90.0))
        self.assertEqual(self.limiter.peek("k"), (1, 90.0))

    def test_refund_gives_back_the_latest_hit(self):
        self.limiter.hit("k")
        self.limiter.hit("k")
        self.assertGreater(self.limiter.hit("k"), 0)

        self.limiter.refund("k")

        self.assertEqual(self.limiter.peek("k")[0], 1)
        self.assertEqual(self.limiter.hit("k"), 0.0)

    def test_refund_of_unknown_key_is_a_no_op(self):
        self.limiter.refund("nobody")
        self.assertEqual(self.limiter.peek("nobody"), (2, 0.0))

    def test_hits_leave_the_window(self):
        self.limiter.hit("k")
        self.clock.now += 100
        self.assertEqual(self.limiter.peek("k"), (2, 0.0))


class ResolveQuotaUserTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.object(
            guest_quota, "get_user_for_session_renewal", return_value=GUEST
        )
        self.lookup = patcher.start()
        self.addCleanup(patcher.stop)

    async def test_restrictions_off_limits_nobody(self):
        with restrictions(False):
            request = make_request({"X-Session-Id": str(SESSION)})
            self.assertIsNone(await resolve_quota_user(request))
        self.lookup.assert_not_called()

    async def test_guest_is_resolved_from_the_session(self):
        with restrictions(True):
            request = make_request({"X-Session-Id": str(SESSION)})
            self.assertEqual(await resolve_quota_user(request), GUEST)
        self.lookup.assert_called_once_with(SESSION)

    @mock.patch.object(account, "decode_keycloak_token")
    async def test_account_is_not_limited(self, decode):
        decode.return_value = KeycloakUser(keycloak_id=uuid.uuid4())
        with restrictions(True):
            request = make_request({"Authorization": "Bearer a.b.c"})
            self.assertIsNone(await resolve_quota_user(request))
        self.lookup.assert_not_called()

    async def test_guest_without_session_is_refused(self):
        with restrictions(True), self.assertRaises(SessionNotFoundError):
            await resolve_quota_user(make_request({}))

    async def test_unknown_session_is_refused(self):
        self.lookup.side_effect = SessionNotFoundError()
        with restrictions(True), self.assertRaises(SessionNotFoundError):
            await resolve_quota_user(make_request({"X-Session-Id": str(SESSION)}))


class GuestQuotaTests(unittest.TestCase):
    def setUp(self):
        self.response = Response()
        self.quota = GuestQuota("syllabus", GUEST, self.response)
        self.limit = settings.GUEST_SYLLABUS_DAILY_LIMIT

    def test_consume_until_exceeded(self):
        for _ in range(self.limit):
            self.quota.consume()
        self.assertEqual(self.response.headers["X-Guest-Remaining"], "0")

        with self.assertRaises(HTTPException) as ctx:
            self.quota.consume()

        error = ctx.exception
        self.assertEqual(error.status_code, 403)
        self.assertEqual(error.detail["code"], "GUEST_QUOTA_EXCEEDED")
        self.assertEqual(error.detail["feature"], "syllabus")
        self.assertGreater(error.detail["retry_after"], 0)
        self.assertEqual(error.headers["Retry-After"], str(error.detail["retry_after"]))

    def test_refund_after_consume(self):
        self.quota.consume()
        self.quota.refund()
        self.assertEqual(self.quota.remaining, self.limit)
        self.assertEqual(self.response.headers["X-Guest-Remaining"], str(self.limit))

    def test_refund_without_consume_gives_nothing_back(self):
        GuestQuota("syllabus", GUEST).consume()
        self.quota.refund()
        self.assertEqual(self.quota.remaining, self.limit - 1)

    def test_spend_refunds_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with self.quota.spend():
                raise RuntimeError("llm down")
        self.assertEqual(self.quota.remaining, self.limit)

    def test_spend_keeps_the_use_on_success(self):
        with self.quota.spend():
            pass
        self.assertEqual(self.quota.remaining, self.limit - 1)

    def test_unlimited_caller_is_never_counted(self):
        quota = GuestQuota("syllabus", None, self.response)
        for _ in range(self.limit + 3):
            quota.consume()
        quota.refund()
        self.assertIsNone(quota.remaining)
        self.assertEqual(quota.headers, {})
        self.assertNotIn("X-Guest-Remaining", self.response.headers)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class GuestQuotaRouteTests(unittest.TestCase):
    headers = {"X-API-Key": "test", "X-Session-Id": str(SESSION)}

    def setUp(self):
        patcher = mock.patch.object(
            guest_quota, "get_user_for_session_renewal", return_value=GUEST
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_guest_sees_what_is_left(self):
        GuestQuota("chat", GUEST).consume()

        with restrictions(True):
            response = client.get(f"{BASE}/guest_quota", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["limited"])
        chat = body["chat"]
        self.assertEqual(chat["limit"], settings.GUEST_CHAT_DAILY_LIMIT)
        self.assertEqual(chat["remaining"], settings.GUEST_CHAT_DAILY_LIMIT - 1)
        self.assertGreater(chat["resets_in"], 0)
        self.assertEqual(
            body["syllabus"],
            {
                "limit": settings.GUEST_SYLLABUS_DAILY_LIMIT,
                "remaining": settings.GUEST_SYLLABUS_DAILY_LIMIT,
                "resets_in": 0,
            },
        )

    def test_reading_the_quota_uses_nothing(self):
        with restrictions(True):
            for _ in range(3):
                client.get(f"{BASE}/guest_quota", headers=self.headers)
        self.assertEqual(
            GuestQuota("chat", GUEST).remaining, settings.GUEST_CHAT_DAILY_LIMIT
        )

    def test_restrictions_off_is_not_limited(self):
        with restrictions(False):
            response = client.get(f"{BASE}/guest_quota", headers=self.headers)

        self.assertEqual(
            response.json(), {"limited": False, "chat": None, "syllabus": None}
        )

    def test_guest_without_session_gets_401(self):
        with restrictions(True):
            response = client.get(f"{BASE}/guest_quota", headers={"X-API-Key": "test"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"]["code"], "SESSION_NOT_FOUND")
