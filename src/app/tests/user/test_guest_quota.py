import unittest
import uuid
from datetime import datetime, timedelta
from unittest import mock

from fastapi import HTTPException, Response
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.requests import Request
from welearn_database.data.models import Feature, GuestFeatureUsage

from src.app.core.config import settings
from src.app.shared.domain.exceptions import SessionNotFoundError, UserNotFoundError
from src.app.tests.user.guest_usage_db import guest_left, guest_use, use_guest_usage_db
from src.app.user.infra.guest_usage import (
    add_guest_usage_sync,
    delete_guest_usage_sync,
    get_guest_usage_sync,
)
from src.app.user.infra.keycloak import KeycloakUser
from src.app.user.services import account, guest_quota
from src.app.user.services.guest_quota import (
    GUEST_QUOTA_WINDOW,
    GUEST_USAGE_KEYS,
    GuestQuota,
    resolve_quota_user,
)
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


class GuestUsageQueriesTests(unittest.TestCase):
    def setUp(self):
        self.s_maker = use_guest_usage_db(self, GUEST)
        self.key = GUEST_USAGE_KEYS["chat"]
        self.now = datetime.now()
        self.since = self.now - GUEST_QUOTA_WINDOW

    def add(self, limit=2, user_id=GUEST, key=None):
        return add_guest_usage_sync(
            user_id, key or self.key, limit, self.since, self.now
        )

    def test_add_until_the_limit(self):
        first_id, used, oldest = self.add()
        self.assertIsNotNone(first_id)
        self.assertEqual((used, oldest), (0, None))

        second_id, used, oldest = self.add()
        self.assertIsNotNone(second_id)
        self.assertEqual((used, oldest), (1, self.now))

        refused_id, used, _ = self.add()
        self.assertIsNone(refused_id)
        self.assertEqual(used, 2)
        self.assertEqual(get_guest_usage_sync(GUEST, self.key, self.since)[0], 2)

    def test_uses_before_the_window_are_not_counted(self):
        with self.s_maker() as s:
            feature_id = s.execute(
                select(Feature.id).where(Feature.title == "chat")
            ).scalar_one()
            s.add(
                GuestFeatureUsage(
                    inferred_user_id=GUEST,
                    feature_id=feature_id,
                    action="question",
                    created_at=self.since - timedelta(seconds=1),
                )
            )
            s.commit()

        self.assertEqual(get_guest_usage_sync(GUEST, self.key, self.since), (0, None))
        self.assertIsNotNone(self.add(limit=1)[0])

    def test_each_feature_and_action_is_counted_apart(self):
        self.add(limit=1)

        for feature in ("syllabus", "feedback", "files"):
            key = GUEST_USAGE_KEYS[feature]
            self.assertIsNotNone(self.add(limit=1, key=key)[0])
        self.assertIsNone(self.add(limit=1)[0])

    def test_delete_removes_only_that_use(self):
        first_id, _, _ = self.add()
        self.add()

        delete_guest_usage_sync(first_id)

        self.assertEqual(get_guest_usage_sync(GUEST, self.key, self.since)[0], 1)

    def test_unknown_user_is_refused(self):
        with self.assertRaises(UserNotFoundError):
            self.add(user_id=uuid.uuid4())

    def test_missing_feature_is_an_error(self):
        with self.assertRaises(LookupError):
            self.add(key=("unknown", "question"))


class RequireLeftTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        use_guest_usage_db(self, GUEST)

    async def test_raises_for_the_other_feature_without_counting(self):
        guest_use("syllabus", GUEST, settings.GUEST_SYLLABUS_DAILY_LIMIT)
        files = GuestQuota("files", GUEST)

        with self.assertRaises(HTTPException) as raised:
            await files.require_left("syllabus")

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail["feature"], "syllabus")
        self.assertGreater(raised.exception.detail["retry_after"], 0)
        self.assertNotIn("X-Guest-Remaining", raised.exception.headers)
        self.assertEqual(guest_left("files", GUEST), settings.GUEST_FILES_DAILY_LIMIT)

    async def test_passes_while_some_is_left(self):
        await GuestQuota("files", GUEST).require_left("syllabus")

    async def test_unlimited_caller_always_passes(self):
        guest_use("syllabus", GUEST, settings.GUEST_SYLLABUS_DAILY_LIMIT)
        await GuestQuota("files", None).require_left("syllabus")


class GuestQuotaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        use_guest_usage_db(self, GUEST)
        self.response = Response()
        self.quota = GuestQuota("syllabus", GUEST, self.response)
        self.limit = settings.GUEST_SYLLABUS_DAILY_LIMIT

    async def test_consume_until_exceeded(self):
        for _ in range(self.limit):
            await GuestQuota("syllabus", GUEST, self.response).consume()
        self.assertEqual(self.response.headers["X-Guest-Remaining"], "0")

        with self.assertRaises(HTTPException) as ctx:
            await self.quota.consume()

        error = ctx.exception
        self.assertEqual(error.status_code, 403)
        self.assertEqual(error.detail["code"], "GUEST_QUOTA_EXCEEDED")
        self.assertEqual(error.detail["feature"], "syllabus")
        self.assertGreater(error.detail["retry_after"], 0)
        self.assertEqual(error.headers["Retry-After"], str(error.detail["retry_after"]))
        self.assertEqual(error.headers["X-Guest-Remaining"], "0")

    async def test_consume_sets_what_is_left(self):
        await self.quota.consume()
        self.assertEqual(self.quota.headers, {"X-Guest-Remaining": str(self.limit - 1)})
        self.assertEqual(
            self.response.headers["X-Guest-Remaining"], str(self.limit - 1)
        )

    async def test_refund_after_consume(self):
        await self.quota.consume()
        await self.quota.refund()
        self.assertEqual(guest_left("syllabus", GUEST), self.limit)
        self.assertEqual(self.response.headers["X-Guest-Remaining"], str(self.limit))

    async def test_refund_gives_back_once(self):
        await self.quota.consume()
        await self.quota.refund()
        guest_use("syllabus", GUEST)
        await self.quota.refund()
        self.assertEqual(guest_left("syllabus", GUEST), self.limit - 1)

    async def test_refund_without_consume_gives_nothing_back(self):
        guest_use("syllabus", GUEST)
        await self.quota.refund()
        self.assertEqual(guest_left("syllabus", GUEST), self.limit - 1)
        self.assertEqual(self.quota.headers, {})

    async def test_a_failed_refund_is_only_logged(self):
        await self.quota.consume()
        with mock.patch.object(
            guest_quota, "delete_guest_usage_sync", side_effect=RuntimeError("db down")
        ):
            await self.quota.refund()
        self.assertEqual(guest_left("syllabus", GUEST), self.limit - 1)

    async def test_spend_refunds_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            async with self.quota.spend():
                raise RuntimeError("llm down")
        self.assertEqual(guest_left("syllabus", GUEST), self.limit)

    async def test_spend_keeps_the_use_on_success(self):
        async with self.quota.spend():
            pass
        self.assertEqual(guest_left("syllabus", GUEST), self.limit - 1)

    async def test_unlimited_caller_is_never_counted(self):
        quota = GuestQuota("syllabus", None, self.response)
        for _ in range(self.limit + 3):
            await quota.consume()
        await quota.refund()
        self.assertEqual(guest_left("syllabus", GUEST), self.limit)
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
        use_guest_usage_db(self, GUEST)

    def test_guest_sees_what_is_left(self):
        guest_use("chat", GUEST)

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

    def test_every_feature_is_listed(self):
        guest_use("feedback", GUEST)

        with restrictions(True):
            body = client.get(f"{BASE}/guest_quota", headers=self.headers).json()

        limits = {
            "chat": settings.GUEST_CHAT_DAILY_LIMIT,
            "syllabus": settings.GUEST_SYLLABUS_DAILY_LIMIT,
            "feedback": settings.GUEST_FEEDBACK_DAILY_LIMIT,
            "files": settings.GUEST_FILES_DAILY_LIMIT,
            "adaptation": settings.GUEST_ADAPTATION_DAILY_LIMIT,
        }
        for feature, limit in limits.items():
            self.assertEqual(body[feature]["limit"], limit)
        self.assertEqual(body["feedback"]["remaining"], limits["feedback"] - 1)
        self.assertEqual(body["files"]["remaining"], limits["files"])

    def test_reading_the_quota_uses_nothing(self):
        with restrictions(True):
            for _ in range(3):
                client.get(f"{BASE}/guest_quota", headers=self.headers)
        self.assertEqual(guest_left("chat", GUEST), settings.GUEST_CHAT_DAILY_LIMIT)

    def test_restrictions_off_is_not_limited(self):
        with restrictions(False):
            response = client.get(f"{BASE}/guest_quota", headers=self.headers)

        self.assertEqual(
            response.json(),
            {
                "limited": False,
                "chat": None,
                "syllabus": None,
                "feedback": None,
                "files": None,
                "adaptation": None,
            },
        )

    def test_guest_without_session_gets_401(self):
        with restrictions(True):
            response = client.get(f"{BASE}/guest_quota", headers={"X-API-Key": "test"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"]["code"], "SESSION_NOT_FOUND")
