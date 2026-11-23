import unittest
import uuid
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from starlette.requests import Request

from src.app.shared.domain.exceptions import CourseTooLargeError
from src.app.shared.infra.rate_limit import InFlightLimiter, SlidingWindowRateLimiter
from src.app.tests.tutor.test_tutor_api import BASE, HEADERS, TutorApiTestCase
from src.app.tutor.domain.connections import ConfirmedCourse, ConnectionIdeasResponse
from src.app.tutor.services import course_limits
from src.app.tutor.services.course_limits import (
    course_llm_request,
    ensure_course_fits,
    limit_key,
)

ROUTER = "src.app.tutor.api.router"
SESSION = "11111111-2222-3333-4444-555555555555"


def request(session=None, ip="10.0.0.1"):
    headers = [(b"x-session-id", session.encode())] if session else []
    return Request({"type": "http", "headers": headers, "client": (ip, 1234)})


class LimitKeyTests(unittest.TestCase):
    def test_the_session_or_else_the_client_ip(self):
        self.assertEqual(limit_key(request(SESSION)), f"session:{SESSION}")
        self.assertEqual(limit_key(request()), "ip:10.0.0.1")
        self.assertEqual(limit_key(request("not-a-uuid")), "ip:10.0.0.1")


class CourseLlmRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for name, limiter in (
            ("_in_flight", InFlightLimiter(2)),
            ("_requests", SlidingWindowRateLimiter(3, 600)),
            ("_ip_requests", SlidingWindowRateLimiter(5, 600)),
        ):
            patcher = patch.object(course_limits, name, limiter)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def enter(self, req):
        dependency = course_llm_request(req)
        await dependency.__anext__()
        return dependency

    async def test_two_requests_of_a_session_at_once_not_three(self):
        first = await self.enter(request(SESSION))
        await self.enter(request(SESSION))

        with self.assertRaises(HTTPException) as raised:
            await self.enter(request(SESSION))
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.detail["code"], "TOO_MANY_REQUESTS_IN_FLIGHT")
        self.assertEqual(raised.exception.headers["Retry-After"], "5")
        self.assertEqual(raised.exception.detail["retry_after"], 5)
        # another session is not held back
        await self.enter(request("99999999-2222-3333-4444-555555555555"))

        # a request that ends frees its place
        with self.assertRaises(StopAsyncIteration):
            await first.__anext__()
        await self.enter(request(SESSION))

    async def test_requests_per_window_then_rate_limited(self):
        for _ in range(3):
            dependency = await self.enter(request(SESSION))
            with self.assertRaises(StopAsyncIteration):
                await dependency.__anext__()

        with self.assertRaises(HTTPException) as raised:
            await self.enter(request(SESSION))
        self.assertEqual(raised.exception.detail["code"], "RATE_LIMITED")
        self.assertGreaterEqual(int(raised.exception.headers["Retry-After"]), 1)
        self.assertEqual(
            str(raised.exception.detail["retry_after"]),
            raised.exception.headers["Retry-After"],
        )
        # the refused request did not keep a place
        self.assertEqual(course_limits._in_flight.running(f"session:{SESSION}"), 0)

    async def test_a_new_session_id_per_request_is_still_counted_by_its_ip(self):
        for _ in range(5):
            dependency = await self.enter(request(str(uuid.uuid4())))
            with self.assertRaises(StopAsyncIteration):
                await dependency.__anext__()

        with self.assertRaises(HTTPException) as raised:
            await self.enter(request(str(uuid.uuid4())))
        self.assertEqual(raised.exception.detail["code"], "RATE_LIMITED")


class EnsureCourseFitsTests(unittest.TestCase):
    def test_a_course_longer_than_a_prompt_block_is_refused(self):
        ensure_course_fits(ConfirmedCourse(title="Ecology"))
        with patch.object(course_limits, "MAX_DATA_CHARS", 20):
            with self.assertRaises(CourseTooLargeError):
                ensure_course_fits(ConfirmedCourse(overview="x" * 30))


class CourseRouteLimitsTests(TutorApiTestCase):
    """The limits apply to every /tutor/course/* route."""

    COURSE = {"title": "Ecology", "topics": [{"title": "Soils"}]}

    def setUp(self):
        super().setUp()
        patcher = patch.object(
            course_limits, "_requests", SlidingWindowRateLimiter(1, 600)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch(f"{ROUTER}.find_connection_ideas", new_callable=AsyncMock)
    def test_a_session_past_its_requests_gets_429(self, mock_find):
        mock_find.return_value = ConnectionIdeasResponse(
            ideas=[], none_reason="None fits."
        )
        headers = {**HEADERS, "X-Session-Id": SESSION}
        url = f"{BASE}/course/connection_ideas"
        first = self.safe_client.post(
            url, json={"course": self.COURSE}, headers=headers
        )

        self.assertEqual(first.status_code, 200)
        response = self.safe_client.post(
            url, json={"course": self.COURSE}, headers=headers
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "RATE_LIMITED")
        self.assertEqual(
            response.headers["Retry-After"],
            str(response.json()["detail"]["retry_after"]),
        )
        self.assertEqual(mock_find.await_count, 1)

    def test_every_course_route_is_limited(self):
        for path in ("read", "connection_ideas", "connection_evidence", "adaptation"):
            with self.subTest(path=path):
                route = next(
                    r
                    for r in self.client.app.routes
                    if r.path.endswith(f"/course/{path}")
                )
                calls = [d.call for d in route.dependant.dependencies]
                self.assertIn(course_llm_request, calls)

    @patch(f"{ROUTER}.find_connection_ideas", new_callable=AsyncMock)
    def test_a_course_too_large_is_413(self, mock_find):
        with patch.object(course_limits, "MAX_DATA_CHARS", 20):
            response = self.safe_client.post(
                f"{BASE}/course/connection_ideas",
                json={"course": {"overview": "x" * 30}},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "COURSE_TOO_LARGE")
        mock_find.assert_not_awaited()
