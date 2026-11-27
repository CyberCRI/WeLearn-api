import asyncio
import unittest
import uuid
from unittest import mock

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import OperationalError as SAOperationalError

from src.app.core.config import settings
from src.app.shared.api.monitor_requests import MonitorRequestsMiddleware
from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.user.services import account, guest_quota
from src.main import app

MONITOR = "src.app.shared.api.monitor_requests"
API_KEY = "src.app.shared.infra.security.check_api_key_sync"
QUOTA_URL = f"{settings.API_V1_STR}/user/guest_quota"
SESSION = uuid.uuid4()
HEADERS = {"X-API-Key": "test", "X-Session-Id": str(SESSION)}


class MonitorRequestsTests(unittest.TestCase):
    """The row of each request, through the middleware stack of the app."""

    def setUp(self):
        patcher = mock.patch(f"{MONITOR}.register_endpoint")
        self.register = patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def get(self, headers=HEADERS, api_key_result=True):
        with mock.patch(
            API_KEY,
            **(
                {"side_effect": api_key_result}
                if isinstance(api_key_result, BaseException)
                else {"return_value": api_key_result}
            ),
        ):
            return self.client.get(QUOTA_URL, headers=headers)

    def assert_recorded(self, http_code, message=None):
        self.register.assert_called_once_with(
            endpoint=QUOTA_URL,
            session_id=SESSION,
            http_code=http_code,
            message=message,
        )

    def test_a_success_is_recorded_as_200(self):
        with mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", False):
            response = self.get()

        self.assertEqual(response.status_code, 200)
        self.assert_recorded(200)

    def test_a_domain_error_is_recorded_with_its_status_and_code(self):
        with (
            mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", True),
            mock.patch.object(
                guest_quota,
                "get_user_for_session_renewal",
                side_effect=SessionNotFoundError(),
            ),
        ):
            response = self.get()

        self.assertEqual(response.status_code, 401)
        self.assert_recorded(401, "SESSION_NOT_FOUND")

    def test_an_http_error_is_recorded_with_its_code(self):
        error = HTTPException(429, detail={"message": "slow", "code": "RATE_LIMITED"})

        response = self.get(api_key_result=error)

        self.assertEqual(response.status_code, 429)
        self.assert_recorded(429, "RATE_LIMITED")

    def test_an_error_without_code_has_no_message(self):
        response = self.get(api_key_result=False)

        self.assertEqual(response.status_code, 401)
        self.assert_recorded(401)

    def test_a_database_error_is_recorded_with_its_status(self):
        response = self.get(api_key_result=SAOperationalError("SELECT", {}, None))

        self.assertEqual(response.status_code, 503)
        self.assert_recorded(503, "DATABASE_UNAVAILABLE")

    def test_an_unhandled_error_is_recorded_as_500(self):
        response = self.get(api_key_result=RuntimeError("boom"))

        self.assertEqual(response.status_code, 500)
        self.assert_recorded(500, "INTERNAL_ERROR")

    def test_nothing_is_recorded_without_a_session(self):
        response = self.get(headers={"X-API-Key": "test"})

        self.assertEqual(response.status_code, 200)
        self.register.assert_not_called()

    def test_nothing_is_recorded_outside_the_api(self):
        self.client.get("/", headers=HEADERS)

        self.register.assert_not_called()

    def test_an_unknown_session_is_skipped_quietly(self):
        self.register.side_effect = IntegrityError("INSERT", {}, Exception("fk"))

        with (
            mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", False),
            self.assertNoLogs(MONITOR, "WARNING"),
        ):
            response = self.get()

        self.assertEqual(response.status_code, 200)

    def test_a_failed_record_does_not_change_the_response(self):
        self.register.side_effect = SAOperationalError("INSERT", {}, None)

        with (
            mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", False),
            self.assertLogs(MONITOR, "ERROR"),
        ):
            response = self.get()

        self.assertEqual(response.status_code, 200)


class MonitorStreamTests(unittest.TestCase):
    """Recording waits for the start of a response, not for its body."""

    def test_a_stream_is_recorded_before_its_body_is_sent(self):
        events: list[str] = []
        stream_app = FastAPI()
        stream_app.add_middleware(MonitorRequestsMiddleware)

        @stream_app.get("/api/v1/stream")
        async def stream():
            async def chunks():
                yield "first;"
                # only asked for once the middleware passed the first chunk on
                events.append("second chunk")
                await asyncio.sleep(0)
                yield "second"

            return StreamingResponse(chunks(), media_type="text/event-stream")

        with mock.patch(
            f"{MONITOR}.register_endpoint",
            side_effect=lambda **kwargs: events.append(
                f"recorded {kwargs['http_code']}"
            ),
        ):
            response = TestClient(stream_app).get("/api/v1/stream", headers=HEADERS)

        self.assertEqual(response.text, "first;second")
        self.assertEqual(events, ["recorded 200", "second chunk"])


if __name__ == "__main__":
    unittest.main()
