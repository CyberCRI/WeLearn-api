import uuid
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

from starlette.requests import Request

from src.app.shared.domain.constants import SESSION_COOKIE_NAME
from src.app.shared.utils.decorators import (
    log_time_and_error,
    log_time_and_error_sync,
    singleton,
)
from src.app.shared.utils.requests import (
    extract_origin_from_request,
    extract_session_cookie,
)


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw})


class ExtractSessionCookieTests(TestCase):
    def test_cookie_value_is_parsed(self):
        sid = uuid.uuid4()
        req = _request({"cookie": f"{SESSION_COOKIE_NAME}={sid}"})
        self.assertEqual(extract_session_cookie(req), sid)

    def test_header_is_used_when_no_cookie(self):
        sid = uuid.uuid4()
        req = _request({"X-Session-Id": str(sid)})
        self.assertEqual(extract_session_cookie(req), sid)

    def test_missing_session_returns_none(self):
        self.assertIsNone(extract_session_cookie(_request({})))

    def test_invalid_uuid_returns_none(self):
        req = _request({"X-Session-Id": "not-a-uuid"})
        self.assertIsNone(extract_session_cookie(req))


class ExtractOriginTests(TestCase):
    def test_origin_header(self):
        req = _request({"origin": "https://welearn.example"})
        self.assertEqual(extract_origin_from_request(req), "https://welearn.example")

    def test_missing_origin(self):
        self.assertEqual(extract_origin_from_request(_request({})), "unknown")


@patch("src.app.shared.utils.decorators.logger")
class LogTimeAndErrorAsyncTests(IsolatedAsyncioTestCase):
    async def test_returns_result_and_logs_relevant_kwargs(self, logger_mock):
        @log_time_and_error
        async def search(collection_name=None, model=None, other=None):
            return "ok"

        result = await search(collection_name="col", model="m", other="ignored")

        self.assertEqual(result, "ok")
        self.assertEqual(search.__name__, "search")
        fmt = logger_mock.info.call_args.args[0]
        self.assertIn("collection_name=col", fmt)
        self.assertIn("model=m", fmt)
        self.assertNotIn("other", fmt)

    async def test_logs_and_reraises_errors(self, logger_mock):
        @log_time_and_error
        async def failing():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            await failing()
        logger_mock.error.assert_called_once()
        self.assertEqual(logger_mock.error.call_args.args[1], "failing")


@patch("src.app.shared.utils.decorators.logger")
class LogTimeAndErrorSyncTests(TestCase):
    def test_returns_result_and_logs_relevant_kwargs(self, logger_mock):
        @log_time_and_error_sync
        def compute(collection_x=None, model=None):
            return 42

        self.assertEqual(compute(collection_x="c", model="mm"), 42)
        fmt = logger_mock.info.call_args.args[0]
        self.assertIn("collection_x=c", fmt)
        self.assertIn("model=mm", fmt)

    def test_logs_and_reraises_errors(self, logger_mock):
        @log_time_and_error_sync
        def failing():
            raise KeyError("missing")

        with self.assertRaises(KeyError):
            failing()
        logger_mock.error.assert_called_once()


class SingletonTests(TestCase):
    def test_same_instance_is_returned(self):
        init = MagicMock()

        @singleton
        class Service:
            def __init__(self, value):
                init(value)
                self.value = value

        first = Service(1)
        second = Service(2)

        self.assertIs(first, second)
        self.assertEqual(second.value, 1)
        init.assert_called_once_with(1)
