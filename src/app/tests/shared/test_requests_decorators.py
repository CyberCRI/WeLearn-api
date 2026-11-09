import asyncio
import inspect
import uuid
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

from starlette.requests import Request

from src.app.shared.domain.constants import SESSION_COOKIE_NAME
from src.app.shared.utils.decorators import log_time_and_error, singleton
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


class FakeClock:
    """Stands in for ``time`` in the decorators module; the decorated
    function advances it to simulate work."""

    def __init__(self):
        self.now = 100.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def _logged_latency(logger_mock) -> float:
    # logger.info(fmt, method_name, latency)
    return logger_mock.info.call_args.args[2]


@patch("src.app.shared.utils.decorators.logger")
class LogTimeAndErrorAsyncTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = FakeClock()
        patcher = patch("src.app.shared.utils.decorators.time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_returns_result_and_logs_relevant_kwargs(self, logger_mock):
        @log_time_and_error
        async def search(collection_name=None, model=None, other=None):
            return "ok"

        result = await search(collection_name="col", model="m", other="ignored")

        self.assertEqual(result, "ok")
        fmt = logger_mock.info.call_args.args[0]
        self.assertIn("collection_name=col", fmt)
        self.assertIn("model=m", fmt)
        self.assertNotIn("other", fmt)
        self.assertEqual(logger_mock.info.call_args.args[1], "search")

    async def test_latency_covers_the_awaited_body(self, logger_mock):
        # D-2: the old log_time_and_error_sync on an async function only timed
        # the creation of the coroutine, so it logged a latency of 0.
        clock = self.clock

        @log_time_and_error
        async def slow():
            await asyncio.sleep(0)
            clock.advance(2.5)
            return "done"

        self.assertEqual(await slow(), "done")
        self.assertEqual(_logged_latency(logger_mock), 2.5)

    async def test_logs_and_reraises_errors_raised_while_awaiting(self, logger_mock):
        # D-2: the old sync decorator returned the coroutine before its body
        # ran, so an exception raised on await was never logged.
        @log_time_and_error
        async def failing():
            await asyncio.sleep(0)
            raise ValueError("boom")

        with self.assertRaises(ValueError) as ctx:
            await failing()

        logger_mock.error.assert_called_once()
        self.assertEqual(logger_mock.error.call_args.args[1], "failing")
        # non-database errors are logged with their message (see db_errors)
        self.assertEqual(logger_mock.error.call_args.args[2], str(ctx.exception))
        logger_mock.info.assert_not_called()

    async def test_wrapper_is_a_coroutine_function_and_keeps_metadata(self, _):
        async def original():
            """Doc."""

        wrapped = log_time_and_error(original)

        self.assertTrue(inspect.iscoroutinefunction(wrapped))
        self.assertEqual(wrapped.__name__, "original")
        self.assertEqual(wrapped.__doc__, "Doc.")
        self.assertIs(wrapped.__wrapped__, original)


@patch("src.app.shared.utils.decorators.logger")
class LogTimeAndErrorSyncTests(TestCase):
    def setUp(self):
        self.clock = FakeClock()
        patcher = patch("src.app.shared.utils.decorators.time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_result_and_logs_relevant_kwargs(self, logger_mock):
        @log_time_and_error
        def compute(collection_x=None, model=None):
            return 42

        self.assertEqual(compute(collection_x="c", model="mm"), 42)
        fmt = logger_mock.info.call_args.args[0]
        self.assertIn("collection_x=c", fmt)
        self.assertIn("model=mm", fmt)
        self.assertEqual(logger_mock.info.call_args.args[1], "compute")

    def test_logs_latency_of_the_call(self, logger_mock):
        clock = self.clock

        @log_time_and_error
        def slow():
            clock.advance(1.25)
            return "done"

        self.assertEqual(slow(), "done")
        self.assertEqual(_logged_latency(logger_mock), 1.25)

    def test_logs_and_reraises_errors(self, logger_mock):
        @log_time_and_error
        def failing():
            raise KeyError("missing")

        with self.assertRaises(KeyError) as ctx:
            failing()

        logger_mock.error.assert_called_once()
        self.assertEqual(logger_mock.error.call_args.args[1], "failing")
        # non-database errors are logged with their message (see db_errors)
        self.assertEqual(logger_mock.error.call_args.args[2], str(ctx.exception))
        logger_mock.info.assert_not_called()

    def test_wrapper_stays_sync_and_keeps_metadata(self, _):
        def original():
            """Doc."""

        wrapped = log_time_and_error(original)

        self.assertFalse(inspect.iscoroutinefunction(wrapped))
        self.assertEqual(wrapped.__name__, "original")
        self.assertEqual(wrapped.__doc__, "Doc.")
        self.assertIs(wrapped.__wrapped__, original)


class LogTimeAndErrorGeneratorTests(TestCase):
    def test_async_generator_is_rejected(self):
        async def agen():
            yield 1

        with self.assertRaises(TypeError):
            log_time_and_error(agen)

    def test_generator_is_rejected(self):
        def gen():
            yield 1

        with self.assertRaises(TypeError):
            log_time_and_error(gen)


class DecoratedAsyncFunctionsTests(TestCase):
    """D-2: these async functions used to be wrapped by the sync decorator."""

    def test_previously_misdecorated_functions_are_coroutine_functions(self):
        from src.app.micro_learning.services.helpers import (
            collection_and_model_id_according_lang,
        )
        from src.app.search.services.search import SearchService
        from src.app.shared.utils.utils import get_files_content

        for func in (
            SearchService.get_query_embed,
            SearchService._embed_query,
            collection_and_model_id_according_lang,
            get_files_content,
        ):
            with self.subTest(func=func.__qualname__):
                self.assertTrue(inspect.iscoroutinefunction(func))
                self.assertTrue(inspect.iscoroutinefunction(func.__wrapped__))


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
