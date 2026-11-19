import asyncio
from unittest import TestCase

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError as SAOperationalError

from src.app.shared.api import errors as errors_module
from src.app.shared.api.errors import (
    INTERNAL_ERROR_DETAIL,
    UnhandledErrorMiddleware,
    describe_chain,
    register_exception_handlers,
)
from src.app.shared.domain.exceptions import (
    CollectionNotFoundError,
    EmptyQueryError,
    LanguageNotSupportedError,
    MetricsNotReadyError,
    ModelNotFoundError,
    NoResultsError,
    SessionNotFoundError,
    SubjectNotFoundError,
    UserNotFoundError,
    WeLearnError,
)

JSON_ERRORS = [
    SessionNotFoundError,
    UserNotFoundError,
    EmptyQueryError,
    LanguageNotSupportedError,
    CollectionNotFoundError,
    ModelNotFoundError,
    SubjectNotFoundError,
    MetricsNotReadyError,
]


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise/{name}")
    async def raise_error(name: str):
        exc_class = {c.__name__: c for c in [*JSON_ERRORS, NoResultsError]}[name]
        raise exc_class()

    @app.get("/raise_custom")
    def raise_custom():
        raise UserNotFoundError("User 42 does not exist")

    @app.get("/raise_base")
    def raise_base():
        raise WeLearnError()

    return app


class WeLearnErrorHandlerTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(_build_app())

    def test_each_domain_error_maps_to_its_status_and_detail(self):
        for exc_class in JSON_ERRORS:
            with self.subTest(exc_class=exc_class.__name__):
                response = self.client.get(f"/raise/{exc_class.__name__}")
                self.assertEqual(response.status_code, exc_class.status_code)
                self.assertEqual(
                    response.json(),
                    {
                        "detail": {
                            "message": exc_class.default_message,
                            "code": exc_class.default_code,
                        }
                    },
                )

    def test_no_results_returns_204_without_body(self):
        response = self.client.get("/raise/NoResultsError")
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(response.content, b"")

    def test_custom_message_is_returned(self):
        response = self.client.get("/raise_custom")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(
            response.json()["detail"],
            {"message": "User 42 does not exist", "code": "USER_NOT_FOUND"},
        )

    def test_base_error_returns_500(self):
        response = self.client.get("/raise_base")
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(response.json()["detail"]["code"], "INTERNAL_ERROR")


SECRET = "password=hunter2 host=db.internal"


def _build_cors_app() -> FastAPI:
    """Like main.py: the middleware added just before CORSMiddleware, so inside it."""
    app = FastAPI()
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["https://client.example.com"])

    @app.get("/crash")
    async def crash():
        try:
            raise RuntimeError("Already borrowed")
        except RuntimeError as e:
            raise RuntimeError("Not able to create embed", "EMBED_ERROR") from e

    @app.get("/db_crash")
    async def db_crash():
        raise RuntimeError("query failed") from SAOperationalError(
            "SELECT 1", {}, Exception(SECRET)
        )

    @app.get("/stream_crash")
    async def stream_crash():
        async def chunks():
            yield b"first"
            raise RuntimeError("stream broke")

        return StreamingResponse(chunks())

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    return app


class UnhandledErrorMiddlewareTests(TestCase):
    ORIGIN = {"Origin": "https://client.example.com"}

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(_build_cors_app(), raise_server_exceptions=False)

    def test_an_unhandled_error_is_a_500_the_browser_can_read(self):
        with self.assertLogs(errors_module.logger, level="ERROR") as logs:
            response = self.client.get("/crash", headers=self.ORIGIN)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": INTERNAL_ERROR_DETAIL})
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "https://client.example.com",
        )
        # the cause is logged, after the error it caused
        log = "\n".join(logs.output)
        self.assertIn("unhandled_error method=GET endpoint=/crash", log)
        self.assertIn("EMBED_ERROR", log)
        self.assertIn("caused by RuntimeError: Already borrowed", log)

    def test_a_database_cause_is_logged_without_its_message(self):
        with self.assertLogs(errors_module.logger, level="ERROR") as logs:
            response = self.client.get("/db_crash", headers=self.ORIGIN)

        self.assertEqual(response.status_code, 500)
        log = "\n".join(logs.output)
        self.assertIn("OperationalError", log)
        self.assertNotIn("hunter2", log)

    def test_a_response_already_started_is_not_replaced(self):
        client = TestClient(_build_cors_app())
        with self.assertRaises(RuntimeError):
            client.get("/stream_crash")

    def test_other_requests_are_untouched(self):
        response = self.client.get("/ok", headers=self.ORIGIN)
        self.assertEqual(response.json(), {"ok": True})

    def test_a_non_http_scope_goes_through(self):
        seen = []

        async def inner(scope, receive, send):
            seen.append(scope["type"])

        asyncio.run(UnhandledErrorMiddleware(inner)({"type": "lifespan"}, None, None))
        self.assertEqual(seen, ["lifespan"])

    def test_the_app_puts_it_inside_cors(self):
        from src.main import app

        classes = [m.cls for m in app.user_middleware]
        # user_middleware lists the outermost first
        self.assertLess(
            classes.index(CORSMiddleware), classes.index(UnhandledErrorMiddleware)
        )


class DescribeChainTests(TestCase):
    def test_each_error_once_even_in_a_cycle(self):
        first, second = ValueError("a"), KeyError("b")
        first.__cause__, second.__cause__ = second, first
        text = describe_chain(first)
        self.assertEqual(text.count("ValueError"), 1)
        self.assertEqual(text.count("KeyError"), 1)
