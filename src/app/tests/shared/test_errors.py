from unittest import TestCase

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.app.shared.api.errors import register_exception_handlers
from src.app.shared.domain.exceptions import (
    CollectionNotFoundError,
    EmptyQueryError,
    LanguageNotSupportedError,
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
