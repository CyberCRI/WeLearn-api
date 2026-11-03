from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, Response, status

from src.app.shared.domain.exceptions import (
    CollectionNotFoundError,
    EmptyQueryError,
    LanguageNotSupportedError,
    ModelNotFoundError,
    NoResultsError,
    PartialResponseResultError,
    SessionNotFoundError,
    SubjectNotFoundError,
    UserNotFoundError,
    handle_error,
)


class DomainErrorDefaultsTests(TestCase):
    def test_session_not_found_error(self):
        err = SessionNotFoundError()
        self.assertEqual(err.message, "Session not found")
        self.assertEqual(err.msg_code, "SESSION_NOT_FOUND")
        self.assertNotIsInstance(err, Exception)

    def test_user_not_found_error(self):
        err = UserNotFoundError(message="custom", msg_code="CODE")
        self.assertEqual(err.message, "custom")
        self.assertEqual(err.msg_code, "CODE")
        self.assertEqual(err.args, ("custom", "CODE"))


class HandleErrorTests(TestCase):
    def _assert_http(self, exc, expected_status):
        with self.assertRaises(HTTPException) as ctx:
            handle_error(exc)
        self.assertEqual(ctx.exception.status_code, expected_status)
        self.assertEqual(
            ctx.exception.detail, {"message": exc.message, "code": exc.msg_code}
        )

    def test_partial_result_sets_206_on_response(self):
        response = Response()
        handle_error(PartialResponseResultError(), response)
        self.assertEqual(response.status_code, status.HTTP_206_PARTIAL_CONTENT)

    def test_partial_result_without_response_does_not_raise(self):
        self.assertIsNone(handle_error(PartialResponseResultError()))

    def test_no_results_raises_204(self):
        self._assert_http(NoResultsError(), status.HTTP_204_NO_CONTENT)

    def test_language_not_supported_raises_400(self):
        self._assert_http(LanguageNotSupportedError(), status.HTTP_400_BAD_REQUEST)

    def test_collection_not_found_raises_404(self):
        self._assert_http(CollectionNotFoundError(), status.HTTP_404_NOT_FOUND)

    def test_model_not_found_raises_404(self):
        self._assert_http(ModelNotFoundError(), status.HTTP_404_NOT_FOUND)

    def test_empty_query_raises_400(self):
        self._assert_http(EmptyQueryError(), status.HTTP_400_BAD_REQUEST)

    def test_subject_not_found_raises_404(self):
        self._assert_http(SubjectNotFoundError(), status.HTTP_404_NOT_FOUND)

    @patch("src.app.shared.domain.exceptions.logger")
    def test_unknown_exception_is_only_logged(self, logger_mock: MagicMock):
        exc = ValueError("boom")
        self.assertIsNone(handle_error(exc))
        logger_mock.error.assert_called_once_with("Unhandled exception: %s", exc)
