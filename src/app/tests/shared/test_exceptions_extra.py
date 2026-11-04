from unittest import TestCase

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
    WeLearnError,
    handle_error,
)

EXPECTED = [
    (SessionNotFoundError, status.HTTP_401_UNAUTHORIZED, "SESSION_NOT_FOUND"),
    (UserNotFoundError, status.HTTP_404_NOT_FOUND, "USER_NOT_FOUND"),
    (EmptyQueryError, status.HTTP_400_BAD_REQUEST, "EMPTY_QUERY"),
    (LanguageNotSupportedError, status.HTTP_400_BAD_REQUEST, "LANG_NOT_SUPPORTED"),
    (NoResultsError, status.HTTP_204_NO_CONTENT, "NO_RESULTS"),
    (CollectionNotFoundError, status.HTTP_404_NOT_FOUND, "COLL_NOT_FOUND"),
    (ModelNotFoundError, status.HTTP_404_NOT_FOUND, "MODEL_NOT_FOUND"),
    (SubjectNotFoundError, status.HTTP_404_NOT_FOUND, "SUBJECT_NOT_FOUND"),
    (PartialResponseResultError, status.HTTP_206_PARTIAL_CONTENT, "PARTIAL_RESULT"),
]


class DomainErrorDefaultsTests(TestCase):
    def test_every_domain_error_is_a_welearn_exception(self):
        for exc_class, expected_status, expected_code in EXPECTED:
            with self.subTest(exc_class=exc_class.__name__):
                err = exc_class()
                self.assertIsInstance(err, Exception)
                self.assertIsInstance(err, WeLearnError)
                self.assertEqual(err.status_code, expected_status)
                self.assertEqual(err.msg_code, expected_code)
                self.assertEqual(
                    err.detail, {"message": err.message, "code": expected_code}
                )

    def test_caught_by_except_exception(self):
        try:
            raise SessionNotFoundError()
        except Exception as e:  # noqa: BLE001 - this is what the test checks
            caught = e
        self.assertIsInstance(caught, SessionNotFoundError)

    def test_session_not_found_error_defaults(self):
        err = SessionNotFoundError()
        self.assertEqual(err.message, "Session not found")
        self.assertEqual(err.msg_code, "SESSION_NOT_FOUND")
        self.assertEqual(err.args, ("Session not found", "SESSION_NOT_FOUND"))

    def test_user_not_found_error_custom(self):
        err = UserNotFoundError(message="custom", msg_code="CODE")
        self.assertEqual(err.message, "custom")
        self.assertEqual(err.msg_code, "CODE")
        self.assertEqual(err.args, ("custom", "CODE"))

    def test_positional_message_keeps_default_code(self):
        err = UserNotFoundError("User 42 does not exist")
        self.assertEqual(err.message, "User 42 does not exist")
        self.assertEqual(err.msg_code, "USER_NOT_FOUND")


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

    def test_session_not_found_raises_401(self):
        self._assert_http(SessionNotFoundError(), status.HTTP_401_UNAUTHORIZED)

    def test_unknown_exception_is_re_raised(self):
        exc = ValueError("boom")
        with self.assertRaises(ValueError) as ctx:
            handle_error(exc)
        self.assertIs(ctx.exception, exc)
