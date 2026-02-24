import unittest

from fastapi import HTTPException

from src.app.services.exceptions import (
    CollectionNotFoundError,
    EmptyQueryError,
    InvalidQuestionError,
    LanguageNotSupportedError,
    ModelNotFoundError,
    NoResultsError,
    PartialResponseResultError,
    SubjectNotFoundError,
    bad_request,
    no_content,
    not_found,
)


class ExceptionsTests(unittest.TestCase):
    def test_bad_request(self):
        self.assertRaises(HTTPException, bad_request, "message ", "msg_code")
        try:
            bad_request("message ", "msg_code")
        except HTTPException as e:
            self.assertEqual(e.status_code, 400)

    def test_no_content(self):
        self.assertRaises(HTTPException, no_content, "message ", "msg_code")
        try:
            no_content("message ", "msg_code")
        except HTTPException as e:
            self.assertEqual(e.status_code, 204)

    def test_not_found(self):
        self.assertRaises(HTTPException, not_found, "message ", "msg_code")
        try:
            not_found("message ", "msg_code")
        except HTTPException as e:
            self.assertEqual(e.status_code, 404)

    def test_empty_query_error(self):
        error = EmptyQueryError()
        self.assertEqual(error.message, "Empty query")
        self.assertEqual(error.msg_code, "EMPTY_QUERY")

    def test_language_not_supported_error(self):
        error = LanguageNotSupportedError()
        self.assertEqual(error.message, "Language not supported")
        self.assertEqual(error.msg_code, "LANG_NOT_SUPPORTED")

    def test_invalid_question_error(self):
        error = InvalidQuestionError()
        self.assertEqual(error.message, "Please provide a valid question")
        self.assertEqual(error.msg_code, "INVALID_QUESTION")

    def test_no_results_error(self):
        error = NoResultsError()
        self.assertEqual(error.message, "No results found")
        self.assertEqual(error.msg_code, "NO_RESULTS")

    def test_collection_not_found_error(self):
        error = CollectionNotFoundError()
        self.assertEqual(error.message, "Collection not found")
        self.assertEqual(error.msg_code, "COLL_NOT_FOUND")

    def test_model_not_found_error(self):
        error = ModelNotFoundError()
        self.assertEqual(error.message, "Model not found")
        self.assertEqual(error.msg_code, "MODEL_NOT_FOUND")

    def test_partial_response_result_error(self):
        error = PartialResponseResultError()
        self.assertEqual(error.message, "Partial response result")
        self.assertEqual(error.msg_code, "PARTIAL_RESULT")

    def test_subject_not_found_error(self):
        error = SubjectNotFoundError()
        self.assertEqual(error.message, "Subject not found")
        self.assertEqual(error.msg_code, "SUBJECT_NOT_FOUND")
