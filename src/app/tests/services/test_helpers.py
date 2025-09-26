from unittest import TestCase, mock

from langdetect.language import Language

from src.app.models.documents import Document, DocumentPayloadModel
from src.app.services.exceptions import LanguageNotSupportedError
from src.app.services.helpers import (
    detect_language_from_entry,
    extract_json_from_response,
    stringify_docs_content,
)


class HelpersTests(TestCase):
    def test_detect_language_from_entry_no_lang(self):
        with mock.patch("src.app.services.helpers.detect_langs", return_value=[]):
            with self.assertRaises(LanguageNotSupportedError):
                detect_language_from_entry("test")

    def test_detect_language_from_entry(self):
        with mock.patch(
            "src.app.services.helpers.detect_langs", return_value=[Language("en", 1.0)]
        ):
            self.assertEqual(detect_language_from_entry("test again"), "en")

    def test_detect_language_from_entry_lang_not_supported(self):
        with mock.patch(
            "src.app.services.helpers.detect_langs", return_value=[Language("es", 1.0)]
        ):
            with self.assertRaises(LanguageNotSupportedError):
                detect_language_from_entry("test es")

    def test_langdetect_error(self):
        with mock.patch("src.app.services.helpers.detect_langs", side_effect=Exception):
            with self.assertRaises(LanguageNotSupportedError):
                detect_language_from_entry("test es")

    def test_stringify_docs_content(self):
        docs = [
            Document(
                score=0.5,
                payload=DocumentPayloadModel(
                    document_corpus="test",
                    document_desc="desc",
                    document_details={},
                    document_id="1",
                    document_lang="en",
                    document_sdg=[],
                    document_title="title",
                    document_url="url",
                    slice_content="content",
                    slice_sdg=None,
                ),
            ),
            Document(
                score=0.7,
                payload=DocumentPayloadModel(
                    document_corpus="test",
                    document_desc="desc",
                    document_details={},
                    document_id="2",
                    document_lang="en",
                    document_sdg=[],
                    document_title="title 2",
                    document_url="url 2",
                    slice_content="content 2",
                    slice_sdg=None,
                ),
            ),
        ]

        expected = """<article>\nDoc 1: title\ncontent\n\nurl:url</article>

<article>\nDoc 2: title 2\ncontent 2\n\nurl:url 2</article>"""

        self.assertEqual(stringify_docs_content(docs), expected)

    def test_stringify_docs_content_error(self):
        docs = [
            {"score": 0.5, "payload": {"document_corpus": "test"}},
            {"score": 0.7, "payload": {"document_corpus": "test"}},
        ]

        with mock.patch("src.app.services.helpers.logger.error") as mock_logger:
            self.assertEqual(stringify_docs_content(docs), "")
            mock_logger.assert_called_once()

    def test_extract_json_from_response(self):
        response = 'Here is the JSON: {"key": "value"}'
        expected = {"key": "value"}
        self.assertEqual(extract_json_from_response(response), expected)

    def test_extract_json_from_response_no_json(self):
        response = "Here is the JSON: "
        with self.assertRaises(ValueError):
            extract_json_from_response(response)
