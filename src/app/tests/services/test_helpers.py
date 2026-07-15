from unittest import TestCase, mock

import numpy
from langdetect.language import Language

from src.app.models.documents import Document, DocumentPayloadModel
from src.app.services.helpers import (
    convert_embedding_bytes,
    detect_language_from_entry,
    extract_json_from_response,
    linkify_missing_citations,
    stringify_docs_content,
)
from src.app.shared.domain.exceptions import LanguageNotSupportedError


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

    def test_stringify_docs_content(self):
        docs = [
            Document(
                score=0.5,
                payload=DocumentPayloadModel(
                    document_corpus="test",
                    document_desc="desc",
                    document_details={},
                    document_id="12345678-1234-5678-1234-567812345678",
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
                    document_id="12345677-1234-5678-1234-567812345678",
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

        self.assertEqual(stringify_docs_content(docs), "")

    def test_extract_json_from_response(self):
        response = 'Here is the JSON: {"key": "value"}'
        expected = {"key": "value"}
        self.assertEqual(extract_json_from_response(response), expected)

    def test_extract_json_from_response_no_json(self):
        response = "Here is the JSON: "
        with self.assertRaises(ValueError):
            extract_json_from_response(response)

    def _make_doc(self, url: str) -> Document:
        return Document(
            score=0.5,
            payload=DocumentPayloadModel(
                document_corpus="test",
                document_desc="desc",
                document_details={},
                document_id="12345678-1234-5678-1234-567812345678",
                document_lang="en",
                document_sdg=[],
                document_title="title",
                document_url=url,
                slice_content="content",
                slice_sdg=None,
            ),
        )

    def test_linkify_missing_citations_wraps_bare_marker(self):
        docs = [self._make_doc("https://example.org/1")]
        text = "Sustainability matters [Doc 1]."
        expected = (
            'Sustainability matters <a href="https://example.org/1" '
            'target="_blank">[Doc 1]</a>.'
        )
        self.assertEqual(linkify_missing_citations(text, docs), expected)

    def test_linkify_missing_citations_leaves_existing_link_untouched(self):
        docs = [self._make_doc("https://example.org/1")]
        text = 'Already linked <a href="https://example.org/1" target="_blank">[Doc 1]</a>.'
        self.assertEqual(linkify_missing_citations(text, docs), text)

    def test_linkify_missing_citations_mixed_bare_and_linked(self):
        docs = [
            self._make_doc("https://example.org/1"),
            self._make_doc("https://example.org/2"),
        ]
        text = (
            'See <a href="https://example.org/1" target="_blank">[Doc 1]</a> '
            "and also [Doc 2]."
        )
        expected = (
            'See <a href="https://example.org/1" target="_blank">[Doc 1]</a> '
            'and also <a href="https://example.org/2" target="_blank">[Doc 2]</a>.'
        )
        self.assertEqual(linkify_missing_citations(text, docs), expected)

    def test_linkify_missing_citations_out_of_range_untouched(self):
        docs = [self._make_doc("https://example.org/1")]
        text = "See [Doc 9] for more."
        self.assertEqual(linkify_missing_citations(text, docs), text)

    def test_linkify_missing_citations_empty_docs_or_text(self):
        docs = [self._make_doc("https://example.org/1")]
        self.assertEqual(linkify_missing_citations("", docs), "")
        self.assertEqual(linkify_missing_citations("[Doc 1]", []), "[Doc 1]")

    def test_linkify_missing_citations_dict_payload_fallback(self):
        docs = [{"document_url": "https://example.org/1"}]
        text = "See [Doc 1] for more."
        expected = (
            'See <a href="https://example.org/1" target="_blank">[Doc 1]</a> for more.'
        )
        self.assertEqual(linkify_missing_citations(text, docs), expected)

    def test_convert_embedding_bytes(self):
        x = numpy.random.rand(
            5,
        )
        ret = convert_embedding_bytes(embeddings_byte=x.tobytes(), dtype=numpy.float64)

        self.assertEqual(x.tolist(), ret.tolist())
