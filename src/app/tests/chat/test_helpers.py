from unittest import TestCase

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.app.chat.services.helpers import (
    latest_tool_docs,
    linkify_missing_citations,
    stringify_docs_content,
)
from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)


class ChatHelpersTests(TestCase):
    def test_stringify_docs_content(self):
        docs = [
            EnhancedDocumentSlice(
                score=0.5,
                payload=EnhancedDocumentSlicePayloadModel(
                    document_corpus="test",
                    document_sub_corpus="subtest",
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
            EnhancedDocumentSlice(
                score=0.7,
                payload=EnhancedDocumentSlicePayloadModel(
                    document_corpus="test",
                    document_sub_corpus="subtest",
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

    def _make_doc(self, url: str) -> EnhancedDocumentSlice:
        return EnhancedDocumentSlice(
            score=0.5,
            payload=EnhancedDocumentSlicePayloadModel(
                document_corpus="test",
                document_sub_corpus="subtest",
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
        expected = "Sustainability matters [[Doc 1]](https://example.org/1)."
        self.assertEqual(linkify_missing_citations(text, docs), expected)

    def test_linkify_missing_citations_leaves_existing_double_bracket_link_untouched(
        self,
    ):
        docs = [self._make_doc("https://example.org/1")]
        text = "Already linked [[Doc 1]](https://example.org/1)."
        self.assertEqual(linkify_missing_citations(text, docs), text)

    def test_linkify_missing_citations_leaves_existing_markdown_link_untouched(self):
        docs = [self._make_doc("https://example.org/1")]
        text = "Already linked [Doc 1](https://example.org/1)."
        self.assertEqual(linkify_missing_citations(text, docs), text)

    def test_linkify_missing_citations_leaves_existing_html_link_untouched(self):
        docs = [self._make_doc("https://example.org/1")]
        text = 'Already linked <a href="https://example.org/1" target="_blank">[Doc 1]</a>.'
        self.assertEqual(linkify_missing_citations(text, docs), text)

    def test_linkify_missing_citations_mixed_bare_and_linked(self):
        docs = [
            self._make_doc("https://example.org/1"),
            self._make_doc("https://example.org/2"),
        ]
        text = "See [[Doc 1]](https://example.org/1) and also [Doc 2]."
        expected = (
            "See [[Doc 1]](https://example.org/1) "
            "and also [[Doc 2]](https://example.org/2)."
        )
        self.assertEqual(linkify_missing_citations(text, docs), expected)

    def test_linkify_missing_citations_out_of_range_untouched(self):
        docs = [self._make_doc("https://example.org/1")]
        text = "See [Doc 9] for more."
        self.assertEqual(linkify_missing_citations(text, docs), text)

    def test_linkify_missing_citations_combined_marker_untouched(self):
        docs = [
            self._make_doc("https://example.org/1"),
            self._make_doc("https://example.org/2"),
            self._make_doc("https://example.org/3"),
        ]
        text = "See [Docs 3 et 5] for more."
        self.assertEqual(linkify_missing_citations(text, docs), text)

    def test_linkify_missing_citations_empty_docs_or_text(self):
        docs = [self._make_doc("https://example.org/1")]
        self.assertEqual(linkify_missing_citations("", docs), "")
        self.assertEqual(linkify_missing_citations("[Doc 1]", []), "[Doc 1]")

    def test_linkify_missing_citations_dict_payload_fallback(self):
        docs = [{"document_url": "https://example.org/1"}]
        text = "See [Doc 1] for more."
        expected = "See [[Doc 1]](https://example.org/1) for more."
        self.assertEqual(linkify_missing_citations(text, docs), expected)

    def test_latest_tool_docs_returns_most_recent_call_only(self):
        earlier_docs = [self._make_doc("https://example.org/earlier")]
        latest_docs = [self._make_doc("https://example.org/latest")]
        messages = [
            HumanMessage(content="first question"),
            AIMessage(content=""),
            ToolMessage(content="...", tool_call_id="1", artifact=earlier_docs),
            AIMessage(content="answer using earlier docs"),
            HumanMessage(content="follow-up question"),
            AIMessage(content=""),
            ToolMessage(content="...", tool_call_id="2", artifact=latest_docs),
            AIMessage(content="answer using latest docs"),
        ]
        self.assertEqual(latest_tool_docs(messages), latest_docs)

    def test_latest_tool_docs_no_new_call_falls_back_to_last_one(self):
        docs = [self._make_doc("https://example.org/only")]
        messages = [
            HumanMessage(content="first question"),
            AIMessage(content=""),
            ToolMessage(content="...", tool_call_id="1", artifact=docs),
            AIMessage(content="answer"),
            HumanMessage(content="follow-up that needs no new search"),
            AIMessage(content="answer reusing the same docs"),
        ]
        self.assertEqual(latest_tool_docs(messages), docs)

    def test_latest_tool_docs_no_tool_calls_returns_none(self):
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
        self.assertIsNone(latest_tool_docs(messages))
