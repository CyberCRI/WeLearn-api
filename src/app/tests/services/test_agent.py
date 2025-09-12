from unittest import IsolatedAsyncioTestCase, mock
from unittest.mock import AsyncMock

from src.app.services.agent import (
    _get_resources_about_sustainability,
    trim_conversation_history,
)


class TestAgent(IsolatedAsyncioTestCase):
    async def test_get_resources_about_sustainability_found(self):
        # Mock SearchService and its search_handler
        with mock.patch("src.app.services.agent.SearchService") as MockSearchService:
            mock_service = MockSearchService.return_value
            mock_payload = {
                "document_title": "Test Title",
                "slice_content": "Test Content",
                "document_url": "http://test.url",
            }
            mock_doc = mock.Mock()
            mock_doc.payload = mock_payload
            mock_service.search_handler = AsyncMock(return_value=[mock_doc] * 8)

            # config with dummy values
            config = {"configurable": {"sdg_filter": None, "corpora": None}}
            content, docs = await _get_resources_about_sustainability(
                "test query", config
            )
            self.assertIsInstance(content, str)
            self.assertEqual(len(docs), 8)
            self.assertIn("Doc 1", content)

    async def test_get_resources_about_sustainability_not_found(self):
        with mock.patch("src.app.services.agent.SearchService") as MockSearchService:
            mock_service = MockSearchService.return_value
            mock_service.search_handler = AsyncMock(return_value=[])
            config = {"configurable": {"sdg_filter": None, "corpora": None}}
            content, docs = await _get_resources_about_sustainability(
                "test query", config
            )
            self.assertEqual(content, "No relevant documents found.")
            self.assertEqual(docs, [])

    def test_trim_conversation_history(self):
        state = {"messages": ["msg1", "msg2", "msg3", "msg4", "msg5", "msg6"]}
        result = trim_conversation_history(state)
        self.assertIn("llm_input_messages", result)
        self.assertIsInstance(result["llm_input_messages"], list)
