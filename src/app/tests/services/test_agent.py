from unittest import IsolatedAsyncioTestCase, mock
from unittest.mock import AsyncMock

from fastapi import BackgroundTasks

from src.app.services.agent import _get_resources_about_sustainability


class TestAgent(IsolatedAsyncioTestCase):
    async def test_get_resources_about_sustainability_found(self):
        # Mock SearchService and its search_handler
        with mock.patch("src.app.services.agent.SearchService") as MockSearchService:
            mock_service = MockSearchService.return_value
            mock_doc = mock.Mock()
            mock_doc.payload = {
                "document_title": "Test Title",
                "slice_content": "Test Content",
                "document_url": "http://test.url",
            }
            mock_service.search_handler = AsyncMock(return_value=[mock_doc] * 8)
            config = {
                "configurable": {
                    "sdg_filter": None,
                    "corpora": None,
                    "sp": mock_service,
                    "background_tasks": BackgroundTasks(),
                }
            }

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

    async def test_get_resources_about_sustainability_no_search_service(self):
        # config without 'sp' (SearchService)
        config = {
            "configurable": {
                "sdg_filter": None,
                "corpora": None,
                "sp": None,
                "background_tasks": None,
            }
        }
        content, docs = await _get_resources_about_sustainability("test query", config)
        self.assertEqual(content, "No relevant documents found.")
        self.assertEqual(docs, [])

    async def test_get_resources_about_sustainability_with_background_tasks(self):
        # Mock SearchService and its search_handler
        with mock.patch("src.app.services.agent.SearchService") as MockSearchService:
            mock_service = MockSearchService.return_value
            mock_doc = mock.Mock()
            mock_doc.payload = {
                "document_title": "Test Title",
                "slice_content": "Test Content",
                "document_url": "http://test.url",
            }
            mock_service.search_handler = AsyncMock(return_value=[mock_doc])
            config = {
                "configurable": {
                    "sdg_filter": None,
                    "corpora": None,
                    "sp": mock_service,
                    "background_tasks": BackgroundTasks(),
                }
            }
            content, docs = await _get_resources_about_sustainability(
                "test query", config
            )
            self.assertIsInstance(content, str)
            self.assertEqual(len(docs), 1)

    async def test_get_resources_about_sustainability_limits_to_seven_docs(self):
        # Mock SearchService and its search_handler
        with mock.patch("src.app.services.agent.SearchService") as MockSearchService:
            mock_service = MockSearchService.return_value
            mock_doc = mock.Mock()
            mock_doc.payload = {
                "document_title": "Test Title",
                "slice_content": "Test Content",
                "document_url": "http://test.url",
            }
            mock_service.search_handler = AsyncMock(return_value=[mock_doc] * 10)
            config = {
                "configurable": {
                    "sdg_filter": None,
                    "corpora": None,
                    "sp": mock_service,
                    "background_tasks": BackgroundTasks(),
                }
            }
            with mock.patch(
                "src.app.services.agent.stringify_docs_content"
            ) as mock_stringify:
                mock_stringify.return_value = "stringified content"
                content, docs = await _get_resources_about_sustainability(
                    "test query", config
                )
                mock_stringify.assert_called_once()
                called_docs = mock_stringify.call_args[0][0]
                self.assertEqual(len(called_docs), 7)
                self.assertEqual(content, "stringified content")
                self.assertEqual(len(docs), 10)
