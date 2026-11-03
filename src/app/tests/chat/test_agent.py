from unittest import IsolatedAsyncioTestCase, mock
from unittest.mock import AsyncMock

from fastapi import BackgroundTasks

from src.app.chat.services.agent import (
    NO_DOCUMENTS_FOUND,
    SEARCH_UNAVAILABLE,
    _get_resources_about_sustainability,
)


class TestAgent(IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch(
            "src.app.chat.services.agent.get_visible_corpora_sync",
            return_value=("a", "b"),
        )
        self.mock_visible = patcher.start()
        self.addCleanup(patcher.stop)

    def _config(self, corpora):
        mock_service = mock.Mock()
        mock_service.search_handler = AsyncMock(return_value=[])
        config = {
            "configurable": {
                "sdg_filter": None,
                "corpora": corpora,
                "sp": mock_service,
                "background_tasks": BackgroundTasks(),
            }
        }
        return config, mock_service

    async def _searched_corpora(self, corpora):
        config, mock_service = self._config(corpora)
        content, docs = await _get_resources_about_sustainability("q", config)
        self.assertEqual(content, NO_DOCUMENTS_FOUND)
        self.assertEqual(docs, [])
        return mock_service.search_handler.call_args.kwargs["qp"].corpora

    async def test_no_corpora_selected_searches_all_visible(self):
        self.assertEqual(await self._searched_corpora(None), ("a", "b"))

    async def test_empty_corpora_searches_all_visible(self):
        # the client sends [] when no source filter is selected
        self.assertEqual(await self._searched_corpora(()), ("a", "b"))

    async def test_hidden_corpora_are_dropped(self):
        self.assertEqual(await self._searched_corpora(("a", "hidden")), ("a",))

    async def test_only_hidden_corpora_searches_nothing(self):
        self.assertEqual(await self._searched_corpora(("hidden",)), ())

    async def test_visible_corpora_lookup_failure(self):
        self.mock_visible.side_effect = Exception("db down")
        config, mock_service = self._config(None)
        content, docs = await _get_resources_about_sustainability("q", config)
        self.assertEqual(content, SEARCH_UNAVAILABLE)
        self.assertEqual(docs, [])
        mock_service.search_handler.assert_not_called()

    async def test_get_resources_about_sustainability_found(self):
        # Mock SearchService and its search_handler
        with mock.patch(
            "src.app.chat.services.agent.SearchService"
        ) as MockSearchService:
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
        with mock.patch(
            "src.app.chat.services.agent.SearchService"
        ) as MockSearchService:
            mock_service = MockSearchService.return_value
            mock_service.search_handler = AsyncMock(return_value=[])
            config = {"configurable": {"sdg_filter": None, "corpora": None}}
            content, docs = await _get_resources_about_sustainability(
                "test query", config
            )
            self.assertEqual(content, NO_DOCUMENTS_FOUND)
            self.assertEqual(docs, [])

    async def test_get_resources_about_sustainability_search_error(self):
        mock_service = mock.Mock()
        mock_service.search_handler = AsyncMock(side_effect=Exception("qdrant down"))
        config = {
            "configurable": {
                "sdg_filter": None,
                "corpora": None,
                "sp": mock_service,
                "background_tasks": BackgroundTasks(),
            }
        }
        content, docs = await _get_resources_about_sustainability("test query", config)
        self.assertEqual(content, SEARCH_UNAVAILABLE)
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
        self.assertEqual(content, NO_DOCUMENTS_FOUND)
        self.assertEqual(docs, [])

    async def test_get_resources_about_sustainability_with_background_tasks(self):
        # Mock SearchService and its search_handler
        with mock.patch(
            "src.app.chat.services.agent.SearchService"
        ) as MockSearchService:
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
        with mock.patch(
            "src.app.chat.services.agent.SearchService"
        ) as MockSearchService:
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
                "src.app.chat.services.agent.stringify_docs_content"
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
