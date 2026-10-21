import unittest
import uuid
from unittest import mock

from src.app.api.api_v1.endpoints import chat_utils


class TestChatUtils(unittest.TestCase):
    def test_resolve_thread_id_with_value(self):
        test_uuid = uuid.uuid4()
        result = chat_utils._resolve_thread_id(test_uuid)
        self.assertEqual(result, test_uuid)

    def test_resolve_thread_id_without_value(self):
        result = chat_utils._resolve_thread_id(None)
        self.assertIsInstance(result, uuid.UUID)

    def test_update_agent_stream_state_processing(self):
        chunk = {"status": "processing", "docs": ["doc1"]}
        final_content, docs = chat_utils._update_agent_stream_state(chunk, "", None)
        self.assertEqual(docs, ["doc1"])
        self.assertEqual(final_content, "")

    def test_update_agent_stream_state_streaming(self):
        chunk = {"status": "streaming", "content": "new token"}
        final_content, docs = chat_utils._update_agent_stream_state(
            chunk, "old ", "docs"
        )
        self.assertEqual(final_content, "old new token")
        self.assertEqual(docs, "docs")

    def test_update_agent_stream_state_stop(self):
        chunk = {"status": "stop", "content": "final answer"}
        final_content, docs = chat_utils._update_agent_stream_state(
            chunk, "old", "docs"
        )
        self.assertEqual(final_content, "final answer")
        self.assertEqual(docs, "docs")

    def test_update_agent_stream_state_default(self):
        chunk = {"status": "other"}
        final_content, docs = chat_utils._update_agent_stream_state(
            chunk, "old", "docs"
        )
        self.assertEqual(final_content, "old")
        self.assertEqual(docs, "docs")

    def test_serialize_agent_stream_chunk(self):
        chunk = {
            "status": "processing",
            "step": "fetching_resources",
        }
        result = chat_utils._serialize_agent_stream_chunk(chunk)
        self.assertEqual(
            result,
            '{"content": null, "status": "processing", "step": "fetching_resources", "label": null, "docs": null}',
        )

    def test_serialize_agent_stream_chunk_with_docs(self):
        chunk = {
            "status": "processing",
            "step": "analyzing_resources",
            "docs": [{"id": "doc-1"}],
        }
        result = chat_utils._serialize_agent_stream_chunk(chunk)
        self.assertEqual(
            result,
            '{"content": null, "status": "processing", "step": "analyzing_resources", "label": null, "docs": [{"id": "doc-1"}]}',
        )

    def test_format_sse_event(self):
        result = chat_utils._format_sse_event('{"content": "abc"}')
        self.assertEqual(result, 'data: {"content": "abc"}\n\n')


class TestStreamAgentWithMemory(unittest.IsolatedAsyncioTestCase):
    @mock.patch("src.app.api.api_v1.endpoints.chat_utils.AsyncPostgresSaver")
    @mock.patch("psycopg.AsyncConnection.connect", new_callable=mock.AsyncMock)
    async def test_skips_state_fetch_when_tool_call_happened_this_turn(
        self, mock_connect, mock_saver
    ):
        async def fake_agent_stream():
            yield {"status": "processing", "docs": [{"id": "fresh-doc"}]}
            yield {"status": "stop", "content": "answer"}

        chatfactory = mock.Mock()
        chatfactory.agent_message = mock.AsyncMock(return_value=fake_agent_stream())
        chatfactory.agent_get_latest_docs = mock.AsyncMock()

        chunks = [
            chunk
            async for chunk in chat_utils._stream_agent_with_memory(
                db_uri="postgresql://test",
                async_dict_row_factory=mock.Mock(),
                chatfactory=chatfactory,
                body=mock.Mock(query="q", corpora=None, sdg_filter=None),
                sp=mock.Mock(),
                background_tasks=mock.Mock(),
                thread_id=uuid.uuid4(),
            )
        ]

        chatfactory.agent_get_latest_docs.assert_not_called()
        self.assertEqual(len(chunks), 2)

    @mock.patch("src.app.api.api_v1.endpoints.chat_utils.AsyncPostgresSaver")
    @mock.patch("psycopg.AsyncConnection.connect", new_callable=mock.AsyncMock)
    async def test_falls_back_to_latest_docs_when_no_tool_call_this_turn(
        self, mock_connect, mock_saver
    ):
        async def fake_agent_stream():
            yield {"status": "stop", "content": "answer reusing prior docs"}

        chatfactory = mock.Mock()
        chatfactory.agent_message = mock.AsyncMock(return_value=fake_agent_stream())
        chatfactory.agent_get_latest_docs = mock.AsyncMock(
            return_value=[{"id": "reused-doc"}]
        )

        chunks = [
            chunk
            async for chunk in chat_utils._stream_agent_with_memory(
                db_uri="postgresql://test",
                async_dict_row_factory=mock.Mock(),
                chatfactory=chatfactory,
                body=mock.Mock(query="q", corpora=None, sdg_filter=None),
                sp=mock.Mock(),
                background_tasks=mock.Mock(),
                thread_id=uuid.uuid4(),
            )
        ]

        chatfactory.agent_get_latest_docs.assert_called_once()
        self.assertEqual(
            chunks[-1], {"status": "docs_final", "docs": [{"id": "reused-doc"}]}
        )


class TestStreamAgentResponse(unittest.IsolatedAsyncioTestCase):
    async def test_docs_final_chunk_overrides_docs_and_is_not_forwarded(self):
        async def fake_stream(**kwargs):
            yield {"status": "streaming", "content": "partial answer"}
            yield {"status": "docs_final", "docs": [{"id": "fallback-doc"}]}
            yield {"status": "stop", "content": "partial answer"}

        data_collection = mock.Mock()
        data_collection.register_chat_data = mock.AsyncMock(
            return_value=(None, uuid.uuid4())
        )

        with mock.patch.object(
            chat_utils, "_stream_agent_with_memory", side_effect=fake_stream
        ):
            events = [
                event
                async for event in chat_utils._stream_agent_response(
                    db_uri="postgresql://test",
                    async_dict_row_factory=mock.Mock(),
                    body=mock.Mock(query="q"),
                    chatfactory=mock.Mock(),
                    sp=mock.Mock(),
                    background_tasks=mock.Mock(),
                    data_collection=data_collection,
                    session_id=None,
                    thread_id=uuid.uuid4(),
                )
            ]

        self.assertFalse(any('"status": "docs_final"' in e for e in events))
        self.assertTrue(any('"id": "fallback-doc"' in e for e in events))
        data_collection.register_chat_data.assert_called_once()
        _, kwargs = data_collection.register_chat_data.call_args
        self.assertEqual(kwargs["sources"], [{"id": "fallback-doc"}])


if __name__ == "__main__":
    unittest.main()
