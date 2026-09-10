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


class TestJudgeAgentAnswer(unittest.IsolatedAsyncioTestCase):
    async def test_returns_verdict_and_logs_on_violation(self):
        thread_id = uuid.uuid4()
        verdict = mock.Mock(compliant=False, unsupported_citations=["[Doc 9]"])
        chatfactory = mock.Mock()
        chatfactory.judge_source_grounding = mock.AsyncMock(return_value=verdict)

        with mock.patch.object(chat_utils.logger, "warning") as mock_warning:
            result = await chat_utils.judge_agent_answer(
                chatfactory=chatfactory,
                content="some answer",
                docs=None,
                thread_id=thread_id,
                log_prefix="test_prefix",
            )

        self.assertIs(result, verdict)
        mock_warning.assert_called_once()

    async def test_returns_none_on_judge_failure(self):
        chatfactory = mock.Mock()
        chatfactory.judge_source_grounding = mock.AsyncMock(side_effect=Exception())

        result = await chat_utils.judge_agent_answer(
            chatfactory=chatfactory,
            content="some answer",
            docs=None,
            thread_id=uuid.uuid4(),
            log_prefix="test_prefix",
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
