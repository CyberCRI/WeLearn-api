import types
import unittest
from unittest import mock

from src.app.shared.infra import abst_chat


class TestAbstChatUtils(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = abst_chat.AbstractChat(client=mock.AsyncMock())

    async def test_json_formatter_agent(self):
        self.chat.chat_client.completion = mock.AsyncMock(
            return_value='{"key": "value"}'
        )
        with mock.patch(
            "src.app.services.helpers.extract_json_from_response",
            return_value={"key": "value"},
        ):
            result = await self.chat.json_formatter_agent("bad", "schema")
            self.assertEqual(result, {"key": "value"})

    async def test_get_stream_chunks_async_and_sync(self):
        # Create a mock chunk object with .choices attribute
        class MockDelta:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, delta, finish_reason=None):
                self.delta = delta
                self.finish_reason = finish_reason

        class MockChunk:
            def __init__(self, content):
                self.choices = [MockChoice(MockDelta(content))]

        # Async generator
        async def async_stream():
            yield MockChunk("abc")

        # Async path
        chunks = []
        async for part in self.chat.get_stream_chunks(async_stream()):
            chunks.append(part)
        self.assertIn("abc", chunks)

        # Sync fallback path: force async for to raise, fallback to sync
        def sync_stream():
            yield MockChunk("abc")

        # Patch async for to raise to force sync fallback
        async def broken_async_for(*args, **kwargs):
            raise Exception()

        with mock.patch.object(
            self.chat, "_extract_stream_chunk", wraps=self.chat._extract_stream_chunk
        ):
            with self.assertRaises(Exception):
                await self.chat.get_stream_chunks(broken_async_for())

    def test_extract_stream_chunk(self):
        # With choices and delta content
        chunk = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="abc"), finish_reason=None
                )
            ]
        )
        result = list(self.chat._extract_stream_chunk(chunk))
        self.assertIn("abc", result)
        # With finish_reason
        chunk = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=None), finish_reason="stop"
                )
            ]
        )
        # Should not yield, just log
        result = list(self.chat._extract_stream_chunk(chunk))
        self.assertEqual(result, [])

    def test_extract_agent_chunk(self):
        # Not a dict
        self.assertIsNone(next(self.chat._extract_agent_chunk("notadict"), None))
        # With tools
        chunk = {
            "tools": {"messages": [mock.Mock(artifact=[1, 2, 3])]},
            "content": "foo",
        }
        result = next(self.chat._extract_agent_chunk(chunk))
        self.assertEqual(result["status"], "processing")
        self.assertEqual(result["step"], "analyzing_resources")
        self.assertNotIn("content", result)
        # With model and finish_reason tool_calls
        chunk = {
            "model": {
                "messages": [
                    mock.Mock(response_metadata={"finish_reason": "tool_calls"})
                ]
            }
        }
        result = next(self.chat._extract_agent_chunk(chunk))
        self.assertEqual(result["status"], "processing")
        self.assertEqual(result["step"], "fetching_resources")
        self.assertNotIn("content", result)
        # With model and finish_reason stop (content kept as streaming)
        chunk = {
            "model": {
                "messages": [
                    mock.Mock(
                        response_metadata={"finish_reason": "stop"}, content="done"
                    )
                ]
            }
        }
        result = next(self.chat._extract_agent_chunk(chunk))
        self.assertEqual(result["status"], "streaming")

    def test_extract_agent_chunk_messages_tuple(self):
        msg = mock.Mock(content="hello", response_metadata={})
        chunk = (msg, {"langgraph_node": "model"})

        result = next(self.chat._extract_agent_chunk(chunk))
        self.assertEqual(result, {"status": "streaming", "step": "generating_answer", "content": "hello"})

    def test_extract_agent_chunk_messages_tuple_tool_call(self):
        msg = mock.Mock(content="", response_metadata={"finish_reason": "tool_calls"})
        chunk = (msg, {"langgraph_node": "model"})

        result = next(self.chat._extract_agent_chunk(chunk))
        self.assertEqual(result["status"], "processing")
        self.assertEqual(result["step"], "fetching_resources")
        self.assertNotIn("content", result)

    def test_extract_agent_chunk_messages_tuple_tools_node(self):
        msg = mock.Mock(artifact=[{"id": "doc-1"}], response_metadata={})
        chunk = (msg, {"langgraph_node": "tools"})

        result = next(self.chat._extract_agent_chunk(chunk))
        self.assertEqual(result["status"], "processing")
        self.assertEqual(result["step"], "analyzing_resources")
        self.assertEqual(result["docs"], [{"id": "doc-1"}])
        self.assertNotIn("content", result)

    async def test_run_llm_with_json_parsing_success(self):
        self.chat.chat_client.completion = mock.AsyncMock(return_value='{"foo": "bar"}')
        with mock.patch(
            "src.app.services.helpers.extract_json_from_response",
            return_value={"foo": "bar"},
        ):

            class Dummy:
                def __init__(self, foo):
                    self.foo = foo

            result = await self.chat.run_llm_with_json_parsing([], Dummy)
            self.assertIsInstance(result, Dummy)
            self.assertEqual(result.foo, "bar")

    async def test_run_llm_with_json_parsing_fallback(self):
        self.chat.chat_client.completion = mock.AsyncMock(return_value="notjson")
        with mock.patch(
            "src.app.services.helpers.extract_json_from_response",
            side_effect=Exception(),
        ):
            self.chat.json_formatter_agent = mock.AsyncMock(return_value={"foo": "bar"})

            class Dummy:
                def __init__(self, foo):
                    self.foo = foo

            result = await self.chat.run_llm_with_json_parsing(
                [], Dummy, fallback_formatter="schema"
            )
            self.assertEqual(result, {"foo": "bar"})

    async def test_run_llm_with_json_parsing_error(self):
        self.chat.chat_client.completion = mock.AsyncMock(return_value="notjson")
        with mock.patch(
            "src.app.services.helpers.extract_json_from_response",
            side_effect=Exception(),
        ):
            self.chat.json_formatter_agent = mock.AsyncMock(side_effect=Exception())

            class Dummy:
                def __init__(self, foo):
                    self.foo = foo

            with self.assertRaises(Exception):
                await self.chat.run_llm_with_json_parsing(
                    [], Dummy, fallback_formatter="schema"
                )


if __name__ == "__main__":
    unittest.main()
