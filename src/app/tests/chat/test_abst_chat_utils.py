import unittest
from unittest import mock

from src.app.chat.services import abst_chat


class TestAbstChatUtils(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = abst_chat.AbstractChat()

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
        self.assertEqual(
            result,
            {"status": "streaming", "step": "generating_answer", "content": "hello"},
        )

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


if __name__ == "__main__":
    unittest.main()
