import unittest
import uuid

from src.app.api.api_v1.endpoints import chat


class TestChatUtils(unittest.TestCase):
    def test_resolve_thread_id_with_value(self):
        test_uuid = uuid.uuid4()
        result = chat._resolve_thread_id(test_uuid)
        self.assertEqual(result, test_uuid)

    def test_resolve_thread_id_without_value(self):
        result = chat._resolve_thread_id(None)
        self.assertIsInstance(result, uuid.UUID)

    def test_update_agent_stream_state_processing(self):
        chunk = {"status": "processing", "docs": ["doc1"]}
        final_content, docs = chat._update_agent_stream_state(chunk, "", None)
        self.assertEqual(docs, ["doc1"])
        self.assertEqual(final_content, "")

    def test_update_agent_stream_state_stop(self):
        chunk = {"status": "stop", "content": "final answer"}
        final_content, docs = chat._update_agent_stream_state(chunk, "old", "docs")
        self.assertEqual(final_content, "final answer")
        self.assertEqual(docs, "docs")

    def test_update_agent_stream_state_default(self):
        chunk = {"status": "other"}
        final_content, docs = chat._update_agent_stream_state(chunk, "old", "docs")
        self.assertEqual(final_content, "old")
        self.assertEqual(docs, "docs")

    def test_serialize_agent_stream_chunk(self):
        chunk = {"content": "abc", "status": "processing"}
        result = chat._serialize_agent_stream_chunk(chunk)
        self.assertEqual(result, '{"content": "abc", "status": "processing"}')


if __name__ == "__main__":
    unittest.main()
