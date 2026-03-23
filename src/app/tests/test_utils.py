import unittest
import uuid
from unittest.mock import patch

from src.app.user.utils import utils


class TestResolveUserAndSession(unittest.IsolatedAsyncioTestCase):
    @patch("src.app.user.utils.utils.run_in_threadpool")
    async def test_existing_user_and_session(self, run_in_threadpool_mock):
        """Should return user_id and session_uuid for existing user and session"""
        user_id = uuid.uuid4()
        session_uuid = uuid.uuid4()
        host = "localhost"
        referer = "test"
        # First call: get_user_from_session_id returns user_id
        # Second call: get_or_create_session_sync returns session_uuid
        run_in_threadpool_mock.side_effect = [user_id, session_uuid]

        result_user_id, result_session_uuid = await utils.resolve_user_and_session(
            session_uuid, host, referer
        )
        self.assertEqual(result_user_id, user_id)
        self.assertEqual(result_session_uuid, session_uuid)
        self.assertEqual(run_in_threadpool_mock.call_count, 2)

    @patch("src.app.user.utils.utils.run_in_threadpool")
    async def test_new_user_and_session(self, run_in_threadpool_mock):
        """Should create new user and session if user_id not found"""
        session_uuid = uuid.uuid4()
        host = "localhost"
        referer = "test"
        new_user_id = uuid.uuid4()
        new_session_uuid = uuid.uuid4()
        # First call: get_user_from_session_id returns None
        # Second call: get_or_create_user_sync returns new_user_id
        # Third call: get_or_create_session_sync returns new_session_uuid
        run_in_threadpool_mock.side_effect = [None, new_user_id, new_session_uuid]

        result_user_id, result_session_uuid = await utils.resolve_user_and_session(
            session_uuid, host, referer
        )
        self.assertEqual(result_user_id, new_user_id)
        self.assertEqual(result_session_uuid, new_session_uuid)
        self.assertEqual(run_in_threadpool_mock.call_count, 3)

    @patch("src.app.user.utils.utils.run_in_threadpool")
    async def test_logger_called(self, run_in_threadpool_mock):
        """Should log info for both new and existing user cases"""
        user_id = uuid.uuid4()
        session_uuid = uuid.uuid4()
        host = "localhost"
        referer = "test"
        run_in_threadpool_mock.side_effect = [user_id, session_uuid]
        with patch.object(utils.logger, "info") as logger_info:
            await utils.resolve_user_and_session(session_uuid, host, referer)
            logger_info.assert_called()
