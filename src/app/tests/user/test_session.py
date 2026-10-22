import unittest
import uuid
from unittest.mock import patch

from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.user.services import session


class TestResolveUserAndSession(unittest.IsolatedAsyncioTestCase):
    @patch("src.app.user.services.session.run_in_threadpool")
    async def test_existing_user_and_session(self, run_in_threadpool_mock):
        """Should return user_id and session_uuid for existing user and session"""
        user_id = uuid.uuid4()
        session_uuid = uuid.uuid4()
        host = "localhost"
        referer = "test"
        # First call: get_user_from_session_id returns user_id
        # Second call: get_or_create_session_sync returns session_uuid
        run_in_threadpool_mock.side_effect = [user_id, session_uuid]

        result_user_id, result_session_uuid = await session.resolve_user_and_session(
            session_uuid, host, referer
        )
        self.assertEqual(result_user_id, user_id)
        self.assertEqual(result_session_uuid, session_uuid)
        self.assertEqual(run_in_threadpool_mock.call_count, 2)

    @patch("src.app.user.services.session.get_user_from_session_id")
    async def test_session_not_existing(self, get_user_from_session_id_mock):
        """Should return user_id and session_uuid for existing user and session"""
        session_uuid = uuid.uuid4()
        host = "localhost"
        referer = "test"
        # First call: get_user_from_session_id returns user_id
        # Second call: get_or_create_session_sync returns session_uuid
        get_user_from_session_id_mock.side_effect = SessionNotFoundError(
            "Session not found"
        )

        # expect resolve_user_and_session to raise SessionNotFoundError
        with self.assertRaises(SessionNotFoundError):
            await session.resolve_user_and_session(session_uuid, host, referer)

    @patch("src.app.user.services.session.run_in_threadpool")
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

        result_user_id, result_session_uuid = await session.resolve_user_and_session(
            session_uuid, host, referer
        )
        self.assertEqual(result_user_id, new_user_id)
        self.assertEqual(result_session_uuid, new_session_uuid)
        self.assertEqual(run_in_threadpool_mock.call_count, 3)

    @patch("src.app.user.services.session.run_in_threadpool")
    async def test_logger_called(self, run_in_threadpool_mock):
        """Should log info for both new and existing user cases"""
        user_id = uuid.uuid4()
        session_uuid = uuid.uuid4()
        host = "localhost"
        referer = "test"
        run_in_threadpool_mock.side_effect = [user_id, session_uuid]
        with patch.object(session.logger, "info") as logger_info:
            await session.resolve_user_and_session(session_uuid, host, referer)
            logger_info.assert_called()


class TestResolveKeycloakUserAndSession(unittest.IsolatedAsyncioTestCase):
    @patch("src.app.user.services.session.get_or_create_session_sync")
    @patch("src.app.user.services.session.get_or_create_keycloak_user")
    @patch("src.app.user.services.session.get_user_from_session_id")
    async def test_links_anonymous_user_from_session(
        self, get_user_mock, get_keycloak_user_mock, get_session_mock
    ):
        """Should pass the session's anonymous user so it gets linked"""
        anonymous_user_id = uuid.uuid4()
        session_uuid = uuid.uuid4()
        get_user_mock.return_value = anonymous_user_id
        get_keycloak_user_mock.return_value = (anonymous_user_id, None)
        get_session_mock.return_value = session_uuid

        user_id, result_session = await session.resolve_keycloak_user_and_session(
            "token", session_uuid, "localhost", "test"
        )

        self.assertEqual(user_id, anonymous_user_id)
        self.assertEqual(result_session, session_uuid)
        get_keycloak_user_mock.assert_awaited_once_with(
            "token", anonymous_user_id, "test"
        )
        get_session_mock.assert_called_once_with(
            anonymous_user_id, session_uuid, "localhost", "test"
        )

    @patch("src.app.user.services.session.get_or_create_session_sync")
    @patch("src.app.user.services.session.get_or_create_keycloak_user")
    @patch("src.app.user.services.session.get_user_from_session_id")
    async def test_unknown_session_is_ignored(
        self, get_user_mock, get_keycloak_user_mock, get_session_mock
    ):
        """Should not fail on an unknown session cookie and create a new session"""
        keycloak_user_id = uuid.uuid4()
        new_session = uuid.uuid4()
        get_user_mock.side_effect = SessionNotFoundError("Session not found")
        get_keycloak_user_mock.return_value = (keycloak_user_id, None)
        get_session_mock.return_value = new_session

        user_id, result_session = await session.resolve_keycloak_user_and_session(
            "token", uuid.uuid4(), "localhost", None
        )

        self.assertEqual(user_id, keycloak_user_id)
        self.assertEqual(result_session, new_session)
        get_keycloak_user_mock.assert_awaited_once_with("token", None, None)
        get_session_mock.assert_called_once_with(
            keycloak_user_id, None, "localhost", None
        )

    @patch("src.app.user.services.session.get_or_create_session_sync")
    @patch("src.app.user.services.session.get_or_create_keycloak_user")
    @patch("src.app.user.services.session.get_user_from_session_id")
    async def test_no_session(
        self, get_user_mock, get_keycloak_user_mock, get_session_mock
    ):
        """Without session cookie, no anonymous user is looked up"""
        get_keycloak_user_mock.return_value = (uuid.uuid4(), None)

        await session.resolve_keycloak_user_and_session(
            "token", None, "localhost", None
        )

        get_user_mock.assert_not_called()
        get_keycloak_user_mock.assert_awaited_once_with("token", None, None)
