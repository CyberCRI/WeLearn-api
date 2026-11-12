import unittest
import uuid
from unittest import mock

from fastapi.testclient import TestClient
from langgraph.checkpoint.base import CheckpointTuple

from src.app.chat.services import thread_access
from src.app.core.config import settings
from src.app.shared.domain.exceptions import SessionNotFoundError, ThreadNotFoundError
from src.main import app

QNA_URL = f"{settings.API_V1_STR}/qna"
LATEST_URL = f"{QNA_URL}/chat/threads/latest"
HEADERS = {"X-API-Key": "test"}
THREAD_ACCESS = "src.app.chat.services.thread_access"

USER_A, SESSION_A = uuid.uuid4(), uuid.uuid4()
USER_B, SESSION_B = uuid.uuid4(), uuid.uuid4()
SESSION_USERS = {SESSION_A: USER_A, SESSION_B: USER_B}
HEADERS_A = {**HEADERS, "X-Session-Id": str(SESSION_A)}


def _user_of_session(session_id):
    if session_id not in SESSION_USERS:
        raise SessionNotFoundError()
    return SESSION_USERS[session_id]


def _checkpoint(metadata, thread_id: uuid.UUID | None = None) -> CheckpointTuple:
    return CheckpointTuple(
        config={"configurable": {"thread_id": str(thread_id or uuid.uuid4())}},
        checkpoint=mock.MagicMock(),
        metadata=metadata,
    )


def _alist(*checkpoints):
    """Replacement for AsyncPostgresSaver.alist (an async generator)."""

    async def alist(config, *, filter=None, before=None, limit=None):
        for checkpoint in checkpoints:
            yield checkpoint

    return mock.MagicMock(side_effect=alist)


def _fake_connect():
    """Replacement for psycopg.AsyncConnection.connect (awaited, then async-with)."""
    conn = mock.MagicMock()
    conn.execute = mock.AsyncMock()
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=conn)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.AsyncMock(return_value=cm), conn


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class ChatThreadsRouteTests(unittest.TestCase):
    def setUp(self):
        self.connect, self.conn = _fake_connect()
        self.saver = mock.MagicMock()
        self.saver.aget_tuple = mock.AsyncMock(return_value=None)
        self.saver.adelete_thread = mock.AsyncMock()
        self.saver.alist = _alist()
        for target, kwargs in (
            (f"{THREAD_ACCESS}.psycopg.AsyncConnection.connect", {"new": self.connect}),
            (f"{THREAD_ACCESS}.AsyncPostgresSaver", {"return_value": self.saver}),
            (
                f"{THREAD_ACCESS}.get_user_from_session_id",
                {"side_effect": _user_of_session},
            ),
        ):
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    # --- GET /chat/threads/latest -------------------------------------------

    def test_latest_returns_the_callers_latest_thread(self):
        thread_id = uuid.uuid4()
        self.saver.alist = _alist(
            _checkpoint({thread_access.OWNER_METADATA_KEY: str(USER_A)}, thread_id)
        )

        response = self.client.get(LATEST_URL, headers=HEADERS_A)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"thread_id": str(thread_id)})
        self.conn.execute.assert_awaited_once_with("SET SEARCH_PATH to agent_related")

    def test_latest_filters_on_the_callers_id_with_limit_one(self):
        self.client.get(LATEST_URL, headers=HEADERS_A)

        self.saver.alist.assert_called_once_with(
            None, filter={thread_access.OWNER_METADATA_KEY: str(USER_A)}, limit=1
        )

    def test_latest_without_thread_returns_null(self):
        response = self.client.get(LATEST_URL, headers=HEADERS_A)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"thread_id": None})

    def test_latest_without_session_returns_null_without_db(self):
        response = self.client.get(LATEST_URL, headers=HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"thread_id": None})
        self.connect.assert_not_awaited()
        self.saver.alist.assert_not_called()

    def test_latest_with_unknown_session_returns_null_without_db(self):
        response = self.client.get(
            LATEST_URL, headers={**HEADERS, "X-Session-Id": str(uuid.uuid4())}
        )

        self.assertEqual(response.json(), {"thread_id": None})
        self.connect.assert_not_awaited()

    # --- DELETE /chat/threads/{thread_id} -----------------------------------

    def _delete(self, thread_id: uuid.UUID, headers=HEADERS_A):
        return self.client.delete(
            f"{QNA_URL}/chat/threads/{thread_id}", headers=headers
        )

    def test_delete_own_thread(self):
        thread_id = uuid.uuid4()
        self.saver.aget_tuple.return_value = _checkpoint(
            {thread_access.OWNER_METADATA_KEY: str(USER_A)}, thread_id
        )

        response = self._delete(thread_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"thread_id": str(thread_id)})
        self.saver.aget_tuple.assert_awaited_once_with(
            {"configurable": {"thread_id": str(thread_id)}}
        )
        self.saver.adelete_thread.assert_awaited_once_with(str(thread_id))

    def _assert_not_found(self, response):
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "THREAD_NOT_FOUND")
        self.saver.adelete_thread.assert_not_awaited()

    def test_delete_thread_of_another_user_returns_404(self):
        self.saver.aget_tuple.return_value = _checkpoint(
            {thread_access.OWNER_METADATA_KEY: str(USER_B)}
        )

        self._assert_not_found(self._delete(uuid.uuid4()))

    def test_delete_thread_without_owner_returns_404(self):
        self.saver.aget_tuple.return_value = _checkpoint({"source": "loop"})

        self._assert_not_found(self._delete(uuid.uuid4()))

    def test_delete_missing_thread_returns_404(self):
        self._assert_not_found(self._delete(uuid.uuid4()))

    def test_delete_without_session_returns_404(self):
        self.saver.aget_tuple.return_value = _checkpoint(
            {thread_access.OWNER_METADATA_KEY: str(USER_A)}
        )

        self._assert_not_found(self._delete(uuid.uuid4(), headers=HEADERS))
        self.saver.aget_tuple.assert_not_awaited()

    def test_delete_latest_is_not_a_thread_id(self):
        response = self.client.delete(LATEST_URL, headers=HEADERS_A)

        self.assertEqual(response.status_code, 422)
        self.saver.adelete_thread.assert_not_awaited()


class EnsureThreadOwnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory = mock.MagicMock()
        self.memory.aget_tuple = mock.AsyncMock(return_value=None)
        self.thread_id = uuid.uuid4()

    async def test_owner_passes(self):
        self.memory.aget_tuple.return_value = _checkpoint(
            {thread_access.OWNER_METADATA_KEY: str(USER_A)}
        )

        await thread_access.ensure_thread_owner(self.memory, self.thread_id, USER_A)

    async def test_refusals(self):
        cases = {
            "no checkpoint": (None, USER_A),
            "no metadata": (_checkpoint(None), USER_A),
            "no owner": (_checkpoint({"source": "loop"}), USER_A),
            "other owner": (
                _checkpoint({thread_access.OWNER_METADATA_KEY: str(USER_B)}),
                USER_A,
            ),
            "no caller": (
                _checkpoint({thread_access.OWNER_METADATA_KEY: str(USER_A)}),
                None,
            ),
        }
        for name, (checkpoint, user_id) in cases.items():
            with self.subTest(name):
                self.memory.aget_tuple.return_value = checkpoint
                with self.assertRaises(ThreadNotFoundError):
                    await thread_access.ensure_thread_owner(
                        self.memory, self.thread_id, user_id
                    )


class FindLatestThreadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connect, self.conn = _fake_connect()
        self.saver = mock.MagicMock()
        for target, kwargs in (
            (f"{THREAD_ACCESS}.psycopg.AsyncConnection.connect", {"new": self.connect}),
            (f"{THREAD_ACCESS}.AsyncPostgresSaver", {"return_value": self.saver}),
        ):
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def _find(self):
        return await thread_access.find_latest_thread(
            db_uri="dbname=test",
            async_dict_row_factory=mock.MagicMock(),
            user_id=USER_A,
        )

    async def test_returns_the_thread_of_the_first_checkpoint(self):
        thread_id = uuid.uuid4()
        self.saver.alist = _alist(_checkpoint({}, thread_id))

        self.assertEqual(await self._find(), thread_id)

    async def test_returns_none_without_checkpoint(self):
        self.saver.alist = _alist()

        self.assertIsNone(await self._find())


if __name__ == "__main__":
    unittest.main()
