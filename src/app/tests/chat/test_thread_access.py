import unittest
import uuid
from unittest import mock

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from starlette.requests import Request

from src.app.chat.services import thread_access
from src.app.chat.services.abst_chat import AbstractChat
from src.app.shared.domain.exceptions import SessionNotFoundError, ThreadNotFoundError

MODULE = "src.app.chat.services.thread_access"


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


class ResolveCallerUserIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_session_resolves_to_none_without_db_lookup(self):
        with mock.patch(f"{MODULE}.get_user_from_session_id") as lookup:
            self.assertIsNone(await thread_access.resolve_caller_user_id(_request({})))
        lookup.assert_not_called()

    async def test_valid_session_resolves_to_its_user(self):
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        with mock.patch(
            f"{MODULE}.get_user_from_session_id", return_value=user_id
        ) as lookup:
            result = await thread_access.resolve_caller_user_id(
                _request({"X-Session-Id": str(session_id)})
            )
        self.assertEqual(result, user_id)
        lookup.assert_called_once_with(session_id)

    async def test_unknown_or_expired_session_resolves_to_none(self):
        with mock.patch(
            f"{MODULE}.get_user_from_session_id", side_effect=SessionNotFoundError()
        ):
            result = await thread_access.resolve_caller_user_id(
                _request({"X-Session-Id": str(uuid.uuid4())})
            )
        self.assertIsNone(result)


class EnsureThreadAccessTests(unittest.IsolatedAsyncioTestCase):
    """Runs a real LangGraph graph through `AbstractChat.agent_message`, so the
    owner goes through the run config metadata into the checkpoint the way it
    does in production (InMemorySaver and AsyncPostgresSaver build checkpoint
    metadata with the same `get_checkpoint_metadata`)."""

    def setUp(self):
        self.memory = InMemorySaver()
        graph = StateGraph(MessagesState)
        graph.add_node("model", lambda state: {"messages": [AIMessage(content="ok")]})
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        self.chat = AbstractChat()
        self.chat.agent_executor = graph.compile(checkpointer=self.memory)
        self.user_a, self.user_b = uuid.uuid4(), uuid.uuid4()

    async def _send(self, thread_id, owner_user_id):
        await self.chat.agent_message(
            query="q",
            memory=self.memory,
            thread_id=thread_id,
            owner_user_id=owner_user_id,
        )

    async def test_unknown_thread_is_allowed(self):
        await thread_access.ensure_thread_access(self.memory, uuid.uuid4(), None)

    async def test_owner_is_written_and_only_the_owner_gets_access(self):
        thread_id = uuid.uuid4()
        await self._send(thread_id, self.user_a)

        tuple_ = await self.memory.aget_tuple(
            {"configurable": {"thread_id": str(thread_id)}}
        )
        self.assertEqual(
            tuple_.metadata[thread_access.OWNER_METADATA_KEY], str(self.user_a)
        )

        await thread_access.ensure_thread_access(self.memory, thread_id, self.user_a)
        with self.assertRaises(ThreadNotFoundError):
            await thread_access.ensure_thread_access(
                self.memory, thread_id, self.user_b
            )
        with self.assertRaises(ThreadNotFoundError):
            await thread_access.ensure_thread_access(self.memory, thread_id, None)

    async def test_owner_is_kept_across_turns(self):
        thread_id = uuid.uuid4()
        await self._send(thread_id, self.user_a)
        await self._send(thread_id, self.user_a)

        with self.assertRaises(ThreadNotFoundError):
            await thread_access.ensure_thread_access(
                self.memory, thread_id, self.user_b
            )

    async def test_legacy_thread_is_allowed_then_claimed_by_next_writer(self):
        thread_id = uuid.uuid4()
        await self._send(thread_id, None)  # written before owners were recorded

        await thread_access.ensure_thread_access(self.memory, thread_id, self.user_b)
        await thread_access.ensure_thread_access(self.memory, thread_id, None)

        await self._send(thread_id, self.user_b)

        await thread_access.ensure_thread_access(self.memory, thread_id, self.user_b)
        with self.assertRaises(ThreadNotFoundError):
            await thread_access.ensure_thread_access(
                self.memory, thread_id, self.user_a
            )

    async def test_owner_cannot_be_overridden_by_trace_context(self):
        thread_id = uuid.uuid4()
        await self.chat.agent_message(
            query="q",
            memory=self.memory,
            thread_id=thread_id,
            owner_user_id=self.user_a,
            trace_context={thread_access.OWNER_METADATA_KEY: str(self.user_b)},
        )

        await thread_access.ensure_thread_access(self.memory, thread_id, self.user_a)


class CheckThreadAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_opens_a_connection_in_the_agent_schema(self):
        conn = mock.MagicMock()
        conn.execute = mock.AsyncMock()
        cm = mock.MagicMock()
        cm.__aenter__ = mock.AsyncMock(return_value=conn)
        cm.__aexit__ = mock.AsyncMock(return_value=False)
        thread_id, user_id = uuid.uuid4(), uuid.uuid4()

        with (
            mock.patch(
                f"{MODULE}.psycopg.AsyncConnection.connect",
                mock.AsyncMock(return_value=cm),
            ) as connect,
            mock.patch(f"{MODULE}.AsyncPostgresSaver") as saver_cls,
            mock.patch(f"{MODULE}.ensure_thread_access") as ensure,
        ):
            await thread_access.check_thread_access(
                db_uri="postgresql://db",
                async_dict_row_factory=mock.MagicMock(),
                thread_id=thread_id,
                user_id=user_id,
            )

        self.assertEqual(connect.await_args.args[0], "postgresql://db")
        conn.execute.assert_awaited_once_with("SET SEARCH_PATH to agent_related")
        saver_cls.assert_called_once_with(conn)
        ensure.assert_awaited_once_with(saver_cls.return_value, thread_id, user_id)
