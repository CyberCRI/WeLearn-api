import unittest
import uuid
from unittest import mock

from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import CheckpointTuple
from starlette.requests import Request

from src.app.chat.api import router as chat_router
from src.app.chat.domain.models import AgentContext
from src.app.chat.services.abst_chat import get_chat_service
from src.app.core.config import settings
from src.app.data_collection.services.data_collection import get_data_collection_service
from src.app.search.services.search import get_search_service
from src.app.shared.domain.exceptions import (
    EmptyQueryError,
    LanguageNotSupportedError,
    SessionNotFoundError,
)
from src.main import app

QNA_URL = f"{settings.API_V1_STR}/qna"
HEADERS = {"X-API-Key": "test"}
ROUTER = "src.app.chat.api.router"
THREAD_ACCESS = "src.app.chat.services.thread_access"

USER_A, SESSION_A = uuid.uuid4(), uuid.uuid4()
USER_B, SESSION_B = uuid.uuid4(), uuid.uuid4()
SESSION_USERS = {SESSION_A: USER_A, SESSION_B: USER_B}
HEADERS_A = {**HEADERS, "X-Session-Id": str(SESSION_A)}
HEADERS_B = {**HEADERS, "X-Session-Id": str(SESSION_B)}


def _user_of_session(session_id):
    if session_id not in SESSION_USERS:
        raise SessionNotFoundError()
    return SESSION_USERS[session_id]


def _checkpoint(metadata) -> CheckpointTuple:
    return CheckpointTuple(
        config={"configurable": {"thread_id": "t"}},
        checkpoint=mock.MagicMock(),
        metadata=metadata,
    )


def _fake_connect():
    """Replacement for psycopg.AsyncConnection.connect (awaited, then async-with)."""
    conn = mock.MagicMock()
    conn.execute = mock.AsyncMock()
    conn.commit = mock.AsyncMock()
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=conn)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.AsyncMock(return_value=cm), conn


def _empty_request() -> Request:
    return Request({"type": "http", "headers": []})


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class ChatRouterTests(unittest.TestCase):
    def setUp(self):
        self.chat = mock.MagicMock()
        self.chat.agent_message = mock.AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="question"),
                    AIMessage(content="final answer"),
                ]
            }
        )
        self.chat.agent_get_history = mock.AsyncMock()
        self.sp = mock.MagicMock()
        self.data_collection = mock.MagicMock()
        self.message_id = uuid.uuid4()
        self.data_collection.register_chat_data = mock.AsyncMock(
            return_value=(uuid.uuid4(), self.message_id)
        )
        app.dependency_overrides[get_chat_service] = lambda: self.chat
        app.dependency_overrides[get_search_service] = lambda: self.sp
        app.dependency_overrides[get_data_collection_service] = (
            lambda: self.data_collection
        )

        self.connect, self.conn = _fake_connect()
        patches = [
            mock.patch(f"{ROUTER}.psycopg.AsyncConnection.connect", self.connect),
            mock.patch(f"{ROUTER}.AsyncPostgresSaver"),
        ]
        self.saver_cls = patches[1].start()
        patches[0].start()
        for p in patches:
            self.addCleanup(p.stop)
        # no checkpoint by default: every thread is a new one
        self.saver = self.saver_cls.return_value
        self.saver.aget_tuple = mock.AsyncMock(return_value=None)
        for target, kwargs in (
            (f"{THREAD_ACCESS}.AsyncPostgresSaver", {"return_value": self.saver}),
            (
                "src.app.chat.services.chat_utils.AsyncPostgresSaver",
                {"return_value": self.saver},
            ),
            (
                f"{THREAD_ACCESS}.get_user_from_session_id",
                {"side_effect": _user_of_session},
            ),
        ):
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def tearDown(self):
        for dep in (get_chat_service, get_search_service, get_data_collection_service):
            app.dependency_overrides.pop(dep, None)

    # --- get_agent_params -------------------------------------------------

    def test_agent_empty_query_returns_400(self):
        response = self.client.post(
            f"{QNA_URL}/chat/agent", json={"query": ""}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "EMPTY_QUERY")
        self.chat.agent_message.assert_not_awaited()

    def test_agent_stream_missing_query_returns_400(self):
        response = self.client.post(
            f"{QNA_URL}/chat/agent_stream", json={}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "EMPTY_QUERY")

    # --- /chat/history ----------------------------------------------------

    def test_history_reads_from_agent_memory(self):
        thread_id = uuid.uuid4()
        history = [{"role": "user", "content": "hi", "docs": None}]
        self.chat.agent_get_history.return_value = history

        response = self.client.get(
            f"{QNA_URL}/chat/history",
            params={"thread_id": str(thread_id)},
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), history)
        self.assertEqual(self.connect.await_args.args[0], chat_router.DB_URI)
        self.assertTrue(self.connect.await_args.kwargs["autocommit"])
        self.conn.execute.assert_awaited_once_with("SET SEARCH_PATH to agent_related")
        self.saver_cls.assert_called_once_with(self.conn)
        self.chat.agent_get_history.assert_awaited_once_with(
            thread_id=thread_id, memory=self.saver_cls.return_value
        )

    def test_history_requires_valid_thread_id(self):
        response = self.client.get(
            f"{QNA_URL}/chat/history",
            params={"thread_id": "not-a-uuid"},
            headers=HEADERS,
        )
        self.assertEqual(response.status_code, 422)
        self.chat.agent_get_history.assert_not_awaited()

    # --- /chat/agent_stream -----------------------------------------------

    def test_agent_stream_language_not_supported_returns_400(self):
        with mock.patch(
            f"{ROUTER}._resolve_thread_id",
            side_effect=LanguageNotSupportedError(),
        ):
            response = self.client.post(
                f"{QNA_URL}/chat/agent_stream",
                json={"query": "bonjour"},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "LANG_NOT_SUPPORTED")

    # --- /chat/agent ------------------------------------------------------

    def test_agent_without_thread_id_generates_one(self):
        session_id = uuid.uuid4()

        response = self.client.post(
            f"{QNA_URL}/chat/agent",
            json={"query": "What are the SDGs?", "sdg_filter": [6]},
            headers={**HEADERS, "X-Session-Id": str(session_id)},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["content"], "final answer")
        generated = uuid.UUID(body["thread_id"])

        kwargs = self.chat.agent_message.await_args.kwargs
        self.assertEqual(kwargs["thread_id"], generated)
        self.assertEqual(kwargs["query"], "What are the SDGs?")
        self.assertEqual(kwargs["sdg_filter"], [6])
        self.assertIs(kwargs["memory"], self.saver_cls.return_value)
        self.assertIs(kwargs["sp"], self.sp)
        self.assertEqual(kwargs["trace_context"]["thread_id"], str(generated))
        self.assertEqual(kwargs["trace_context"]["session_id"], str(session_id))
        self.assertEqual(kwargs["trace_context"]["endpoint"], "/api/v1/qna/chat/agent")

        self.data_collection.register_chat_data.assert_awaited_once_with(
            session_id=session_id,
            user_query="What are the SDGs?",
            conversation_id=generated,
            answer_content="final answer",
            sources=None,
        )

    def test_agent_keeps_given_thread_id(self):
        thread_id = uuid.uuid4()

        response = self.client.post(
            f"{QNA_URL}/chat/agent",
            json={"query": "q", "thread_id": str(thread_id)},
            headers=HEADERS,
        )

        self.assertEqual(response.json()["thread_id"], str(thread_id))
        self.assertEqual(
            self.chat.agent_message.await_args.kwargs["thread_id"], thread_id
        )

    @mock.patch(f"{ROUTER}.logger")
    def test_agent_answer_survives_data_collection_failure(self, logger_mock):
        self.data_collection.register_chat_data.side_effect = RuntimeError("db down")
        thread_id = uuid.uuid4()

        response = self.client.post(
            f"{QNA_URL}/chat/agent",
            json={"query": "q", "thread_id": str(thread_id)},
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "final answer")
        self.assertEqual(response.json()["thread_id"], str(thread_id))
        logger_mock.error.assert_called_once()
        self.assertEqual(
            logger_mock.error.call_args.args[0], "Error while registering chat data: %s"
        )

    @mock.patch(f"{ROUTER}.logger")
    def test_agent_answer_survives_stale_session_in_data_collection(self, logger_mock):
        # a domain error is an Exception now, so the fail-soft block catches it
        self.data_collection.register_chat_data.side_effect = SessionNotFoundError()

        response = self.client.post(
            f"{QNA_URL}/chat/agent", json={"query": "q"}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "final answer")
        self.assertIsNone(response.json().get("message_id"))
        logger_mock.error.assert_called_once()

    def test_agent_language_not_supported_returns_400(self):
        self.chat.agent_message.side_effect = LanguageNotSupportedError()

        response = self.client.post(
            f"{QNA_URL}/chat/agent", json={"query": "q"}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "LANG_NOT_SUPPORTED")
        self.data_collection.register_chat_data.assert_not_awaited()

    # --- thread ownership -------------------------------------------------

    def _thread_owned_by(self, user_id):
        self.saver.aget_tuple.return_value = _checkpoint(
            {"source": "loop", "step": 1, "owner_user_id": str(user_id)}
        )

    def _legacy_thread(self):
        self.saver.aget_tuple.return_value = _checkpoint({"source": "loop", "step": 1})

    def _get_history(self, thread_id, headers):
        return self.client.get(
            f"{QNA_URL}/chat/history",
            params={"thread_id": str(thread_id)},
            headers=headers,
        )

    def _post_agent(self, path, thread_id, headers):
        return self.client.post(
            f"{QNA_URL}/chat/{path}",
            json={"query": "q", "thread_id": str(thread_id)},
            headers=headers,
        )

    def _assert_thread_not_found(self, response):
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "THREAD_NOT_FOUND")

    def test_history_of_another_users_thread_returns_404(self):
        thread_id = uuid.uuid4()
        self._thread_owned_by(USER_A)

        response = self._get_history(thread_id, HEADERS_B)

        self._assert_thread_not_found(response)
        self.saver.aget_tuple.assert_awaited_once_with(
            {"configurable": {"thread_id": str(thread_id)}}
        )
        self.chat.agent_get_history.assert_not_awaited()

    def test_history_of_own_thread_is_returned(self):
        self._thread_owned_by(USER_A)
        self.chat.agent_get_history.return_value = [{"role": "user", "content": "hi"}]

        response = self._get_history(uuid.uuid4(), HEADERS_A)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{"role": "user", "content": "hi"}])

    def test_history_of_legacy_thread_without_owner_is_returned(self):
        self._legacy_thread()
        self.chat.agent_get_history.return_value = []

        response = self._get_history(uuid.uuid4(), HEADERS_B)

        self.assertEqual(response.status_code, 200)
        self.chat.agent_get_history.assert_awaited_once()

    def test_history_of_owned_thread_without_session_returns_404(self):
        self._thread_owned_by(USER_A)

        response = self._get_history(uuid.uuid4(), HEADERS)

        self._assert_thread_not_found(response)

    def test_history_of_owned_thread_with_unknown_session_returns_404(self):
        self._thread_owned_by(USER_A)

        response = self._get_history(
            uuid.uuid4(), {**HEADERS, "X-Session-Id": str(uuid.uuid4())}
        )

        self._assert_thread_not_found(response)

    def test_agent_on_another_users_thread_returns_404(self):
        self._thread_owned_by(USER_A)

        response = self._post_agent("agent", uuid.uuid4(), HEADERS_B)

        self._assert_thread_not_found(response)
        self.chat.agent_message.assert_not_awaited()
        self.data_collection.register_chat_data.assert_not_awaited()

    def test_agent_on_own_thread_writes_owner(self):
        thread_id = uuid.uuid4()
        self._thread_owned_by(USER_A)

        response = self._post_agent("agent", thread_id, HEADERS_A)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["thread_id"], str(thread_id))
        kwargs = self.chat.agent_message.await_args.kwargs
        self.assertEqual(kwargs["owner_user_id"], USER_A)

    def test_agent_on_legacy_thread_claims_it(self):
        self._legacy_thread()

        response = self._post_agent("agent", uuid.uuid4(), HEADERS_B)

        self.assertEqual(response.status_code, 200)
        kwargs = self.chat.agent_message.await_args.kwargs
        self.assertEqual(kwargs["owner_user_id"], USER_B)

    def test_agent_new_thread_skips_check_and_writes_owner(self):
        response = self.client.post(
            f"{QNA_URL}/chat/agent", json={"query": "q"}, headers=HEADERS_A
        )

        self.assertEqual(response.status_code, 200)
        self.saver.aget_tuple.assert_not_awaited()
        kwargs = self.chat.agent_message.await_args.kwargs
        self.assertEqual(kwargs["owner_user_id"], USER_A)

    def test_agent_stream_on_another_users_thread_returns_404(self):
        self._thread_owned_by(USER_A)

        response = self._post_agent("agent_stream", uuid.uuid4(), HEADERS_B)

        self._assert_thread_not_found(response)
        self.chat.agent_message.assert_not_awaited()

    def test_agent_stream_on_own_thread_streams_and_writes_owner(self):
        thread_id = uuid.uuid4()
        self._thread_owned_by(USER_A)

        async def _stream():
            yield {"status": "streaming", "content": "answer"}

        self.chat.agent_message = mock.AsyncMock(return_value=_stream())
        self.chat.agent_get_latest_docs = mock.AsyncMock(return_value=None)

        response = self._post_agent("agent_stream", thread_id, HEADERS_A)

        self.assertEqual(response.status_code, 200)
        self.assertIn('"content": "answer"', response.text)
        self.assertIn(str(thread_id), response.text)
        self.saver.aget_tuple.assert_awaited_once_with(
            {"configurable": {"thread_id": str(thread_id)}}
        )
        kwargs = self.chat.agent_message.await_args.kwargs
        self.assertEqual(kwargs["owner_user_id"], USER_A)

    def test_agent_stream_on_legacy_thread_is_allowed(self):
        self._legacy_thread()

        async def _stream():
            yield {"status": "streaming", "content": "answer"}

        self.chat.agent_message = mock.AsyncMock(return_value=_stream())
        self.chat.agent_get_latest_docs = mock.AsyncMock(return_value=None)

        response = self._post_agent("agent_stream", uuid.uuid4(), HEADERS_B)

        self.assertEqual(response.status_code, 200)
        kwargs = self.chat.agent_message.await_args.kwargs
        self.assertEqual(kwargs["owner_user_id"], USER_B)

    def test_agent_stream_new_thread_skips_check(self):
        async def _stream():
            yield {"status": "streaming", "content": "answer"}

        self.chat.agent_message = mock.AsyncMock(return_value=_stream())
        self.chat.agent_get_latest_docs = mock.AsyncMock(return_value=None)

        response = self.client.post(
            f"{QNA_URL}/chat/agent_stream", json={"query": "q"}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 200)
        self.saver.aget_tuple.assert_not_awaited()
        kwargs = self.chat.agent_message.await_args.kwargs
        self.assertIsNone(kwargs["owner_user_id"])


class ChatRouterDirectCallTests(unittest.IsolatedAsyncioTestCase):
    """`body.query is None` is rejected by get_agent_params over HTTP, so the
    route functions are called directly to check their own guard."""

    async def test_agent_stream_rejects_missing_query(self):
        with self.assertRaises(EmptyQueryError):
            await chat_router.agent_stream_response(
                request=_empty_request(),
                background_tasks=BackgroundTasks(),
                body=AgentContext(query=None),
                chatfactory=mock.MagicMock(),
                sp=mock.MagicMock(),
                data_collection=mock.MagicMock(),
            )

    async def test_agent_rejects_missing_query(self):
        chat = mock.MagicMock()
        chat.agent_message = mock.AsyncMock()
        with self.assertRaises(EmptyQueryError):
            await chat_router.agent_response(
                request=_empty_request(),
                background_tasks=BackgroundTasks(),
                body=AgentContext(query=None),
                chatfactory=chat,
                sp=mock.MagicMock(),
                data_collection=mock.MagicMock(),
            )
        chat.agent_message.assert_not_awaited()

    def test_get_agent_params_copies_body(self):
        thread_id = uuid.uuid4()
        body = AgentContext(
            query="q", thread_id=thread_id, corpora=("a",), sdg_filter=[1]
        )
        result = chat_router.get_agent_params(body)
        self.assertEqual(result, body)
        self.assertIsNot(result, body)

    def test_get_agent_params_rejects_none_query(self):
        with self.assertRaises(HTTPException) as ctx:
            chat_router.get_agent_params(AgentContext(query=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_build_agent_trace_context(self):
        thread_id = uuid.uuid4()
        ctx = chat_router._build_agent_trace_context(
            endpoint="/e",
            session_id=None,
            thread_id=thread_id,
            body=AgentContext(query="abcd", corpora=("c1", "c2"), sdg_filter=[3]),
        )
        self.assertEqual(
            ctx,
            {
                "endpoint": "/e",
                "feature": "chat_agent",
                "environment": settings.ENV,
                "session_id": None,
                "thread_id": str(thread_id),
                "query_length": 4,
                "sdg_filter": [3],
                "corpora": ["c1", "c2"],
            },
        )
