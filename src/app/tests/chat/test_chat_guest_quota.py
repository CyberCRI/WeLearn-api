import unittest
import uuid
from unittest import mock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from src.app.chat.services.abst_chat import get_chat_service
from src.app.core.config import settings
from src.app.data_collection.services.data_collection import get_data_collection_service
from src.app.search.services.search import get_search_service
from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.tests.user.guest_usage_db import guest_left, use_guest_usage_db
from src.app.user.infra.keycloak import KeycloakUser
from src.app.user.services import account, guest_quota
from src.main import app

QNA_URL = f"{settings.API_V1_STR}/qna"
ROUTER = "src.app.chat.api.router"
THREAD_ACCESS = "src.app.chat.services.thread_access"
LIMIT = settings.GUEST_CHAT_DAILY_LIMIT

USER_A, SESSION_A = uuid.uuid4(), uuid.uuid4()
USER_B, SESSION_B = uuid.uuid4(), uuid.uuid4()
SESSION_USERS = {SESSION_A: USER_A, SESSION_B: USER_B}
HEADERS_A = {"X-API-Key": "test", "X-Session-Id": str(SESSION_A)}
HEADERS_B = {"X-API-Key": "test", "X-Session-Id": str(SESSION_B)}


def _user_of_session(session_id):
    if session_id not in SESSION_USERS:
        raise SessionNotFoundError()
    return SESSION_USERS[session_id]


def _fake_connect():
    conn = mock.MagicMock()
    conn.execute = mock.AsyncMock()
    conn.commit = mock.AsyncMock()
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=conn)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.AsyncMock(return_value=cm)


def _left(user_id) -> int:
    return guest_left("chat", user_id)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class ChatGuestQuotaTests(unittest.TestCase):
    def setUp(self):
        self.chat = mock.MagicMock()
        self.chat.agent_message = mock.AsyncMock(
            return_value={"messages": [HumanMessage("q"), AIMessage("answer")]}
        )
        self.chat.agent_get_latest_docs = mock.AsyncMock(return_value=None)
        self.data_collection = mock.MagicMock()
        self.data_collection.register_chat_data = mock.AsyncMock(
            return_value=(uuid.uuid4(), uuid.uuid4())
        )
        app.dependency_overrides[get_chat_service] = lambda: self.chat
        app.dependency_overrides[get_search_service] = lambda: mock.MagicMock()
        app.dependency_overrides[get_data_collection_service] = (
            lambda: self.data_collection
        )

        self.saver = mock.MagicMock()
        self.saver.aget_tuple = mock.AsyncMock(return_value=None)
        for target, kwargs in (
            (f"{ROUTER}.psycopg.AsyncConnection.connect", {"new": _fake_connect()}),
            (f"{ROUTER}.AsyncPostgresSaver", {"return_value": self.saver}),
            (f"{THREAD_ACCESS}.AsyncPostgresSaver", {"return_value": self.saver}),
            (
                "src.app.chat.services.chat_utils.AsyncPostgresSaver",
                {"return_value": self.saver},
            ),
            (
                f"{THREAD_ACCESS}.get_user_from_session_id",
                {"side_effect": _user_of_session},
            ),
            (
                f"{guest_quota.__name__}.get_user_for_session_renewal",
                {"side_effect": _user_of_session},
            ),
            (account.__name__ + ".settings.GUEST_RESTRICTIONS_ENABLED", {"new": True}),
        ):
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        use_guest_usage_db(self, USER_A, USER_B)
        self.client = TestClient(app)
        self.safe_client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        for dep in (get_chat_service, get_search_service, get_data_collection_service):
            app.dependency_overrides.pop(dep, None)

    def _ask(self, headers, client=None, **body):
        return (client or self.client).post(
            f"{QNA_URL}/chat/agent", json={"query": "q", **body}, headers=headers
        )

    def _stream(self, headers, chunks=None, error=None):
        async def _gen():
            for chunk in chunks or []:
                yield chunk
            if error:
                raise error

        self.chat.agent_message = mock.AsyncMock(return_value=_gen())
        return self.client.post(
            f"{QNA_URL}/chat/agent_stream", json={"query": "q"}, headers=headers
        )

    def assert_quota_exceeded(self, response):
        self.assertEqual(response.status_code, 403)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "GUEST_QUOTA_EXCEEDED")
        self.assertEqual(detail["feature"], "chat")
        self.assertEqual(response.headers["Retry-After"], str(detail["retry_after"]))

    # --- /chat/agent -------------------------------------------------------

    def test_guest_gets_the_daily_limit_then_403(self):
        for left in range(LIMIT - 1, -1, -1):
            response = self._ask(HEADERS_A)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["X-Guest-Remaining"], str(left))

        response = self._ask(HEADERS_A)

        self.assert_quota_exceeded(response)
        self.assertEqual(self.chat.agent_message.await_count, LIMIT)

    def test_quota_is_per_guest(self):
        for _ in range(LIMIT):
            self._ask(HEADERS_A)

        self.assertEqual(self._ask(HEADERS_B).status_code, 200)

    def test_failed_answer_is_refunded(self):
        self.chat.agent_message.side_effect = RuntimeError("mistral down")

        response = self._ask(HEADERS_A, client=self.safe_client)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(_left(USER_A), LIMIT)

    def test_rejected_request_is_not_counted(self):
        self.assertEqual(self._ask(HEADERS_A, query="").status_code, 400)
        self.assertEqual(_left(USER_A), LIMIT)

    def test_thread_of_another_user_is_not_counted(self):
        checkpoint = mock.MagicMock(metadata={"owner_user_id": str(USER_B)})
        self.saver.aget_tuple.return_value = checkpoint

        response = self._ask(HEADERS_A, thread_id=str(uuid.uuid4()))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(_left(USER_A), LIMIT)

    @mock.patch.object(account, "decode_keycloak_token")
    def test_account_is_never_limited(self, decode):
        decode.return_value = KeycloakUser(keycloak_id=uuid.uuid4())
        headers = {**HEADERS_A, "Authorization": "Bearer a.b.c"}

        for _ in range(LIMIT + 2):
            response = self._ask(headers)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("X-Guest-Remaining", response.headers)
        self.assertEqual(_left(USER_A), LIMIT)

    def test_restrictions_off_is_not_limited(self):
        with mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", False):
            for _ in range(LIMIT + 2):
                self.assertEqual(self._ask(HEADERS_A).status_code, 200)

    def test_guest_without_session_gets_401(self):
        response = self._ask({"X-API-Key": "test"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"]["code"], "SESSION_NOT_FOUND")
        self.chat.agent_message.assert_not_awaited()

    # --- /chat/agent_stream ------------------------------------------------

    def test_stream_is_counted_and_sends_what_is_left(self):
        response = self._stream(
            HEADERS_A, chunks=[{"status": "streaming", "content": "answer"}]
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Guest-Remaining"], str(LIMIT - 1))
        self.assertEqual(_left(USER_A), LIMIT - 1)

    def test_stream_is_refused_before_it_starts(self):
        for _ in range(LIMIT):
            self._ask(HEADERS_A)

        response = self._stream(
            HEADERS_A, chunks=[{"status": "streaming", "content": "answer"}]
        )

        self.assert_quota_exceeded(response)
        self.chat.agent_message.assert_not_awaited()

    def test_stream_failing_before_any_content_is_refunded(self):
        response = self._stream(HEADERS_A, error=RuntimeError("mistral down"))

        self.assertEqual(response.status_code, 200)  # the error is an SSE event
        self.assertEqual(_left(USER_A), LIMIT)

    def test_stream_failing_after_content_is_counted(self):
        response = self._stream(
            HEADERS_A,
            chunks=[{"status": "streaming", "content": "half an answer"}],
            error=RuntimeError("mistral down"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(_left(USER_A), LIMIT - 1)
