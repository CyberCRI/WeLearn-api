import unittest
import uuid
from unittest import mock
from unittest.mock import MagicMock

import backoff
from fastapi.testclient import TestClient

from src.app.services.data_collection import get_data_collection_service
from src.app.core.config import settings
from src.app.models.chat import ReformulatedQueryResponse
from src.app.models.documents import Document as DocumentModel
from src.app.models.documents import DocumentPayloadModel
from src.app.services.exceptions import LanguageNotSupportedError
from src.main import app


client = TestClient(app)


JSON_NO_HIST = {
    "sources": [
        {
            "id": "testId",
            "payload": {
                "document_corpus": "testCorpus",
                "document_desc": "testDesc",
                "document_details": {
                    "author": "testAuthor",
                    "duration": 276,
                    "readability": 42.61,
                    "source": "au",
                },
                "document_id": "12345678-1234-5678-1234-567812345678",
                "document_lang": "en",
                "document_scrape_date": "testDate",
                "document_sdg": [11, 12, 13, 15, 2, 8],
                "document_title": "testTitle",
                "document_url": "testUrl",
                "slice_content": "testContent",
                "slice_sdg": 15,
            },
            "score": 0.636549,
            "version": 164925,
        }
    ],
    "query": "Bonjour?",
}

JSON = {
    **JSON_NO_HIST,
    "history": [
        {"role": "user", "content": "How to promote sustainable agriculture?"},
        {"role": "assistant", "content": "here is my answer"},
    ],
    "query": "here is my answer",
}


def make_post_headers(session_id: str | None = None):
    return {
        "X-API-Key": "test",
        "origin": "test",
        **({"X-Session-ID": session_id} if session_id else {}),
    }


def make_test_document_model():
    return DocumentModel(
        score=0.636549,
        payload=DocumentPayloadModel(
            document_corpus="testCorpus",
            document_desc="testDesc",
            document_details={
                "author": "testAuthor",
                "duration": 276,
                "readability": 42.61,
                "source": "au",
            },
            document_id="12345678-1234-5678-1234-567812345678",
            document_lang="en",
            document_sdg=[11, 12, 13, 15, 2, 8],
            document_title="testTitle",
            document_url="testUrl",
            slice_content="testContent",
            slice_sdg=15,
        ),
    )


class QnATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        backoff.on_exception = MagicMock()

        self.mock_data_collection = mock.AsyncMock()
        self.mock_data_collection.register_chat_data.return_value = (
            uuid.uuid4(),
            uuid.uuid4(),
        )
        app.dependency_overrides[get_data_collection_service] = (
            lambda: self.mock_data_collection
        )

        # register_chat_data
        self.patcher_register = mock.patch(
            "src.app.services.data_collection.DataCollection.register_chat_data",
            new=mock.AsyncMock(),
        )
        self.mock_register = self.patcher_register.start()
        self.addCleanup(self.patcher_register.stop)

        # check_api_key_sync
        self.patcher_api_key = mock.patch(
            "src.app.services.security.check_api_key_sync",
            new=mock.MagicMock(return_value=True),
        )
        self.mock_api_key = self.patcher_api_key.start()
        self.addCleanup(self.patcher_api_key.stop)

        # _detect_language
        self.patcher_detect_lang = mock.patch(
            "src.app.services.abst_chat.AbstractChat._detect_language",
        )
        self.mock_detect_language = self.patcher_detect_lang.start()
        self.addCleanup(self.patcher_detect_lang.stop)

        # chat_message
        self.patcher_chat_message = mock.patch(
            "src.app.services.abst_chat.AbstractChat.chat_message",
        )
        self.mock_chat_message = self.patcher_chat_message.start()
        self.addCleanup(self.patcher_chat_message.stop)

    def post(self, path, json_body, session_id=None):
        with TestClient(app) as client:
            return client.post(
                path, json=json_body, headers=make_post_headers(session_id)
            )

    async def test_chat(self):
        self.mock_chat_message.return_value = "ok"
        response = self.post(
            f"{settings.API_V1_STR}/qna/chat/answer", JSON, str(uuid.uuid4())
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "ok")

    async def test_chat_empty_history(self):
        self.mock_chat_message.return_value = "ok"

        response = self.post(
            f"{settings.API_V1_STR}/qna/chat/answer", JSON_NO_HIST, str(uuid.uuid4())
        )

        self.mock_chat_message.assert_called_with(
            query="Bonjour?",
            history=[],
            docs=[make_test_document_model()],
            subject=None,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "ok")

    async def test_chat_not_supported_lang(self):
        # mock raise LanguageNotSupportedError
        self.mock_chat_message.side_effect = LanguageNotSupportedError
        json_payload = JSON_NO_HIST.copy()
        json_payload["query"] = "Bom dia?"
        response = self.post(f"{settings.API_V1_STR}/qna/chat/answer", json_payload)
        self.assertEqual(response.status_code, 400)

    async def test_chat_rephrase(self):
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat.rephrase_message",
            return_value="ok",
        ) as mock_rephrase:
            self.post(f"{settings.API_V1_STR}/qna/chat/rephrase", JSON)

            mock_rephrase.assert_called_with(
                docs=[make_test_document_model()],
                message="here is my answer",
                history=JSON["history"],
                subject=None,
            )

    def test_new_questions_empty_query(self):
        response = self.post(
            f"{settings.API_V1_STR}/qna/reformulate/questions",
            {"history": [], "sources": [], "query": ""},
        )
        self.assertEqual(
            response.json(),
            {"detail": {"message": "Empty query", "code": "EMPTY_QUERY"}},
        )

    async def test_new_questions_ok(self):
        self.mock_detect_language.return_value = {"ISO_CODE": "en"}
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat.get_new_questions",
            return_value={"NEW_QUESTIONS": ["Your reformulated question"]},
        ) as new_questions_mock:
            response = self.post(
                f"{settings.API_V1_STR}/qna/reformulate/questions",
                {
                    "history": [],
                    "sources": [],
                    "query": "bonjour une recherche en français",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(new_questions_mock.call_count, 1)

    def test_reformulate_empty_query(self):
        response = self.post(
            f"{settings.API_V1_STR}/qna/reformulate/query",
            {"history": [], "sources": [], "query": ""},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": {"message": "Empty query", "code": "EMPTY_QUERY"}},
        )

    async def test_reformulate_ok(self):
        self.mock_detect_language.return_value = {"ISO_CODE": "en"}
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat._detect_past_message_ref",
            return_value={"REF_TO_PAST": "false", "CONFIDENCE": "0.9"},
        ), mock.patch(
            "src.app.services.abst_chat.AbstractChat.reformulate_user_query",
            return_value=ReformulatedQueryResponse(
                STANDALONE_QUESTION_EN="Your reformulated question",
                STANDALONE_QUESTION_FR="Votre question reformulée",
                USER_LANGUAGE="en",
                QUERY_STATUS="VALID",
            ),
        ) as standalone_mock:
            response = self.post(
                f"{settings.API_V1_STR}/qna/reformulate/query",
                {
                    "history": [],
                    "sources": [],
                    "query": "bonjour une recherche en français",
                },
            )

            standalone_mock.assert_called_once_with(
                query="bonjour une recherche en français", history=[]
            )
            self.assertEqual(response.status_code, 200)

    async def test_stream(self):
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat.chat_message"
        ) as stream_mock:
            response = self.post(f"{settings.API_V1_STR}/qna/stream", JSON)

            stream_mock.assert_called_with(
                streamed_ans=True,
                query="here is my answer",
                history=JSON["history"],
                docs=[make_test_document_model()],
                subject=None,
            )
            self.assertIsNotNone(response)
            self.assertEqual(response.status_code, 200)

    def test_chat_agent(self):
        with mock.patch(
            "psycopg.AsyncConnection.connect", new_callable=mock.AsyncMock
        ), mock.patch(
            "src.app.services.abst_chat.AbstractChat.agent_message"
        ) as agent_message_mock:
            agent_message_mock.return_value = {
                "messages": [
                    mock.Mock(content="Agent response 1"),
                    mock.Mock(content="Agent response 2"),
                ]
            }

            response = self.post(
                f"{settings.API_V1_STR}/qna/chat/agent",
                {
                    "query": "What are the SDGs?",
                    "thread_id": str(uuid.uuid4()),
                    "corpora": ["corpus1"],
                    "sdg_filter": [1, 2, 3],
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("content", response.json())
            self.assertIn("docs", response.json())
