import unittest
from unittest import mock
from unittest.mock import MagicMock

import backoff
from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.app.models.chat import ReformulatedQueryResponse
from src.app.models.documents import Document as DocumentModel
from src.app.models.documents import DocumentPayloadModel
from src.app.services.exceptions import LanguageNotSupportedError
from src.main import app

client = TestClient(app)


source_example = [
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
            "document_id": "testDocId",
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
]

JSON_NO_HIST = {
    "sources": source_example,
    "query": "Bonjour?",
}

JSON = {
    "sources": source_example,
    "history": [
        {
            "role": "user",
            "content": "How to promote sustainable agriculture?",
        },
        {
            "role": "assistant",
            "content": "here is my answer",
        },
    ],
    "query": "here is my answer",
}


@mock.patch(
    "src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True)
)
@mock.patch(
    "src.app.services.abst_chat.AbstractChat._detect_language",
)
@mock.patch("src.app.services.abst_chat.AbstractChat.chat_message")
class QnATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        backoff.on_exception = MagicMock()

    async def test_chat(self, chat_mock, *mocks):
        chat_mock.return_value = "ok"
        response = client.post(
            f"{settings.API_V1_STR}/qna/chat/answer",
            json=JSON,
            headers={"X-API-Key": "test"},
        )

        response_json = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_json, "ok")

    async def test_chat_empty_history(self, chat_mock, *mocks):
        chat_mock.return_value = "ok"
        response = client.post(
            f"{settings.API_V1_STR}/qna/chat/answer",
            json=JSON_NO_HIST,
            headers={"X-API-Key": "test"},
        )

        chat_mock.assert_called_with(
            query="Bonjour?",
            history=[],
            docs=[
                DocumentModel(
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
                        document_id="testDocId",
                        document_lang="en",
                        document_sdg=[11, 12, 13, 15, 2, 8],
                        document_title="testTitle",
                        document_url="testUrl",
                        slice_content="testContent",
                        slice_sdg=15,
                    ),
                )
            ],
            subject=None,
        )
        response_json = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_json, "ok")

    async def test_chat_not_supported_lang(self, chat_mock, *mocks):
        # mock raise LanguageNotSupportedError
        chat_mock.side_effect = LanguageNotSupportedError
        JSON_NO_HIST["query"] = "Bom dia?"
        response = client.post(
            f"{settings.API_V1_STR}/qna/chat/answer",
            json=JSON_NO_HIST,
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 400)

    async def test_chat_rephrase(self, *mocks):
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat.rephrase_message",
            return_value="ok",
        ) as mock_rephrase:
            client.post(
                f"{settings.API_V1_STR}/qna/chat/rephrase",
                json=JSON,
                headers={"X-API-Key": "test"},
            )

            mock_rephrase.assert_called_with(
                docs=[
                    DocumentModel(
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
                            document_id="testDocId",
                            document_lang="en",
                            document_sdg=[11, 12, 13, 15, 2, 8],
                            document_title="testTitle",
                            document_url="testUrl",
                            slice_content="testContent",
                            slice_sdg=15,
                        ),
                    )
                ],
                message="here is my answer",
                history=[
                    {
                        "role": "user",
                        "content": "How to promote sustainable agriculture?",
                    },
                    {"role": "assistant", "content": "here is my answer"},
                ],
                subject=None,
            )

    def test_new_questions_empty_query(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/qna/reformulate/questions",
            json={"history": [], "sources": [], "query": ""},
            headers={"X-API-Key": "test"},
        )
        # self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "message": "Empty query",
                    "code": "EMPTY_QUERY",
                }
            },
        )

    async def test_new_questions_ok(self, mock_chat_completion, mock__detect_language):
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat.get_new_questions",
            return_value={"NEW_QUESTIONS": ["Your reformulated question"]},
        ) as new_questions_mock:
            mock__detect_language.return_value = {"ISO_CODE": "en"}
            response = client.post(
                f"{settings.API_V1_STR}/qna/reformulate/questions",  # noqa: E501
                json={
                    "history": [],
                    "sources": [],
                    "query": "bonjour une recherche en français",
                },
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(new_questions_mock.call_count, 1)

    def test_reformulate_empty_query(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/qna/reformulate/query",
            json={"history": [], "sources": [], "query": ""},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "message": "Empty query",
                    "code": "EMPTY_QUERY",
                }
            },
        )

    async def test_reformulate_ok(self, mock_chat_completion, mock__detect_language):
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
            mock__detect_language.return_value = {"ISO_CODE": "en"}
            response = client.post(
                f"{settings.API_V1_STR}/qna/reformulate/query",  # noqa: E501
                json={
                    "history": [],
                    "sources": [],
                    "query": "bonjour une recherche en français",
                },
                headers={"X-API-Key": "test"},
            )

            standalone_mock.assert_called_once_with(
                query="bonjour une recherche en français", history=[]
            )
            self.assertEqual(response.status_code, 200)

    async def test_stream(self, *mocks):
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat.chat_message",
        ) as stream_mock:
            response = client.post(
                f"{settings.API_V1_STR}/qna/stream",
                json=JSON,
                headers={"X-API-Key": "test"},
            )

            stream_mock.assert_called_with(
                streamed_ans=True,
                query="here is my answer",
                history=JSON["history"],
                docs=[
                    DocumentModel(
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
                            document_id="testDocId",
                            document_lang="en",
                            document_sdg=[11, 12, 13, 15, 2, 8],
                            document_title="testTitle",
                            document_url="testUrl",
                            slice_content="testContent",
                            slice_sdg=15,
                        ),
                    )
                ],
                subject=None,
            )

            self.assertIsNotNone(response)

            self.assertEqual(response.status_code, 200)
