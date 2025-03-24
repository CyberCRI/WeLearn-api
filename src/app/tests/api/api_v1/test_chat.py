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
@mock.patch("src.app.services.abst_chat.Open_Chat.chat")
class QnATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        backoff.on_exception = MagicMock()

    async def test_chat(self, chat_mock, mock_check_language):
        mock_check_language.return_value = {"ISO_CODE": "en"}
        response = client.post(
            f"{settings.API_V1_STR}/qna/chat/answer",
            json=JSON,
            headers={"X-API-Key": "test"},
        )

        chat_mock.assert_called_with(
            type="text",
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": '\nCONTEXT: You are an expert in sustainable development goals (SDGs).\n\nOBJECTIVE: Answer the user\'s question based on the provided articles (enclosed in XML tags). Always include the reference of the article at the end of the sentence using the following format: <a href="http://document_url" target="_blank">[Doc 2]</a>.\n\nSTYLE: Structured, conversational, and easy to understand, as if explaining to a friend. Always include the reference of the article at the end of the sentence using the following format: <a href="http://document_url" target="_blank">[Doc 2]</a>.\n\nTONE: Informative yet engaging.\n\nAUDIENCE: Non-technical readers, university students aged 18-25 years, on a General cursus.\n\nRESPONSE: It is crucial to use the <a> tag; otherwise, the answer will be considered invalid. Provide a clear and structured response based on the articles and questions provided. Use breaks, bullet points, and lists to structure your answers if relevant. You don\'t have to use all articles, only if it makes sense in the conversation. Answer in the same language as the user did.\n',
                },
                {"role": "user", "content": "How to promote sustainable agriculture?"},
                {"role": "assistant", "content": "here is my answer"},
                {
                    "role": "user",
                    "content": "\nArticles:\n<article>\nDoc 1: testTitle\ntestContent\n\nurl:testUrl</article>\n\nQuestion: here is my answer\n\nIMPORTANT:\n- The answer must be formulated in the same language as the question. Language: {'ISO_CODE': 'en'}.\n- Answer with the facts listed in the articles above. If there isn't enough information, say you don't know.\n- Every element of the answer must be supported by a reference to the article.\n- Add the reference of the article with a <a> tag as follows: <a href=\"http://document_url\" target=\"_blank\">[Doc 2]</a>. The target=\"_blank\" attribute is mandatory.\n- It is very important to use the <a> tag; otherwise, the answer will be considered invalid.\n",
                },
            ],
        )
        self.assertIsNotNone(response)

    async def test_chat_empty_history(self, chat_mock, mock_check_language):
        mock_check_language.return_value = {"ISO_CODE": "en"}

        response = client.post(
            f"{settings.API_V1_STR}/qna/chat/answer",
            json=JSON_NO_HIST,
            headers={"X-API-Key": "test"},
        )

        chat_mock.assert_called_with(
            type="text",
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": '\nCONTEXT: You are an expert in sustainable development goals (SDGs).\n\nOBJECTIVE: Answer the user\'s question based on the provided articles (enclosed in XML tags). Always include the reference of the article at the end of the sentence using the following format: <a href="http://document_url" target="_blank">[Doc 2]</a>.\n\nSTYLE: Structured, conversational, and easy to understand, as if explaining to a friend. Always include the reference of the article at the end of the sentence using the following format: <a href="http://document_url" target="_blank">[Doc 2]</a>.\n\nTONE: Informative yet engaging.\n\nAUDIENCE: Non-technical readers, university students aged 18-25 years, on a General cursus.\n\nRESPONSE: It is crucial to use the <a> tag; otherwise, the answer will be considered invalid. Provide a clear and structured response based on the articles and questions provided. Use breaks, bullet points, and lists to structure your answers if relevant. You don\'t have to use all articles, only if it makes sense in the conversation. Answer in the same language as the user did.\n',
                },
                {
                    "role": "user",
                    "content": "\nArticles:\n<article>\nDoc 1: testTitle\ntestContent\n\nurl:testUrl</article>\n\nQuestion: Bonjour?\n\nIMPORTANT:\n- The answer must be formulated in the same language as the question. Language: {'ISO_CODE': 'en'}.\n- Answer with the facts listed in the articles above. If there isn't enough information, say you don't know.\n- Every element of the answer must be supported by a reference to the article.\n- Add the reference of the article with a <a> tag as follows: <a href=\"http://document_url\" target=\"_blank\">[Doc 2]</a>. The target=\"_blank\" attribute is mandatory.\n- It is very important to use the <a> tag; otherwise, the answer will be considered invalid.\n",
                },
            ],
        )

        self.assertIsNotNone(response)

    async def test_chat_not_supported_lang(self, chat_mock, mock__detect_language):
        # mock raise LanguageNotSupportedError
        mock__detect_language.side_effect = LanguageNotSupportedError
        JSON_NO_HIST["query"] = "Bom dia?"
        response = client.post(
            f"{settings.API_V1_STR}/qna/chat/answer",
            json=JSON_NO_HIST,
            headers={"X-API-Key": "test"},
        )

        chat_mock.assert_not_called()
        self.assertIsNotNone(response)

    async def test_chat_rephrase(self, mock_chat, mock_check_language):
        with mock.patch(
            "src.app.services.abst_chat.AbstractChat.chat_message",
            return_value=[
                {"role": "user", "content": "How to promote sustainable agriculture?"}
            ],
        ):
            client.post(
                f"{settings.API_V1_STR}/qna/chat/rephrase",
                json=JSON,
                headers={"X-API-Key": "test"},
            )

            mock_chat.assert_called_with(
                type="text",
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": '\nCONTEXT: You are an expert in sustainable development goals (SDGs).\n\nOBJECTIVE: Answer the user\'s question based on the provided articles (enclosed in XML tags). Always include the reference of the article at the end of the sentence using the following format: <a href="http://document_url" target="_blank">[Doc 2]</a>.\n\nSTYLE: Structured, conversational, and easy to understand, as if explaining to a friend. Always include the reference of the article at the end of the sentence using the following format: <a href="http://document_url" target="_blank">[Doc 2]</a>.\n\nTONE: Informative yet engaging.\n\nAUDIENCE: Non-technical readers, university students aged 18-25 years, on a General cursus.\n\nRESPONSE: It is crucial to use the <a> tag; otherwise, the answer will be considered invalid. Provide a clear and structured response based on the articles and questions provided. Use breaks, bullet points, and lists to structure your answers if relevant. You don\'t have to use all articles, only if it makes sense in the conversation. Answer in the same language as the user did.\n',
                    },
                    {
                        "role": "user",
                        "content": "How to promote sustainable agriculture?",
                    },
                    {
                        "role": "user",
                        "content": '\nCONTEXT: You are a sustainable development goals (SDGs) expert. You are given a prompt and extracted parts of documents. Each document is delimited with XML tags <article> </article>.\n\nOBJECTIVE: Reformulate the given prompt based on the chat conversation and given articles. Always add the reference of the article at the end of the sentence (as follows, <a href="http://document_url" target="_blank">[Doc 2]</a>).\n\nSTYLE: Structured, conversational, and easy to understand, like explaining to a friend. Always add the reference of the article at the end of the sentence (as follows, <a href="http://document_url" target="_blank">[Doc 2]</a>).\n\nTONE: Informative yet engaging.\n\nAUDIENCE: Non-technical readers, university students aged 18-25 years.\n\nRESPONSE: It is very important to use the <a> tag; otherwise, the answer will be considered invalid. Provide a clear and structured answer based on the articles and questions provided. If relevant, use breaks, bullet points, and lists to structure your answers. You don\'t have to use all articles, only if it makes sense in the conversation. Use the same language as the user did.\n\nIMPORTANT:\n- You must answer in the same language as the question.\n- Answer with the facts listed in the list of articles above. If there isn\'t enough information, say you don\'t know.\n- Every element of the answer must be supported by a reference to the article.\n- Add the reference of the article with a <a> tag as follows: <a href="http://document_url" target="_blank">[Doc 2]</a>. The target="_blank" attribute is mandatory.\n- It is very important to use the <a> tag; otherwise, the answer will be considered invalid.\n\nArticles:\n<article>\nDoc 1: testTitle\ntestContent\n\nurl:testUrl</article>\n\nPrompt: here is my answer\n\nReformulated prompt:\n',
                    },
                ],
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
