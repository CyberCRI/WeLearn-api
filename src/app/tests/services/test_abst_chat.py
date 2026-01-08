import unittest
from unittest import mock

from src.app.models.chat import ReformulatedQueryResponse
from src.app.services.abst_chat import AbstractChat
from src.app.services.exceptions import LanguageNotSupportedError


class TestAbstractChat(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mocked_client = mock.AsyncMock()
        self.chat = AbstractChat(client=mocked_client)

    @mock.patch("src.app.services.abst_chat.detect_language_from_entry")
    async def test_lang_error_helper(self, mock_detect_lang):
        self.chat._detect_lang_with_llm = mock.AsyncMock()

        mock_detect_lang.side_effect = LanguageNotSupportedError
        await self.chat._detect_language("fake message")
        self.chat._detect_lang_with_llm.assert_called_once()

    @mock.patch(
        "src.app.services.abst_chat.detect_language_from_entry", return_value="en"
    )
    async def test_lang_ok(self, mock_detect_lang):
        lang = await self.chat._detect_language("fake message")
        assert lang == {"ISO_CODE": "en"}

    @mock.patch(
        "src.app.services.abst_chat.detect_language_from_entry",
        side_effect=LanguageNotSupportedError,
    )
    async def test_lang_not_supported(self, mock_detect_lang):
        mocked_chat = {"ISO_CODE": "pt"}
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)

        mocked_chat = "not json format"
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_language("fake message")

    @mock.patch(
        "src.app.services.abst_chat.detect_language_from_entry",
        side_effect=LanguageNotSupportedError,
    )
    async def test_lang_supported(self, mock_detect_lang):
        mocked_chat = {"ISO_CODE": "en"}
        self.chat.chat_client.completion = mock.AsyncMock(
            return_value={"ISO_CODE": "en"}
        )
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "en"}

        mocked_chat = {"ISO_CODE": "fr"}
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "fr"}

    async def test_detect_past_message_true(self):
        mocked_chat = {"REF_TO_PAST": True}
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": True
        }

    async def test_detect_past_message_false(self):
        mocked_chat = {"REF_TO_PAST": False}
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": False
        }

        mocked_chat = {"REF_TO_TOTO": False}
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_past_message_ref("fake message", [])

    async def test_detect_past_message_invalid_format(self):
        mocked_chat = "this is not a true/false answer"
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_past_message_ref("fake message", [])

    async def test_reformulate_user_query_invalid_ref_to_pas(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": True}
        )

        self.chat.chat_client.completion = mock.AsyncMock()
        resp = await self.chat.reformulate_user_query("this is the user query", [])
        assert resp.QUERY_STATUS == "INVALID"
        self.chat.chat_client.completion.assert_not_called()

    async def test_reformulate_user_query_valid_ref_to_pas(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": True}
        )

        self.chat.chat_client.completion = mock.AsyncMock()
        resp = await self.chat.reformulate_user_query(
            "this is the user query",
            [
                {"message": "this is the past message"},
                {"message": "this is the second past message"},
            ],
        )
        assert resp.QUERY_STATUS == "REF_TO_PAST"
        self.chat.chat_client.completion.assert_not_called()

    async def test_reformulate_user_chat_not_called_if_ref_to_past(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": True}
        )
        self.chat._detect_language = mock.AsyncMock()
        self.chat.chat_client.completion = mock.AsyncMock()
        reformulated = await self.chat.reformulate_user_query(
            "this is the user query",
            [
                {"message": "this is the past message"},
                {"message": "this is the second past message"},
            ],
        )

        self.chat._detect_past_message_ref.assert_called_with(
            "this is the user query",
            [
                {"message": "this is the past message"},
                {"message": "this is the second past message"},
            ],
        )
        self.chat.chat_client.completion.assert_not_called()
        assert reformulated == ReformulatedQueryResponse(QUERY_STATUS="REF_TO_PAST")

    async def test_get_new_questions(self):
        with mock.patch.object(
            self.chat, "_detect_language", new_callable=mock.AsyncMock
        ) as mock_detect_lang:
            self.chat.chat_client.completion = mock.AsyncMock(
                return_value="%%Question 1?%% Question 2?%%",
            )
            new_questions = await self.chat.get_new_questions(
                "this is the user query", []
            )

            mock_detect_lang.assert_called_with("this is the user query")
            assert new_questions == {"NEW_QUESTIONS": ["Question 1?", "Question 2?"]}

    async def test_rephrase_message_stream_false(self):
        self.chat.chat_client.completion = mock.AsyncMock()
        self.chat.chat_client.completion_stream = mock.AsyncMock()
        await self.chat.rephrase_message(
            message="this is the user query", history=[], docs=[], subject="default"
        )
        self.chat.chat_client.completion.assert_called_once()
        self.chat.chat_client.completion_stream.assert_not_called()

    async def test_rephrase_message_stream_true(self):
        self.chat.chat_client.completion = mock.AsyncMock()
        self.chat.chat_client.completion_stream = mock.AsyncMock()
        await self.chat.rephrase_message(
            message="this is the user query",
            history=[],
            docs=[],
            subject="default",
            streamed_ans=True,
        )
        self.chat.chat_client.completion_stream.assert_called_once()
        self.chat.chat_client.completion.assert_not_called()

    async def test_chat_message(self):
        with mock.patch.object(
            self.chat, "_detect_language", new_callable=mock.AsyncMock
        ) as mock_detect_lang:
            self.chat.chat_client.completion = mock.AsyncMock()
            self.chat.chat_client.completion_stream = mock.AsyncMock()

            await self.chat.chat_message(
                query="this is a query",
                history=[],
                docs=[],
                subject="default",
                streamed_ans=True,
            )

            mock_detect_lang.assert_called_with("this is a query")
            self.chat.chat_client.completion.assert_not_called()
            self.chat.chat_client.completion_stream.assert_called_once()
