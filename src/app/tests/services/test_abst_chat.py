import unittest
from unittest import mock

from mistralai.models.assistantmessage import AssistantMessage  # type: ignore
from mistralai.models.chatcompletionchoice import ChatCompletionChoice
from mistralai.models.chatcompletionresponse import (  # type: ignore
    ChatCompletionResponse,
)
from mistralai.models.usageinfo import UsageInfo

# test abst_chat file
# test ChatFactory class
# mock open ai api and mistralai api
from src.app.models.chat import ReformulatedQueryResponse
from src.app.services.abst_chat import AbstractChat
from src.app.services.exceptions import LanguageNotSupportedError


def create_chat_responses_mocks(response: str):
    return ChatCompletionResponse(
        id="1",
        object="obj",
        created=1,
        model="model",
        usage=UsageInfo(prompt_tokens=1, total_tokens=1, completion_tokens=1),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content=response, role="assistant"),
                finish_reason="stop",
            )
        ],
    )


class MockedLLMProxy:
    def __init__(self, model, api_key=None, api_base=None, api_version=None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

    async def completion(self, type, model, messages):
        return create_chat_responses_mocks('{"ISO_CODE": "en"}')

    async def completion_stream(self, messages):
        yield "Streamed response"


@mock.patch("src.app.services.abst_chat.LLMProxy")
class TestAbstractChat(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = AbstractChat(model="model", API_KEY="toto_api_key")

    @mock.patch("src.app.services.abst_chat.detect_language_from_entry")
    async def test_lang_error_helper(self, mock_detect_lang, *mocks):
        self.chat._detect_lang_with_llm = mock.AsyncMock()

        mock_detect_lang.side_effect = LanguageNotSupportedError
        await self.chat._detect_language("fake message")
        self.chat._detect_lang_with_llm.assert_called_once()

    @mock.patch(
        "src.app.services.abst_chat.detect_language_from_entry", return_value="en"
    )
    async def test_lang_ok(self, *mocks):
        lang = await self.chat._detect_language("fake message")
        assert lang == {"ISO_CODE": "en"}

    async def test_lang_not_supported(self, *mocks):
        mocked_chat = create_chat_responses_mocks('{"ISO_CODE": "pt"}')
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(LanguageNotSupportedError):
            await self.chat._detect_language("fake message")

        mocked_chat = create_chat_responses_mocks("not json format")
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_language("fake message")

    async def test_lang_supported(self, *mocks):
        mocked_chat = create_chat_responses_mocks('{"ISO_CODE": "en"}')
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "en"}

        mocked_chat = create_chat_responses_mocks('{"ISO_CODE": "fr"}')
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "fr"}

    async def test_detect_past_message(self, *mocks):
        mocked_chat = create_chat_responses_mocks('{"REF_TO_PAST": true}')
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": True
        }

        mocked_chat = create_chat_responses_mocks(' {"REF_TO_PAST": true} ')
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": True
        }

        mocked_chat = create_chat_responses_mocks('{"REF_TO_PAST": false}')
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": False
        }

        mocked_chat = create_chat_responses_mocks('{"REF_TO_TOTO": false}')
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_past_message_ref("fake message", [])

    async def test_detect_past_message_invalid_format(self, *mocks):
        mocked_chat = create_chat_responses_mocks("this is not a true/false answer")
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_past_message_ref("fake message", [])

    async def test_reformulate_user_query_invalid_ref_to_pas(self, *mocks):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": True}
        )

        self.chat.chat_client.completion = mock.AsyncMock()
        resp = await self.chat.reformulate_user_query("this is the user query", [])
        assert resp.QUERY_STATUS == "INVALID"
        self.chat.chat_client.completion.assert_not_called()

    async def test_reformulate_user_query_valid_ref_to_pas(self, *mocks):
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

    async def test_reformulate_user_query_hist(self, *mocks):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": False}
        )
        self.chat._detect_language = mock.AsyncMock()
        self.chat.chat_client.completion = mock.AsyncMock(
            return_value=ReformulatedQueryResponse()
        )
        await self.chat.reformulate_user_query(
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

        self.chat.chat_client.completion.assert_called_once()

    async def test_reformulate_user_query__INVALID__(self, *mocks):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": False}
        )
        self.chat._detect_language = mock.AsyncMock()
        self.chat.chat_client.completion = mock.AsyncMock(
            return_value=ReformulatedQueryResponse(QUERY_STATUS="INVALID"),
        )
        reformulated = await self.chat.reformulate_user_query(
            "this is the user query",
            [
                {"message": "this is the past message"},
                {"message": "this is the second past message"},
            ],
        )

        assert reformulated.QUERY_STATUS == "INVALID"

    async def test_reformulate_user_query_valueError(self, *mocks):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": False}
        )
        self.chat.chat_client.completion = mock.AsyncMock(
            return_value={
                "STANDALONE_QUESTION_EN": "Question 1?",
                "USER_LANGUAGE": "en",
                "STANDALONE_QUESTION_FR": "Question 2?",
            }
        )
        with self.assertRaises(ValueError):
            await self.chat.reformulate_user_query(
                "this is the user query",
                [
                    {"message": "this is the past message"},
                    {"message": "this is the second past message"},
                ],
            )

    async def test_reformulate_user_chat_not_called_if_ref_to_past(self, *mocks):
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

    async def test_get_new_questions(self, *mocks):
        with mock.patch.object(
            self.chat, "_detect_language", new_callable=mock.AsyncMock
        ) as mock_detect_lang:
            self.chat.chat_client.completion = mock.AsyncMock(
                return_value=create_chat_responses_mocks(
                    "%%Question 1?%% Question 2?%%"
                ),
            )
            new_questions = await self.chat.get_new_questions(
                "this is the user query", []
            )

            mock_detect_lang.assert_called_with("this is the user query")
            assert new_questions == {"NEW_QUESTIONS": ["Question 1?", "Question 2?"]}

    async def test_rephrase_message_stream_false(self, *mocks):
        self.chat.chat_client.completion = mock.AsyncMock()
        self.chat.chat_client.completion_stream = mock.AsyncMock()
        await self.chat.rephrase_message(
            message="this is the user query", history=[], docs=[], subject="default"
        )
        self.chat.chat_client.completion.assert_called_once()
        self.chat.chat_client.completion_stream.assert_not_called()

    async def test_rephrase_message_stream_true(self, *mocks):
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

    async def test_chat_message(self, *mocks):
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
                should_check_lang=True,
            )

            mock_detect_lang.assert_called_with("this is a query")
            self.chat.chat_client.completion.assert_not_called()
            self.chat.chat_client.completion_stream.assert_called_once()
