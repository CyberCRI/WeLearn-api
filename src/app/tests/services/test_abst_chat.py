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
from src.app.services.abst_chat import AbstractChat, ChatFactory
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


# test that when calling factory with "mistral" it calls mistral class
@mock.patch("src.app.services.abst_chat.Azure_Chat")
@mock.patch("src.app.services.abst_chat.Open_Chat")
@mock.patch("src.app.services.abst_chat.Mistral_Chat")
class TestChatFactory(unittest.TestCase):
    def test_create_chat_mistral(self, mock_mistral, mock_openai, mock_azure):
        ChatFactory().create_chat("mistral")
        assert mock_mistral.called
        assert not mock_openai.called
        assert not mock_azure.called

    def test_create_openai(self, mock_mistral, mock_openai, mock_azure):
        ChatFactory().create_chat("openai")
        assert not mock_mistral.called
        assert mock_openai.called
        assert not mock_azure.called

    def test_create_azure(self, mock_mistral, mock_openai, mock_azure):
        ChatFactory().create_chat("azure", "Meta-Llama-3.1-8B-Instruct")
        assert not mock_mistral.called
        assert mock_azure.called
        assert not mock_openai.called

    def test_chatfactory_error(self, *mocks):
        with self.assertRaises(ValueError):
            ChatFactory().create_chat("other_service")


# Mock concrete class for AbstractChat for testing purposes
class ConcreteChat(AbstractChat):
    def init_client(self):
        pass  # No-op implementation

    async def chat(self, type, model, messages):
        # Simulate a mock response
        return {"choices": [{"message": {"content": "Mocked chat response"}}]}

    async def chat_schema(self, response_format, model, messages):
        # Simulate a mock response
        return {"choices": [{"message": {"content": "Mocked chat response"}}]}

    async def streamed_chat(self, messages):
        # Simulate a streaming response
        yield "Streamed response"


class TestAbstractChat(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Instantiate ConcreteChat instead of AbstractChat
        self.chat = ConcreteChat("api_key", "model")

    def test_get_message_content(self):
        # Create a mocked chat response
        mocked_chat = create_chat_responses_mocks("this is a message")
        # Ensure the method returns the expected content
        self.assertEqual(
            self.chat.get_message_content(mocked_chat), "this is a message"
        )

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

    async def test_lang_not_supported(self):
        mocked_chat = create_chat_responses_mocks('{"ISO_CODE": "pt"}')
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(LanguageNotSupportedError):
            await self.chat._detect_language("fake message")

        mocked_chat = create_chat_responses_mocks("not json format")
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_language("fake message")

    async def test_lang_supported(self):
        mocked_chat = create_chat_responses_mocks('{"ISO_CODE": "en"}')
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "en"}

        mocked_chat = create_chat_responses_mocks('{"ISO_CODE": "fr"}')
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "fr"}

    async def test_detect_past_message(self):
        mocked_chat = create_chat_responses_mocks('{"REF_TO_PAST": true}')
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": True
        }

        mocked_chat = create_chat_responses_mocks(' {"REF_TO_PAST": true} ')
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": True
        }

        mocked_chat = create_chat_responses_mocks('{"REF_TO_PAST": false}')
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_past_message_ref("fake message", []) == {
            "REF_TO_PAST": False
        }

        mocked_chat = create_chat_responses_mocks('{"REF_TO_TOTO": false}')
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_past_message_ref("fake message", [])

    async def test_detect_past_message_invalid_format(self):
        mocked_chat = create_chat_responses_mocks("this is not a true/false answer")
        self.chat.chat = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_past_message_ref("fake message", [])

    async def test_reformulate_user_query_invalid_ref_to_pas(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": True}
        )

        self.chat.chat_schema = mock.AsyncMock()
        resp = await self.chat.reformulate_user_query("this is the user query", [])
        assert resp.QUERY_STATUS == "INVALID"
        self.chat.chat_schema.assert_not_called()

    async def test_reformulate_user_query_valid_ref_to_pas(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": True}
        )

        self.chat.chat_schema = mock.AsyncMock()
        resp = await self.chat.reformulate_user_query(
            "this is the user query",
            [
                {"message": "this is the past message"},
                {"message": "this is the second past message"},
            ],
        )
        assert resp.QUERY_STATUS == "REF_TO_PAST"
        self.chat.chat_schema.assert_not_called()

    async def test_reformulate_user_query_hist(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": False}
        )
        self.chat._detect_language = mock.AsyncMock()
        self.chat.chat_schema = mock.AsyncMock(return_value=ReformulatedQueryResponse())
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

        self.chat.chat_schema.assert_called_once()

    async def test_reformulate_user_query__INVALID__(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": False}
        )
        self.chat._detect_language = mock.AsyncMock()
        self.chat.chat_schema = mock.AsyncMock(
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

    async def test_reformulate_user_query_valueError(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": False}
        )
        self.chat.chat_schema = mock.AsyncMock(
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

    async def test_reformulate_user_chat_not_called_if_ref_to_past(self):
        self.chat._detect_past_message_ref = mock.AsyncMock(
            return_value={"REF_TO_PAST": True}
        )
        self.chat._detect_language = mock.AsyncMock()
        self.chat.chat_schema = mock.AsyncMock()
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
        self.chat.chat_schema.assert_not_called()
        assert reformulated == ReformulatedQueryResponse(QUERY_STATUS="REF_TO_PAST")

    async def test_get_new_questions(self):
        with mock.patch.object(
            self.chat, "_detect_language", new_callable=mock.AsyncMock
        ) as mock_detect_lang:
            self.chat.chat = mock.AsyncMock(
                return_value=create_chat_responses_mocks(
                    "%%Question 1?%% Question 2?%%"
                ),
            )
            new_questions = await self.chat.get_new_questions(
                "this is the user query", []
            )

            mock_detect_lang.assert_called_with("this is the user query")
            assert new_questions == {"NEW_QUESTIONS": ["Question 1?", "Question 2?"]}

    async def test_rephrase_message_stream_false(self):
        self.chat.chat = mock.AsyncMock()
        self.chat.streamed_chat = mock.AsyncMock()
        await self.chat.rephrase_message(
            message="this is the user query", history=[], docs=[], subject="default"
        )
        self.chat.chat.assert_called_once()
        self.chat.streamed_chat.assert_not_called()

    async def test_rephrase_message_stream_true(self):
        self.chat.chat = mock.AsyncMock()
        self.chat.streamed_chat = mock.AsyncMock()
        await self.chat.rephrase_message(
            message="this is the user query",
            history=[],
            docs=[],
            subject="default",
            streamed_ans=True,
        )
        self.chat.streamed_chat.assert_called_once()
        self.chat.chat.assert_not_called()

    async def test_chat_message(self):
        with mock.patch.object(
            self.chat, "_detect_language", new_callable=mock.AsyncMock
        ) as mock_detect_lang:
            self.chat.chat = mock.AsyncMock()
            self.chat.streamed_chat = mock.AsyncMock()

            await self.chat.chat_message(
                query="this is a query",
                history=[],
                docs=[],
                subject="default",
                streamed_ans=True,
                should_check_lang=True,
            )

            mock_detect_lang.assert_called_with("this is a query")
            self.chat.chat.assert_not_called()
            self.chat.streamed_chat.assert_called_once()

    async def test_flex_chat(self):
        self.chat.chat = mock.AsyncMock()
        await self.chat.flex_chat("this is a query")
        self.chat.chat.assert_called_once()
