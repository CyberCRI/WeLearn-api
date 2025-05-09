import unittest
from unittest import mock

from litellm.types.utils import Choices, Message, ModelResponse

from src.app.services.llm_proxy import LLMProxy


def create_chat_responses_mocks(response: str):
    return ModelResponse(
        choices=[
            Choices(
                message=Message(content=response, role="assistant"),
            )
        ],
    )


@mock.patch("src.app.services.llm_proxy.completion")
@mock.patch("src.app.services.llm_proxy.acompletion", new_callable=mock.AsyncMock)
class TestLLMProxy(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = LLMProxy(model="fake_model")

    async def test_response_as_text(self, mocked_acompletion, *mocks):
        mocked_acompletion.return_value = create_chat_responses_mocks(response="text")
        response = await self.proxy.completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        self.assertIs(response, "text")
        self.assertIsInstance(response, str)

    async def test_response_as_dict(self, mocked_acompletion, *mocks):
        mocked_acompletion.return_value = create_chat_responses_mocks(
            response='{"key": "value"}'
        )
        response = await self.proxy.completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        self.assertIsInstance(response, dict)

    def test_stream(self, mocked_acompletion, mocked_completion):
        self.proxy.completion_stream(
            messages=[{"role": "user", "content": "Hello"}],
        )
        mocked_completion.assert_called_with(
            model="fake_model",
            api_key=None,
            api_base=None,
            api_version=None,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
