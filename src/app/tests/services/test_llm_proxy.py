import unittest
from unittest import mock
from unittest.mock import AsyncMock

from src.app.shared.infra.llm_proxy import LLMProxy


class TestLLMProxy(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with mock.patch("src.app.shared.infra.llm_proxy.Mistral"):
            self.proxy = LLMProxy(model="fake_model", api_key="fake_key")

    async def test_response_as_text(self):
        with mock.patch.object(
            self.proxy, "mistral_completion", new=AsyncMock(return_value="text")
        ):
            response = await self.proxy.completion(
                messages=[{"role": "user", "content": "Hello"}],
            )
        self.assertEqual(response, "text")
        self.assertIsInstance(response, str)

    async def test_response_as_json_string(self):
        with mock.patch.object(
            self.proxy,
            "mistral_completion",
            new=AsyncMock(return_value='{"key": "value"}'),
        ):
            response = await self.proxy.completion(
                messages=[{"role": "user", "content": "Hello"}],
            )
        self.assertIsInstance(response, str)
        self.assertEqual(response, '{"key": "value"}')
