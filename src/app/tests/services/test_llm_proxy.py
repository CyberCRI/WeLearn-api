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

    async def test_completion_forwards_response_format_to_mistral(self):
        response_format = {"type": "json_object"}
        with mock.patch.object(
            self.proxy, "mistral_completion", new=AsyncMock(return_value="text")
        ) as mistral_completion:
            await self.proxy.completion(
                messages=[{"role": "user", "content": "Hello"}],
                response_format=response_format,
            )

        mistral_completion.assert_awaited_once_with(
            [{"role": "user", "content": "Hello"}],
            response_format=response_format,
            max_tokens=2048,
            trace_context=None,
        )

    async def test_completion_forwards_response_format_to_azure(self):
        self.proxy.is_azure_model = True
        response_format = {"type": "json_object"}
        with mock.patch.object(
            self.proxy, "az_completion", new=AsyncMock(return_value="text")
        ) as az_completion:
            await self.proxy.completion(
                messages=[{"role": "user", "content": "Hello"}],
                response_format=response_format,
            )

        az_completion.assert_awaited_once_with(
            [{"role": "user", "content": "Hello"}],
            response_format=response_format,
            max_tokens=2048,
            trace_context=None,
        )

    async def test_completion_stream_routes_to_mistral_and_forwards_trace_context(self):
        messages = [{"role": "user", "content": "Hello"}]
        trace_context = {"trace_id": "abc123"}

        with mock.patch.object(
            self.proxy,
            "mistral_completion_stream",
            new=AsyncMock(return_value="mistral_stream"),
        ) as mistral_completion_stream, mock.patch.object(
            self.proxy,
            "az_completion_stream",
            new=AsyncMock(return_value="azure_stream"),
        ) as az_completion_stream:
            response = await self.proxy.completion_stream(
                messages=messages,
                trace_context=trace_context,
            )

        self.assertEqual(response, "mistral_stream")
        mistral_completion_stream.assert_awaited_once_with(
            messages,
            trace_context=trace_context,
        )
        az_completion_stream.assert_not_awaited()

    async def test_completion_stream_routes_to_azure_and_forwards_trace_context(self):
        self.proxy.is_azure_model = True
        messages = [{"role": "user", "content": "Hello"}]
        trace_context = {"trace_id": "xyz789"}

        with mock.patch.object(
            self.proxy,
            "az_completion_stream",
            new=AsyncMock(return_value="azure_stream"),
        ) as az_completion_stream, mock.patch.object(
            self.proxy,
            "mistral_completion_stream",
            new=AsyncMock(return_value="mistral_stream"),
        ) as mistral_completion_stream:
            response = await self.proxy.completion_stream(
                messages=messages,
                trace_context=trace_context,
            )

        self.assertEqual(response, "azure_stream")
        az_completion_stream.assert_awaited_once_with(
            messages,
            trace_context=trace_context,
        )
        mistral_completion_stream.assert_not_awaited()
