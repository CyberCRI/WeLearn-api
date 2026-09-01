import unittest
from unittest import mock
from unittest.mock import AsyncMock
from types import SimpleNamespace

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
        )

    async def test_completion_stream_routes_to_mistral(self):
        messages = [{"role": "user", "content": "Hello"}]

        with mock.patch.object(
            self.proxy,
            "mistral_completion_stream",
            new=AsyncMock(return_value="mistral_stream"),
        ) as mistral_completion_stream, mock.patch.object(
            self.proxy,
            "az_completion_stream",
            new=AsyncMock(return_value="azure_stream"),
        ) as az_completion_stream:
            response = await self.proxy.completion_stream(messages=messages)

        self.assertEqual(response, "mistral_stream")
        mistral_completion_stream.assert_awaited_once_with(messages)
        az_completion_stream.assert_not_awaited()

    async def test_completion_stream_routes_to_azure(self):
        self.proxy.is_azure_model = True
        messages = [{"role": "user", "content": "Hello"}]

        with mock.patch.object(
            self.proxy,
            "az_completion_stream",
            new=AsyncMock(return_value="azure_stream"),
        ) as az_completion_stream, mock.patch.object(
            self.proxy,
            "mistral_completion_stream",
            new=AsyncMock(return_value="mistral_stream"),
        ) as mistral_completion_stream:
            response = await self.proxy.completion_stream(messages=messages)

        self.assertEqual(response, "azure_stream")
        az_completion_stream.assert_awaited_once_with(messages)
        mistral_completion_stream.assert_not_awaited()

    async def test_mistral_completion_records_langsmith_usage_on_current_run(self):
        messages = [{"role": "user", "content": "Hello"}]
        run_tree = mock.Mock()
        run_tree.metadata = {}
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content="text"))],
        )

        self.proxy.client.chat.complete_async = AsyncMock(return_value=response)

        with mock.patch(
            "src.app.shared.infra.llm_proxy.get_current_run_tree",
            return_value=run_tree,
        ):
            result = await self.proxy.mistral_completion(messages)

        self.assertEqual(result, "text")
        run_tree.set.assert_called_once_with(
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
        )
        run_tree.add_metadata.assert_called_once_with(
            {"ls_provider": "mistral", "ls_model_name": "fake_model"}
        )

    async def test_azure_completion_records_langsmith_usage_on_current_run(self):
        self.proxy.is_azure_model = True
        messages = [{"role": "user", "content": "Hello"}]
        run_tree = mock.Mock()
        run_tree.metadata = {}
        response = SimpleNamespace(
            usage={"prompt_tokens": 13, "completion_tokens": 5},
            choices=[SimpleNamespace(message=SimpleNamespace(content="azure text"))],
        )
        self.proxy.client = SimpleNamespace(complete=AsyncMock(return_value=response))

        with mock.patch(
            "src.app.shared.infra.llm_proxy.get_current_run_tree",
            return_value=run_tree,
        ):
            result = await self.proxy.az_completion(messages)

        self.assertEqual(result, "azure text")
        run_tree.set.assert_called_once_with(
            usage_metadata={"input_tokens": 13, "output_tokens": 5, "total_tokens": 18}
        )
        run_tree.add_metadata.assert_called_once_with(
            {"ls_provider": "azure", "ls_model_name": "fake_model"}
        )

    async def test_record_langsmith_usage_does_not_overwrite_existing_provider_metadata(self):
        run_tree = mock.Mock()
        run_tree.metadata = {"ls_provider": "existing", "ls_model_name": "preset"}

        with mock.patch(
            "src.app.shared.infra.llm_proxy.get_current_run_tree",
            return_value=run_tree,
        ):
            self.proxy._record_langsmith_usage(
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
                )
            )

        run_tree.set.assert_called_once_with(
            usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        )
        run_tree.add_metadata.assert_not_called()
