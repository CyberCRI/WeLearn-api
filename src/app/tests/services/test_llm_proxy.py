import unittest
from unittest import mock
from unittest.mock import AsyncMock

from src.app.shared.infra.llm_proxy import LLMProxy


class TestLLMProxy(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with mock.patch("src.app.shared.infra.llm_proxy.build_chat_model"):
            self.proxy = LLMProxy(model="fake_model", api_key="fake_key")

    async def test_completion_returns_text_content(self):
        self.proxy.client.ainvoke = AsyncMock(
            return_value=mock.Mock(content="text", usage_metadata=None)
        )
        response = await self.proxy.completion(
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(response, "text")
        self.assertIsInstance(response, str)

    async def test_completion_stringifies_list_content(self):
        self.proxy.client.ainvoke = AsyncMock(
            return_value=mock.Mock(
                content=[{"text": "hello "}, {"text": "world"}], usage_metadata=None
            )
        )
        response = await self.proxy.completion(
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(response, "hello world")

    async def test_completion_forwards_response_format_and_max_tokens(self):
        response_format = {"type": "json_object"}
        self.proxy.client.ainvoke = AsyncMock(
            return_value=mock.Mock(content="text", usage_metadata=None)
        )

        await self.proxy.completion(
            messages=[{"role": "user", "content": "Hello"}],
            response_format=response_format,
            max_tokens=123,
        )

        self.proxy.client.ainvoke.assert_awaited_once_with(
            [{"role": "user", "content": "Hello"}],
            max_tokens=123,
            response_format=response_format,
        )

    async def test_completion_stream_returns_native_langchain_astream(self):
        async def fake_astream(messages):
            yield mock.Mock(content="ab", response_metadata={})
            yield mock.Mock(content="", response_metadata={"finish_reason": "stop"})

        self.proxy.client.astream = fake_astream

        stream = await self.proxy.completion_stream(
            messages=[{"role": "user", "content": "Hello"}]
        )

        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[0].content, "ab")
        self.assertEqual(chunks[1].response_metadata["finish_reason"], "stop")

    async def test_completion_records_langsmith_usage_on_current_run(self):
        run_tree = mock.Mock()
        run_tree.metadata = {}
        self.proxy.client.ainvoke = AsyncMock(
            return_value=mock.Mock(
                content="text",
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            )
        )

        with mock.patch(
            "src.app.shared.infra.llm_proxy.get_current_run_tree",
            return_value=run_tree,
        ):
            result = await self.proxy.completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

        self.assertEqual(result, "text")
        run_tree.set.assert_called_once_with(
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
        )
        run_tree.add_metadata.assert_called_once_with(
            {"ls_provider": "mistral", "ls_model_name": "fake_model"}
        )

    async def test_record_langsmith_usage_does_not_overwrite_existing_provider_metadata(
        self,
    ):
        run_tree = mock.Mock()
        run_tree.metadata = {"ls_provider": "existing", "ls_model_name": "preset"}

        with mock.patch(
            "src.app.shared.infra.llm_proxy.get_current_run_tree",
            return_value=run_tree,
        ):
            self.proxy._record_langsmith_usage(
                mock.Mock(
                    usage_metadata={
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "total_tokens": 3,
                    }
                )
            )

        run_tree.set.assert_called_once_with(
            usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        )
        run_tree.add_metadata.assert_not_called()

    async def test_azure_model_uses_azure_chat_client(self):
        with mock.patch(
            "src.app.shared.infra.llm_proxy.build_chat_model"
        ) as mock_build_chat_model:
            proxy = LLMProxy(
                model="fake_model",
                api_key="fake_key",
                api_base="https://fake.endpoint",
                api_version="2024-01-01",
                is_azure_model=True,
            )
        mock_build_chat_model.assert_called_once_with(
            model="fake_model",
            api_key="fake_key",
            api_base="https://fake.endpoint",
            api_version="2024-01-01",
            is_azure_model=True,
            temperature=0.8,
            top_p=0.1,
            max_tokens=2048,
        )
        self.assertIs(proxy.client, mock_build_chat_model.return_value)
        self.assertEqual(proxy._get_langsmith_provider(), "azure")
