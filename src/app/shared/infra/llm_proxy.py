from abc import ABC
from typing import Any, Optional, Type, Union

import litellm
from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from langsmith.run_helpers import get_current_run_tree
from mistralai.client import Mistral
from pydantic import BaseModel

from src.app.utils.decorators import log_time_and_error
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


class LLMProxy(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        is_azure_model: bool = False,
        debug: bool = False,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.client = None
        self.is_azure_model = is_azure_model

        litellm.enable_json_schema_validation = True
        if debug:
            litellm._turn_on_debug()  # type: ignore

        if is_azure_model:
            if api_key is None or api_base is None or api_version is None:
                raise ValueError(
                    "For Azure models, api_key, api_base, and api_version must be provided."
                )

            logger.debug("Initializing Azure ChatCompletionsClient")

            self.client = ChatCompletionsClient(
                endpoint=api_base,
                credential=AzureKeyCredential(api_key),
                api_version=api_version,
            )
        else:
            # We assume that if it's not an Azure model, it's a Mistral model for now. This can be extended in the future to support other types of models.
            if api_key is None:
                raise ValueError("For Mistral models, api_key must be provided.")
            logger.debug("Initializing Mistral client")

            self.client = Mistral(
                api_key=api_key,
            )

    @log_time_and_error
    async def close_client(self):
        if self.client and self.is_azure_model:
            await self.client.close()

    def _get_langsmith_provider(self) -> str:
        return "azure" if self.is_azure_model else "mistral"

    def _extract_usage_metadata(self, response: Any) -> dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage") or response.get("usage_metadata")

        if usage is None:
            return None

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")

        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is None and isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens") or usage.get(
                "output_tokens"
            )

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None and isinstance(usage, dict):
            total_tokens = usage.get("total_tokens")

        usage_metadata: dict[str, int] = {}
        if prompt_tokens is not None:
            usage_metadata["input_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage_metadata["output_tokens"] = int(completion_tokens)
        if total_tokens is not None:
            usage_metadata["total_tokens"] = int(total_tokens)

        if not usage_metadata:
            return None

        if "total_tokens" not in usage_metadata:
            usage_metadata["total_tokens"] = usage_metadata.get(
                "input_tokens", 0
            ) + usage_metadata.get("output_tokens", 0)

        return usage_metadata

    def _record_langsmith_usage(self, response: Any) -> None:
        run_tree = get_current_run_tree()
        if run_tree is None:
            return

        usage_metadata = self._extract_usage_metadata(response)
        if usage_metadata is not None:
            run_tree.set(usage_metadata=usage_metadata)

        metadata = dict(getattr(run_tree, "metadata", {}) or {})
        metadata_to_add: dict[str, Any] = {}
        if "ls_provider" not in metadata:
            metadata_to_add["ls_provider"] = self._get_langsmith_provider()
        if "ls_model_name" not in metadata:
            metadata_to_add["ls_model_name"] = self.model
        if metadata_to_add:
            run_tree.add_metadata(metadata_to_add)

    @log_time_and_error
    async def completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    ) -> dict | str:
        if self.is_azure_model:
            return await self.az_completion(
                messages,
                response_format=response_format,
            )

        else:
            # We assume that if it's not an Azure model, it's a Mistral model for now. This can be extended in the future to support other types of models.
            return await self.mistral_completion(
                messages,
                response_format=response_format,
            )

    async def az_completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    ):
        if self.client is None:
            raise ValueError("Azure client is not initialized.")

        completion_kwargs = {}
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        response = await self.client.complete(
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
            top_p=0.1,
            model=self.model,
            **completion_kwargs,
        )

        self._record_langsmith_usage(response)

        return response.choices[0].message.content

    async def az_completion_stream(
        self,
        messages: list,
    ):
        if self.client is None:
            raise ValueError("Azure client is not initialized.")

        response = await self.client.complete(
            messages=messages, temperature=0.8, top_p=0.1, model=self.model, stream=True
        )

        return response

    async def completion_stream(
        self,
        messages: list,
    ):
        if self.is_azure_model:
            return await self.az_completion_stream(messages)

        return await self.mistral_completion_stream(messages)

    async def mistral_completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    ):
        if self.client is None:
            raise ValueError("Mistral client is not initialized.")

        completion_kwargs = {}
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        response = await self.client.chat.complete_async(
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
            top_p=0.1,
            model=self.model,
            **completion_kwargs,
        )

        self._record_langsmith_usage(response)

        return response.choices[0].message.content

    async def mistral_completion_stream(
        self,
        messages: list,
    ):
        if self.client is None:
            raise ValueError("Mistral client is not initialized.")

        response = await self.client.chat.stream_async(
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
            top_p=0.1,
            model=self.model,
        )
        return response
