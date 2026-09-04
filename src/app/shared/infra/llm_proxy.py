from abc import ABC
from typing import Any, Optional, Type, Union

import litellm
from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from langsmith import traceable
from mistralai.client import Mistral
from pydantic import BaseModel

from src.app.shared.infra.tracing import TRACE_RUN_TYPE_LLM, TraceName
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

    @log_time_and_error
    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.COMPLETION_NON_AGENT.value,
    )
    async def completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        trace_context: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = 2048,
    ) -> dict | str:

        logger.info(
            "starting completion with model_name=%s trace_context=%s",
            self.model,
            trace_context,
        )

        if self.is_azure_model:
            return await self.az_completion(
                messages,
                response_format=response_format,
                trace_context=trace_context,
                max_tokens=max_tokens,
            )

        else:
            # We assume that if it's not an Azure model, it's a Mistral model for now. This can be extended in the future to support other types of models.
            return await self.mistral_completion(
                messages,
                max_tokens=max_tokens,
                response_format=response_format,
                trace_context=trace_context,
            )

    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.AZURE_COMPLETION_NON_AGENT.value,
    )
    async def az_completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        trace_context: Optional[dict[str, Any]] = None,
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

        return response.choices[0].message.content

    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.AZURE_COMPLETION_STREAM_NON_AGENT.value,
    )
    async def az_completion_stream(
        self,
        messages: list,
        trace_context: Optional[dict[str, Any]] = None,
    ):
        if self.client is None:
            raise ValueError("Azure client is not initialized.")

        response = await self.client.complete(
            messages=messages, temperature=0.8, top_p=0.1, model=self.model, stream=True
        )

        return response

    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.COMPLETION_STREAM_NON_AGENT.value,
    )
    async def completion_stream(
        self,
        messages: list,
        trace_context: Optional[dict[str, Any]] = None,
    ):
        logger.info(
            "starting completion_stream with model_name=%s trace_context=%s",
            self.model,
            trace_context,
        )

        if self.is_azure_model:
            return await self.az_completion_stream(
                messages,
                trace_context=trace_context,
            )

        return await self.mistral_completion_stream(
            messages,
            trace_context=trace_context,
        )

    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.MISTRAL_COMPLETION_NON_AGENT.value,
    )
    async def mistral_completion(
        self,
        messages: list,
        max_tokens: Optional[int] = 2048,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        trace_context: Optional[dict[str, Any]] = None,
    ):
        if self.client is None:
            raise ValueError("Mistral client is not initialized.")

        completion_kwargs = {}
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        response = await self.client.chat.complete_async(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=0.1,
            model=self.model,
            **completion_kwargs,
        )

        return response.choices[0].message.content

    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.MISTRAL_COMPLETION_STREAM_NON_AGENT.value,
    )
    async def mistral_completion_stream(
        self,
        messages: list,
        trace_context: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = 2048,
    ):
        if self.client is None:
            raise ValueError("Mistral client is not initialized.")

        response = await self.client.chat.stream_async(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=0.1,
            model=self.model,
        )
        return response
