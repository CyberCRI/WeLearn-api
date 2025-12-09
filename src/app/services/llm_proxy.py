import json
from abc import ABC
from typing import Optional, Type, Union

import litellm
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from litellm import acompletion
from litellm.types.utils import ModelResponse
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

    @log_time_and_error
    async def completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    ) -> dict | str:

        logger.info("starting completion with model_name=%s", self.model)

        if self.is_azure_model:
            return await self.az_completion(messages)

        response = await acompletion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=None,
            messages=messages,
            response_format=response_format,
        )

        assert isinstance(response, ModelResponse)

        response = response["choices"][0]["message"]["content"].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(
                "Response content is not valid JSON. Returning raw content instead."
            )
            return response

    async def completion_stream(
        self,
        messages: list,
    ):

        if self.is_azure_model:
            return await self.az_completion_stream(messages)

        response = await acompletion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version,
            messages=messages,
            stream=True,
        )

        return response

    async def az_completion(self, messages: list):
        if self.client is None:
            raise ValueError("Azure client is not initialized.")

        response = self.client.complete(
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
            top_p=0.1,
            model=self.model,
        )

        return response.choices[0].message.content

    async def az_completion_stream(self, messages: list):
        if self.client is None:
            raise ValueError("Azure client is not initialized.")

        response = self.client.complete(
            messages=messages, temperature=0.8, top_p=0.1, model=self.model, stream=True
        )

        return response
