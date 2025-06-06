import json
from abc import ABC
from typing import Optional, Type, Union

import litellm
from litellm import acompletion
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


class LLMProxy(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        debug: bool = False,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

        litellm.enable_json_schema_validation = True
        if debug:
            litellm._turn_on_debug()  # type: ignore

    async def completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    ) -> dict | str:

        response = await acompletion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version,
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
        response = await acompletion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version,
            messages=messages,
            stream=True,
        )

        return response
