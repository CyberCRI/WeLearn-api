import json
from abc import ABC
from typing import Type, Optional, Union
from pydantic import BaseModel

import litellm
from litellm import acompletion, supports_response_schema
from litellm.types.utils import ModelResponse
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

litellm.enable_json_schema_validation = True
litellm._turn_on_debug()


class LLMProxy(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

    async def schema_completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    ) :

        try:
            assert supports_response_schema(model=self.model)
        except AssertionError:
            response_format = None
        finally:
            response = await acompletion(
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
               api_version=self.api_version,
                messages=messages,
                response_format=response_format,
            )

        assert isinstance(response, ModelResponse)

        return json.loads(response["choices"][0]["message"]["content"])
