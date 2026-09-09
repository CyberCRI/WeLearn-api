from typing import Any, AsyncIterable, Optional, Type, Union, cast

from langchain_core.language_models import BaseChatModel
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel

from src.app.shared.infra.chat_models import build_chat_model
from src.app.utils.decorators import log_time_and_error
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


class LLMProxy:
    """Non-agent chat client, backed by LangChain chat models.

    Exposes a small, stable interface (completion / completion_stream) on top
    of the shared chat_models.build_chat_model factory, so callers don't need
    to know which provider is configured.
    """

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
        self.is_azure_model = is_azure_model

        logger.debug(
            "Initializing %s chat model", "Azure" if is_azure_model else "Mistral"
        )
        self.client: BaseChatModel = build_chat_model(
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            is_azure_model=is_azure_model,
            temperature=0.8,
            top_p=0.1,
            max_tokens=2048,
        )

    async def close_client(self):
        # LangChain chat models manage their own HTTP client lifecycle.
        return None

    def _get_langsmith_provider(self) -> str:
        return "azure" if self.is_azure_model else "mistral"

    def _record_langsmith_usage(self, message: Any) -> None:
        run_tree = get_current_run_tree()
        if run_tree is None:
            return

        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            run_tree.set(usage_metadata=cast(Any, dict(usage_metadata)))

        metadata = dict(getattr(run_tree, "metadata", {}) or {})
        metadata_to_add: dict[str, Any] = {}
        if "ls_provider" not in metadata:
            metadata_to_add["ls_provider"] = self._get_langsmith_provider()
        if "ls_model_name" not in metadata:
            metadata_to_add["ls_model_name"] = self.model
        if metadata_to_add:
            run_tree.add_metadata(metadata_to_add)

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return "" if content is None else str(content)

    @log_time_and_error
    async def completion(
        self,
        messages: list,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        trace_context: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = 2048,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        message = await self.client.ainvoke(messages, **kwargs)
        self._record_langsmith_usage(message)

        return self._stringify_content(message.content)

    async def completion_stream(self, messages: list) -> AsyncIterable[Any]:
        """Streams a completion, yielding LangChain AIMessageChunk objects."""
        return self.client.astream(messages)
