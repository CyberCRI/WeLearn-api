"""Centralized construction of LangChain chat models.

Single place that knows how to build a provider-specific LangChain
BaseChatModel (Mistral or Azure), reused by the non-agent LLMProxy, the RAG
agent, and the tutor chat agents so provider selection/defaults live in one
place instead of being duplicated at each call site.
"""

from typing import Any

from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel  # type: ignore
from langchain_core.language_models import BaseChatModel
from langchain_mistralai import ChatMistralAI  # type: ignore


def build_mistral_chat_model(
    model: str,
    api_key: str,
    **overrides: Any,
) -> ChatMistralAI:
    params: dict[str, Any] = {"model_name": model, "mistral_api_key": api_key}
    params.update(overrides)
    return ChatMistralAI(**params)


def build_azure_chat_model(
    model: str,
    api_key: str,
    api_base: str,
    api_version: str,
    **overrides: Any,
) -> AzureAIOpenAIApiChatModel:
    params: dict[str, Any] = {
        "model": model,
        "credential": api_key,
        "endpoint": api_base,
        "api_version": api_version,
        "use_responses_api": False,
    }
    params.update(overrides)
    return AzureAIOpenAIApiChatModel(**params)


def build_chat_model(
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
    is_azure_model: bool = False,
    **overrides: Any,
) -> BaseChatModel:
    """Builds a LangChain chat model for the given provider (Mistral by default, Azure when is_azure_model=True)."""
    if is_azure_model:
        if api_key is None or api_base is None or api_version is None:
            raise ValueError(
                "For Azure models, api_key, api_base, and api_version must be provided."
            )
        return build_azure_chat_model(model, api_key, api_base, api_version, **overrides)

    if api_key is None:
        raise ValueError("For Mistral models, api_key must be provided.")
    return build_mistral_chat_model(model, api_key, **overrides)
