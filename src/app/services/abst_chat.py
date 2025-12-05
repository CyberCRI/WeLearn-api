"""
abst_chat.py

This module defines the abstract base class for chat services and its concrete implementations for different chat providers.
It also includes a factory class to create instances of these chat services based on the specified type.

Classes:
    AbstractChat: An abstract base class for chat services.
    Open_Chat: A concrete implementation of AbstractChat for OpenAI.
    Mistral_Chat: A concrete implementation of AbstractChat for Mistral.
    Azure_Chat: A concrete implementation of AbstractChat for Azure.
    ChatFactory: A factory class to create instances of chat services.

Functions:
    create_chat: Creates an instance of a chat service based on the specified type and model.
"""

import json
import uuid
from abc import ABC
from typing import AsyncIterable, Dict, List, Optional

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel  # type: ignore
from langchain_core.runnables import RunnableConfig  # type: ignore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore
from langgraph.prebuilt import create_react_agent  # type: ignore

from src.app.api.dependencies import get_settings
from src.app.models.chat import ReformulatedQueryResponse
from src.app.models.documents import Document
from src.app.services import prompts
from src.app.services.agent import (
    get_resources_about_sustainability,
    trim_conversation_history,
)
from src.app.services.exceptions import LanguageNotSupportedError
from src.app.services.helpers import (
    detect_language_from_entry,
    extract_json_from_response,
    stringify_docs_content,
)
from src.app.services.llm_proxy import LLMProxy
from src.app.utils.decorators import log_time_and_error
from src.app.utils.logger import log_environmental_impacts
from src.app.utils.logger import logger as utils_logger

# from ecologits import EcoLogits  # type: ignore


logger = utils_logger(__name__)
# EcoLogits.init(["openai", "mistralai"])


class AbstractChat(ABC):
    """
    An abstract base class for chat services.

    Attributes:
        model (str): The model to be used for the chat service.
        API_KEY (str): The API key for the chat service.
        API_BASE (str): The API base URL for the chat service.
        API_VERSION (str): The API version for the chat service.
        system_prompts (dict): A dictionary of system prompts for the chat service.
    """

    def __init__(
        self,
        model: str,
        API_KEY: str,
        API_BASE: Optional[str] = None,
        API_VERSION: Optional[str] = None,
        is_azure_model: bool = False,
    ):
        self.chat_client = LLMProxy(
            model=model,
            api_key=API_KEY,
            api_base=API_BASE,
            api_version=API_VERSION,
            is_azure_model=is_azure_model,
        )
        self.model = model
        self.API_KEY = API_KEY
        self.API_BASE = API_BASE
        self.API_VERSION = API_VERSION
        self.system_prompts = {
            "reformulate": {
                "role": "system",
                "content": prompts.SYSTEM_PROMPT_STANDALONE_QUESTION,
            },
            "past_message": {
                "role": "system",
                "content": prompts.SYSTEM_PAST_MESSAGE_REF,
            },
        }

    @log_time_and_error
    async def _detect_language(self, query: str) -> Dict[str, str]:
        """
        Detects the language of the query.

        Args:
            query (str): The user query.

        Returns:
            dict: The detected language.
        """
        try:
            lang = detect_language_from_entry(query)
            return {"ISO_CODE": lang}
        except LanguageNotSupportedError:
            logger.info(
                "api_error=LANG_NOT_SUPPORTED using llm to check check_lang=%s", query
            )
            lang = await self._detect_lang_with_llm(query)
            return lang

    @log_time_and_error
    async def _detect_lang_with_llm(self, query: str) -> Dict[str, str]:
        """
        Detects language using LLM.

        Args:
            query (str): The user query.

        Returns:
            dict: The detected language.
        """
        detected_lang = await self.chat_client.completion(
            messages=[
                {
                    "role": "user",
                    "content": prompts.CHECK_LANGUAGE_PROMPT.format(query=query),
                }
            ],
            response_format={
                "type": "json_object",
            },
        )

        if isinstance(detected_lang, str):
            jsn = extract_json_from_response(detected_lang)
        elif isinstance(detected_lang, dict):
            jsn = detected_lang
        else:
            raise ValueError("Invalid response from model")

        return jsn

    @log_time_and_error
    async def _detect_past_message_ref(
        self, query: str, history: List[Dict[str, str]]
    ) -> dict | None:
        """
        Detects reference to past messages.

        Args:
            query (str): The user query.
            history (list): The chat history.

        Returns:
            The detected reference to past messages.
        """
        completion = await self.chat_client.completion(
            messages=[
                self.system_prompts["past_message"],
                *history[:-2],
                {
                    "role": "user",
                    "content": prompts.PAST_MESSAGE_REF.format(query=query),
                },
            ],
            response_format={
                "type": "json_object",
            },
        )

        try:
            jsn = {}
            if isinstance(completion, str):
                jsn = extract_json_from_response(completion)
            elif isinstance(completion, dict):
                jsn = completion
            else:
                raise ValueError("Invalid response from model")

            if "REF_TO_PAST" not in jsn or jsn["REF_TO_PAST"] not in [True, False]:
                raise ValueError("Invalid response from model")

            return jsn
        except json.JSONDecodeError:
            logger.error("api_error=invalid_json, response=%s", completion)

    async def get_stream_chunks(self, stream) -> AsyncIterable[str]:
        """
        Gets content from streamed response.

        Args:
            stream (Generator[dict]): The streamed chat response.

        Yields:
            str: The openai stream content.
        """
        try:
            async for chunk in stream:
                for part in self._extract_stream_chunk(chunk):
                    yield part
        except Exception:
            try:
                for chunk in stream:
                    for part in self._extract_stream_chunk(chunk):
                        yield part
            except Exception as e:
                logger.error("get_stream_chunks api_error=%s", e)
                raise e

    def _extract_stream_chunk(self, chunk):
        choices = getattr(chunk, "choices", None)
        if choices:
            delta_content = getattr(choices[0].delta, "content", None)
            if delta_content:
                yield delta_content
            finish_reason = getattr(choices[0], "finish_reason", None)
            if finish_reason:
                log_environmental_impacts(chunk, logger)

    @log_time_and_error
    async def reformulate_user_query(self, query: str, history: List[Dict[str, str]]):
        """
        Reformulates user query if it's about a new subject.

        Args:
            query (str): The user query.
            history (list): The chat history.

        Returns:
            dict: The reformulated query or None.
        """

        ref_to_past: dict | None = await self._detect_past_message_ref(query, history)
        if ref_to_past and ref_to_past["REF_TO_PAST"]:
            return ReformulatedQueryResponse(
                STANDALONE_QUESTION=None,
                USER_LANGUAGE=None,
                QUERY_STATUS="REF_TO_PAST" if len(history) >= 1 else "INVALID",
            )

        ref_query = ReformulatedQueryResponse(
            STANDALONE_QUESTION=query,
            USER_LANGUAGE="",
            QUERY_STATUS="VALID",
        )

        if not isinstance(ref_query, ReformulatedQueryResponse):
            raise ValueError(
                {
                    "message": "Invalid response from model",
                    "response": ref_query,
                }
            )

        return ref_query

    @log_time_and_error
    async def get_new_questions(
        self, query: str, history: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """
        Gets new questions from chat model based on history.

        Args:
            query (str): The user query.
            history (list): The chat history.

        Returns:
            dict: The new questions.
        """
        await self._detect_language(query)

        res = await self.chat_client.completion(
            messages=[
                *history[::-2][:2],
                {
                    "role": "user",
                    "content": prompts.GENERATE_NEW_QUESTIONS + query,
                },
            ],
        )

        assert isinstance(res, str)

        res_list: List[str] = [r.strip() for r in res.split("%%") if len(r.strip()) > 0]
        return {"NEW_QUESTIONS": res_list}

    @log_time_and_error
    async def rephrase_message(
        self,
        docs: List[Document],
        message: str,
        history: List[Dict[str, str]],
        subject: str | None = None,
        streamed_ans: bool = False,
    ):
        """
        Rephrases last assistant's response based on subject and history.

        Args:
            docs (list): List of documents.
            message (str): Last assistant response.
            history (list): Chat history.
            subject (str): Subject.
            streamed_ans (bool): Whether to stream the answer.

        Returns:
            str: The rephrased message content.
        """
        stringified_docs = stringify_docs_content(docs)
        messages = [
            {
                "role": "system",
                "content": prompts.SYSTEM_PROMPT.format(cursus=subject or "General"),
            },
            *history[-5:-1],
            {
                "role": "user",
                "content": prompts.REPHRASE.format(
                    prompt=message, documents=stringified_docs
                ),
            },
        ]

        if streamed_ans:
            res = self.chat_client.completion_stream(messages)
            return self.get_stream_chunks(res)

        res = await self.chat_client.completion(
            messages=messages,
        )
        return res

    @log_time_and_error
    async def chat_message(
        self,
        query: str,
        history: List[Dict[str, str]],
        docs: List[Document],
        subject: str | None = None,
        streamed_ans: bool = False,
    ):
        """
        Sends a chat message.

        Args:
            query (str): The user query.
            history (list): The chat history.
            docs (list): List of documents.
            subject (str): Subject.
            streamed_ans (bool): Whether to stream the answer.

        Returns:
            str: The chat message content.
        """

        # ISO_CODE = {"ISO_CODE": "en"}

        ISO_CODE = await self._detect_language(query)

        messages = [
            {
                "role": "system",
                "content": prompts.SYSTEM_PROMPT.format(cursus=subject or "General"),
            },
            *history,
            {
                "role": "user",
                "content": prompts.SOURCED_ANSWER.format(
                    documents=stringify_docs_content(docs),
                    query=query,
                    ISO_CODE=ISO_CODE,
                ),
            },
        ]
        if streamed_ans:
            res = await self.chat_client.completion_stream(messages)
            return self.get_stream_chunks(res)

        res = await self.chat_client.completion(
            messages=messages,
        )
        return res

    async def agent_message(
        self,
        query: str,
        memory: AsyncPostgresSaver | None = None,
        thread_id: Optional[uuid.UUID] = None,
        corpora: Optional[tuple[str, ...]] = None,
        sdg_filter: Optional[List[int]] = None,
    ):
        """
        Sends a chat message handled by an agent.

        Args:
            query (str): The user query.
            memory (AsyncPostgresSaver | None): The memory to use for the agent.
            thread_id (uuid.UUID): The thread ID.
            corpora (tuple[str, ...] | None): The corpora to search resources.
            sdg_filter (list[int] | None): The SDG filters to apply to the search.

        Returns:
            str: The chat message content.
        """
        settings = get_settings()
        agent_model = AzureAIChatCompletionsModel(
            endpoint=settings.AZURE_MISTRAL_API_BASE,
            credential=settings.AZURE_MISTRAL_API_KEY,
            model="Mistral-Large-2411",
        )

        agent_executor = create_react_agent(
            model=agent_model,
            tools=[
                get_resources_about_sustainability,
            ],
            checkpointer=memory,
            prompt=prompts.AGENT_SYSTEM_PROMPT,
            pre_model_hook=trim_conversation_history,
        )

        config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "corpora": corpora,
                "sdg_filter": sdg_filter,
            }
        )

        messages = [
            {
                "role": "user",
                "content": query,
            },
        ]

        res = await agent_executor.ainvoke(input={"messages": messages}, config=config)
        return res
