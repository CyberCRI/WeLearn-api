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
from abc import ABC, abstractmethod
from typing import AsyncIterable, Dict, List, Literal, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# from ecologits import EcoLogits  # type: ignore
from mistralai import Mistral
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from src.app.api.dependencies import get_settings
from src.app.models.chat import RESPONSE_TYPE, ReformulatedQueryResponse
from src.app.models.documents import Document
from src.app.services import prompts
from src.app.services.exceptions import LanguageNotSupportedError
from src.app.services.helpers import detect_language_from_entry, stringify_docs_content
from src.app.services.llm_proxy import LLMProxy
from src.app.utils.logger import log_environmental_impacts
from src.app.utils.logger import logger as utils_logger

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
    ):
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

    def init_client(self):
        self.chat_client = LLMProxy(
                model=self.model,
                api_key=self.API_KEY,
                api_base=self.API_BASE,
                api_version=self.API_VERSION,
            )


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

    async def flex_chat(self, messages):
        completion = await self.chat(model=self.model, type="text", messages=messages)
        return completion

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
                }
        )

        try:
            assert isinstance(detected_lang, dict)

            if detected_lang["ISO_CODE"] not in ["en", "fr"]:
                raise LanguageNotSupportedError()

            return detected_lang
        except json.JSONDecodeError:
            logger.error("api_error=invalid_json, response=%s", detected_lang)
            raise ValueError("Invalid response from model")

    async def _detect_past_message_ref(self, query: str, history: List[Dict[str, str]]) -> dict | None:
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
            }
        )

        try:
            assert isinstance(completion, dict)
            jsn = completion

            if "REF_TO_PAST" not in jsn or jsn["REF_TO_PAST"] not in [True, False]:
                raise ValueError("Invalid response from model")

            return jsn
        except json.JSONDecodeError:
            
            logger.error("api_error=invalid_json, response=%s", completion)
        except AssertionError:
            logger.error("api_error=assertion_error, response=%s", completion)
            raise ValueError("Invalid response from model")

    def get_message_content(self, message) -> str:
        """
        Gets content from response.

        Args:
            message (dict): The chat response.

        Returns:
            str: The chat client response content.
        """
        log_environmental_impacts(message, logger)
        check = message.choices[0].message.content
        return check

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
                choices = chunk.choices
                if choices:
                    if choices[0].delta.content:
                        yield choices[0].delta.content
                    if choices[0].finish_reason:
                        log_environmental_impacts(chunk, logger)

                continue
        except Exception as e:
            logger.error("get_stream_chunks api_error=%s", e)
            raise e

    async def reformulate_user_query(self, query: str, history: List[Dict[str, str]]):
        """
        Reformulates user query if it's about a new subject.

        Args:
            query (str): The user query.
            history (list): The chat history.

        Returns:
            dict: The reformulated query or None.
        """
        # lang = await self._detect_language(query)

        ref_to_past: dict | None = await self._detect_past_message_ref(query, history)
        if ref_to_past and ref_to_past["REF_TO_PAST"]:
            return ReformulatedQueryResponse(
                STANDALONE_QUESTION_EN=None,
                STANDALONE_QUESTION_FR=None,
                USER_LANGUAGE=None,
                QUERY_STATUS="REF_TO_PAST" if len(history) >= 1 else "INVALID",
            )

        messages = [
            {
                **self.system_prompts["reformulate"],
                "content": self.system_prompts["reformulate"]["content"].format(
                    iso="en"
                ),
            },
            *history[:-5],
            {
                "role": "user",
                "content": prompts.STANDALONE_QUESTION + query,
            },
        ]

        reformulated_query = await self.chat_client.completion(
            messages=messages,
            response_format=ReformulatedQueryResponse,
        )

        assert isinstance(reformulated_query, dict)

        ref_query = ReformulatedQueryResponse(**reformulated_query)

        if not isinstance(ref_query, ReformulatedQueryResponse):
            raise ValueError(
                {
                    "message": "Invalid response from model",
                    "response": reformulated_query,
                }
            )

        return ref_query 

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

        res_list: List[str] = [
            r.strip() for r in res.split("%%") if len(r.strip()) > 0
        ]
        return {"NEW_QUESTIONS": res_list}

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

    async def chat_message(
        self,
        query: str,
        history: List[Dict[str, str]],
        docs: List[Document],
        subject: str | None = None,
        streamed_ans: bool = False,
        should_check_lang: bool = True,
    ):
        """
        Sends a chat message.

        Args:
            query (str): The user query.
            history (list): The chat history.
            docs (list): List of documents.
            subject (str): Subject.
            streamed_ans (bool): Whether to stream the answer.
            should_check_lang (bool): Whether to check the language.

        Returns:
            str: The chat message content.
        """
        ISO_CODE = {"ISO_CODE": "en"}
        if should_check_lang:
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
            res = self.chat_client.completion_stream(messages)
            return self.get_stream_chunks(res)

        res = await self.chat_client.completion(
            messages=messages,
        )
        return self.get_message_content(res)

