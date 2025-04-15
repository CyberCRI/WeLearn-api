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

    @abstractmethod
    def init_client(self):
        """
        Initializes the chat client.
        """
        pass

    @abstractmethod
    async def chat(
        self, type: RESPONSE_TYPE, model: str, messages: List[Dict[str, str]]
    ):
        """
        Sends a chat message and returns the response.

        Args:
            type (RESPONSE_TYPE): The response type.
            model (str): The model to be used for the chat service.
            messages (list): A list of messages to be sent.

        Returns:
            The response from the chat service.
        """
        pass

    @abstractmethod
    async def chat_schema(
        self, model: str, messages: List[Dict[str, str]], response_format: BaseModel
    ):
        """
        Sends a chat message and returns the response.

        Args:
            type (RESPONSE_TYPE): The response type.
            model (str): The model to be used for the chat service.
            messages (list): A list of messages to be sent.

        Returns:
            The response from the chat service.
        """
        pass

    @abstractmethod
    async def streamed_chat(self, messages: List[Dict[str, str]]):
        """
        Sends a chat message and returns the streamed response.

        Args:
            messages (list): A list of messages to be sent.

        Returns:
            The streamed response from the chat service.
        """
        pass

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
        return self.get_message_content(completion)

    async def _detect_lang_with_llm(self, query: str) -> Dict[str, str]:
        """
        Detects language using LLM.

        Args:
            query (str): The user query.

        Returns:
            dict: The detected language.
        """
        detected_lang = await self.chat(
            type="text",
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompts.CHECK_LANGUAGE_PROMPT.format(query=query),
                }
            ],
        )
        res = self.get_message_content(message=detected_lang).strip()

        try:
            detected = json.loads(res)

            if detected["ISO_CODE"] not in ["en", "fr"]:
                raise LanguageNotSupportedError()

            return detected
        except json.JSONDecodeError:
            logger.error("api_error=invalid_json, response=%s", res)
            raise ValueError("Invalid response from model")

    async def _detect_past_message_ref(self, query: str, history: List[Dict[str, str]]):
        """
        Detects reference to past messages.

        Args:
            query (str): The user query.
            history (list): The chat history.

        Returns:
            The detected reference to past messages.
        """
        completion = await self.chat(
            type="text",
            model=self.model,
            messages=[
                self.system_prompts["past_message"],
                *history[:-2],
                {
                    "role": "user",
                    "content": prompts.PAST_MESSAGE_REF.format(query=query),
                },
            ],
        )
        completion_content = self.get_message_content(message=completion).strip()

        try:
            jsn = json.loads(completion_content)

            if "REF_TO_PAST" not in jsn or jsn["REF_TO_PAST"] not in [True, False]:
                raise ValueError("Invalid response from model")

            return jsn
        except json.JSONDecodeError:
            logger.error("api_error=invalid_json, response=%s", completion_content)
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

        ref_to_past = await self._detect_past_message_ref(query, history)
        if ref_to_past["REF_TO_PAST"]:
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

        reformulated_query = await self.chat_schema(
            messages=messages,
            model=self.model,
            response_format=ReformulatedQueryResponse,
        )

        if not isinstance(reformulated_query, ReformulatedQueryResponse):
            raise ValueError(
                {
                    "message": "Invalid response from model",
                    "response": reformulated_query,
                }
            )

        return reformulated_query

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

        res = await self.chat(
            type="text",
            model=self.model,
            messages=[
                *history[::-2][:2],
                {
                    "role": "user",
                    "content": prompts.GENERATE_NEW_QUESTIONS + query,
                },
            ],
        )

        res_content = self.get_message_content(res)

        res_list: List[str] = [
            res.strip() for res in res_content.split("%%") if len(res.strip()) > 0
        ]
        return {"NEW_QUESTIONS": res_list}

    async def rephrase_message(
        self,
        docs: List[Document],
        message: str,
        history: List[Dict[str, str]],
        subject: str,
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
            res = await self.streamed_chat(messages)
            return self.get_stream_chunks(res)

        res = await self.chat(
            type="text",
            model=self.model,
            messages=messages,
        )
        return self.get_message_content(res)

    async def chat_message(
        self,
        query: str,
        history: List[Dict[str, str]],
        docs: List[Document],
        subject: str,
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
            res = await self.streamed_chat(messages)
            return self.get_stream_chunks(res)

        res = await self.chat(
            type="text",
            model=self.model,
            messages=messages,
        )
        return self.get_message_content(res)


class Open_Chat(AbstractChat):
    """
    A concrete implementation of AbstractChat for OpenAI.
    """

    def init_client(self):
        if not self.API_BASE or not self.API_VERSION:
            raise ValueError("API_BASE or API_VERSION not provided")

        try:
            self.chat_client = AsyncAzureOpenAI(
                api_key=self.API_KEY,
                azure_endpoint=self.API_BASE,
                api_version=self.API_VERSION,
            )
        except Exception as e:
            logger.error("ai_generator=openai error=%s", e)

    async def streamed_chat(self, messages: List[Dict[str, str]]):
        res = await self.chat_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.2,
            max_tokens=1000,
        )

        return res

    async def chat_json(self, messages: List[Dict[str, str]]):
        completion = await self.chat(
            model=self.model,
            type="json_object",
            messages=messages,
        )
        res = self.get_message_content(message=completion)

        return json.loads(res)

    async def chat(
        self, type: RESPONSE_TYPE, model: str, messages: List[Dict[str, str]]
    ):
        try:
            completion = await self.chat_client.chat.completions.create(
                model=model,
                response_format={"type": type},
                messages=messages,
                stream=False,
                temperature=0.2,
            )

            return completion
        except Exception as e:
            logger.error("ai_generator=openai error=%s", e)
            raise e

    async def chat_schema(
        self, model: str, messages: List[Dict[str, str]], response_format
    ):
        try:
            completion = await self.chat_client.beta.chat.completions.parse(
                model=self.model or model,
                messages=messages,
                temperature=0.2,
                response_format=response_format,
            )

            return completion.choices[0].message.parsed
        except Exception as e:
            logger.error("ai_generator=openai error=%s", e)
            raise e

    async def _detect_lang_with_llm(self, query: str) -> Dict[str, str]:
        try:
            detected_lang = await self.chat_json(
                messages=[
                    {
                        "role": "user",
                        "content": prompts.CHECK_LANGUAGE_PROMPT.format(query=query),
                    }
                ],
            )
            return detected_lang

        except Exception as e:
            raise e

    async def _detect_past_message_ref(self, query: str, history: List[Dict[str, str]]):
        try:
            completion = await self.chat_json(
                messages=[
                    self.system_prompts["past_message"],
                    *history[:-2],
                    {
                        "role": "user",
                        "content": prompts.PAST_MESSAGE_REF.format(query=query),
                    },
                ],
            )

            return completion
        except Exception as e:
            raise e


class Mistral_Chat(AbstractChat):
    """
    A concrete implementation of AbstractChat for Mistral.
    """

    def init_client(self):
        logger.debug("ai_generator=mistral")
        self.chat_client = Mistral(api_key=self.API_KEY)

    async def streamed_chat(self, messages: List[Dict[str, str]]):
        resp = self.chat_client.chat.stream_async(model=self.model, messages=messages)

        return resp.data

    async def chat(
        self, type: RESPONSE_TYPE, model: str, messages: List[Dict[str, str]]
    ):
        resp = await self.chat_client.chat.complete_async(
            model=model,
            messages=messages,
            response_format={"type": type},
        )

        return resp.data


class Azure_Chat(AbstractChat):
    """
    A concrete implementation of AbstractChat for Azure.
    """

    def init_client(self):
        logger.debug("ai_generator=azure")
        if not self.API_BASE:
            raise ValueError("API_BASE not provided")

        try:
            self.chat_client = ChatCompletionsClient(
                credential=AzureKeyCredential(self.API_KEY),
                endpoint=self.API_BASE,
            )
        except Exception as e:
            logger.error("ai_generator=azure error=%s", e)
            raise e

    async def streamed_chat(self, messages: List[Dict[str, str]]):
        resp = self.chat_client.complete(
            messages=messages,
            temperature=0.2,
            top_p=1,
            max_tokens=800,
            stream=True,
        )
        return resp

    async def chat(
        self, type: RESPONSE_TYPE, model: str, messages: List[Dict[str, str]]
    ):
        logger.info("chat_azure, model=%s type=%s", model, type)
        resp = self.chat_client.complete(
            messages=messages,
            temperature=0.2,
            top_p=1,
            max_tokens=800,
            stream=False,
        )
        return resp


class ChatFactory:
    """
    A factory class to create instances of chat services.
    """

    def __init__(self):
        pass

    def create_chat(
        self, chat_type: Literal["openai", "mistral", "azure"], model: str | None = None
    ):
        """
        Creates an instance of a chat service based on the specified type and model.

        Args:
            chat_type (Literal["openai", "mistral", "azure"]): The type of chat service.
            model (str | None): The model to be used for the chat service.

        Returns:
            An instance of the specified chat service.
        """
        settings = get_settings()

        chat_classes = {
            "openai": Open_Chat,
            "mistral": Mistral_Chat,
            "azure": Azure_Chat,
        }

        if chat_type not in chat_classes:
            raise ValueError(f"Unsupported chat type: {chat_type}")

        if chat_type == "openai":
            openai_models = {
                "gpt-4o-mini": (
                    settings.AZURE_API_KEY,
                    settings.AZURE_API_BASE,
                    settings.AZURE_API_VERSION,
                ),
                "gpt-4o": (
                    settings.AZURE_GPT_4O_API_KEY,
                    settings.AZURE_GPT_4O_API_BASE,
                    "2025-01-01-preview",
                ),
            }
            if model:
                key, base, version = openai_models.get(model, (None, None, None))
                if not key or not base or not version:
                    raise ValueError(f"Unsupported model: {model}")
                return chat_classes[chat_type](
                    API_KEY=key, API_BASE=base, API_VERSION=version, model=model
                )

            return chat_classes[chat_type](
                API_VERSION=settings.AZURE_API_VERSION,
                API_KEY=settings.AZURE_API_KEY,
                API_BASE=settings.AZURE_API_BASE,
                model="gpt-4o-mini",
            )
        elif chat_type == "mistral":
            return chat_classes[chat_type](
                API_KEY=settings.MISTRAL_API_KEY,
                model="open-mistral-nemo",
            )
        elif chat_type == "azure":
            azure_models = {
                "Meta-Llama-3.1-8B-Instruct": (
                    settings.AZURE_LLAMA_31_8B_API_KEY,
                    settings.AZURE_LLAMA_31_8B_API_BASE,
                ),
                "Meta-Llama-3.1-70B-Instruct": (
                    settings.AZURE_LLAMA_31_70B_API_KEY,
                    settings.AZURE_LLAMA_31_70B_API_BASE,
                ),
                "Meta-Llama-3.1-405B-Instruct": (
                    settings.AZURE_LLAMA_31_405B_API_KEY,
                    settings.AZURE_LLAMA_31_405B_API_BASE,
                ),
            }

            if model not in azure_models:
                raise ValueError(f"Unsupported model: {model}")

            API_KEY, API_BASE = azure_models[model]
            return chat_classes[chat_type](
                API_KEY=API_KEY,
                API_BASE=API_BASE,
                model=model,
            )
