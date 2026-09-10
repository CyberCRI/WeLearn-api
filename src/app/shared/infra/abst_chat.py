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

import uuid
from abc import ABC
from typing import Any, AsyncIterable, Dict, List, Optional, TypedDict, cast

from fastapi import BackgroundTasks, Depends, Request
from langchain.agents import create_agent  # type: ignore
from langchain.agents.middleware import (  # type: ignore
    ClearToolUsesEdit,
    SummarizationMiddleware,
)
from langchain.agents.middleware.types import AgentMiddleware  # type: ignore
from langchain.messages import HumanMessage  # type: ignore
from langchain_core.messages import BaseMessage, RemoveMessage  # type: ignore
from langchain_core.messages.utils import count_tokens_approximately  # type: ignore
from langchain_core.runnables import RunnableConfig  # type: ignore
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # type: ignore
from langgraph.graph.message import REMOVE_ALL_MESSAGES  # type: ignore
from langsmith import traceable

from src.app.models.documents import Document
from src.app.search.services.search import SearchService
from src.app.services import prompts
from src.app.services.agent import get_resources_about_sustainability
from src.app.services.helpers import (
    detect_language_from_entry,
    extract_json_from_response,
    latest_tool_docs,
    stringify_docs_content,
)
from src.app.shared.domain.exceptions import LanguageNotSupportedError
from src.app.shared.infra.tracing import TRACE_RUN_TYPE_LLM, TraceComponent, TraceName
from src.app.shared.utils.dependencies import get_settings
from src.app.utils.decorators import log_time_and_error
from src.app.utils.logger import log_environmental_impacts
from src.app.utils.logger import logger as utils_logger

# from ecologits import EcoLogits  # type: ignore


logger = utils_logger(__name__)
# EcoLogits.init(["openai", "mistralai"])


class _PersistClearedToolUses(AgentMiddleware):
    """Applies a `ClearToolUsesEdit` permanently to the checkpointed state.

    `ContextEditingMiddleware` (the built-in `wrap_model_call` equivalent) only
    edits a deepcopy for one model call, so old tool results stay in persisted
    state forever and `SummarizationMiddleware` still has to wade through them.
    Running this as a `before_model` hook instead persists the clearing, and
    lets it run before summarization (both are `before_model`, so list order
    is honored) instead of being architecturally stuck after it.
    """

    def __init__(self, edit: ClearToolUsesEdit) -> None:
        super().__init__()
        self._edit = edit

    def before_model(self, state, runtime):  # noqa: ANN001, ARG002
        messages = [m.model_copy() for m in state["messages"]]
        self._edit.apply(messages, count_tokens=count_tokens_approximately)
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]}

    async def abefore_model(self, state, runtime):  # noqa: ANN001, ARG002
        return self.before_model(state, runtime)


class _ReinforceHardConstraints(AgentMiddleware):
    """Re-states the full system prompt (condensed) on the latest turn, every
    model call — long conversations dilute the system prompt's influence;
    recency counters that by keeping the rules close to generation time.
    """

    def wrap_model_call(self, request, handler):  # noqa: ANN001
        return handler(request.override(messages=self._with_reminder(request.messages)))

    async def awrap_model_call(self, request, handler):  # noqa: ANN001
        return await handler(
            request.override(messages=self._with_reminder(request.messages))
        )

    @staticmethod
    def _with_reminder(messages):  # noqa: ANN001
        messages = list(messages)
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1] = messages[-1].model_copy(
                update={
                    "content": messages[-1].content
                    + "\n\n"
                    + prompts.AGENT_REMINDER_PROMPT
                }
            )
        return messages


class _AgentInputState(TypedDict):
    messages: list[BaseMessage]


class AbstractChat(ABC):
    """
    An abstract base class for chat services.

    Attributes:
        model (str): The model to be used for the chat service.
        API_KEY (str): The API key for the chat service.
        API_BASE (str): The API base URL for the chat service.
        API_VERSION (str): The API version for the chat service.
    """

    def __init__(
        self,
        client,
    ):
        self.agent_executor = None
        self.chat_client = client

    def _build_non_agent_trace_context(
        self,
        operation: str,
        **extra: Any,
    ) -> dict[str, Any]:
        settings = get_settings()
        trace_context: dict[str, Any] = {
            "component": TraceComponent.CHAT_NON_AGENT.value,
            "operation": operation,
            "environment": settings.ENV,
            "model": getattr(self.chat_client, "model", None),
        }
        trace_context.update(extra)
        return trace_context

    @log_time_and_error
    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.JSON_FORMATTER_AGENT.value,
    )
    async def json_formatter_agent(self, unformatted_input, expected_output):
        output = await self.chat_client.completion(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a json formatter agent that recieves from the user a content wrongly formatted and an expected json schema and replies with the input formatted in the correct way. Here is the expected schema :{expected_output}",
                },
                {
                    "role": "user",
                    "content": f"Reformat my input so it respects the expected output. Input : {unformatted_input}",
                },
            ],
            response_format={
                "type": "json_object",
            },
        )

        json = extract_json_from_response(output)
        return json

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
    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.DETECT_LANG_WITH_LLM.value,
    )
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
            parsed = extract_json_from_response(detected_lang)
        elif isinstance(detected_lang, dict):
            parsed = detected_lang
        else:
            raise ValueError("Invalid response from model")

        if not isinstance(parsed, dict):
            raise ValueError("Invalid response from model")

        jsn: Dict[str, str] = {}
        for key, value in parsed.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Invalid response from model")
            jsn[key] = value

        return jsn

    async def get_stream_chunks(self, stream) -> AsyncIterable[str]:
        """
        Gets content from streamed response.

        Args:
            stream (Generator[dict]): The streamed chat response.

        Yields:
            str: The stream content.
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
                raise

    def _extract_stream_chunk(self, chunk):
        choices = getattr(chunk, "choices", None)
        if choices:
            delta_content = getattr(choices[0].delta, "content", None)
            if delta_content:
                yield delta_content
            finish_reason = getattr(choices[0], "finish_reason", None)
            if finish_reason:
                log_environmental_impacts(chunk, logger)

    async def get_agent_chunks(self, stream) -> AsyncIterable[dict[str, Any]]:
        """
        Gets content from streamed response of an agent.

        Args:
            stream (Generator[dict]): The streamed agent response.

        Yields:
            dict[str, Any]: Normalized agent stream payload chunks.
        """
        try:
            async for chunk in stream:
                for part in self._extract_agent_chunk(chunk):
                    yield part
        except Exception as e:
            logger.error("get_agent_chunks api_error=%s", e)
            raise

    def _extract_agent_chunk(self, chunk):
        if isinstance(chunk, dict):
            if chunk.get("tools"):
                yield {
                    "status": "processing",
                    "step": "analyzing_resources",
                    "docs": chunk["tools"]["messages"][0].artifact,
                }

            elif chunk.get("model"):
                messages = chunk["model"].get("messages")
                if not messages:
                    logger.debug("agent_stream chunk_skipped=missing_or_empty_messages")
                    return None

                last_message = messages[-1]
                response_metadata = (
                    getattr(last_message, "response_metadata", None) or {}
                )
                finish_reason = response_metadata.get("finish_reason")

                if finish_reason == "tool_calls":
                    yield {
                        "status": "processing",
                        "step": "fetching_resources",
                    }
                else:
                    content = self._extract_text_from_message_content(
                        getattr(last_message, "content", "")
                    )
                    if content:
                        yield {
                            "status": "streaming",
                            "step": "generating_answer",
                            "content": content,
                        }
            return None

        if not (isinstance(chunk, tuple) and len(chunk) == 2):
            logger.debug(
                "agent_stream chunk_skipped=invalid_type type=%s",
                type(chunk).__name__,
            )
            return None

        message, metadata = chunk
        node_name = (
            metadata.get("langgraph_node") if isinstance(metadata, dict) else None
        )

        if node_name == "tools":
            docs = getattr(message, "artifact", None)
            payload: dict[str, Any] = {
                "status": "processing",
                "step": "analyzing_resources",
            }
            if docs is not None:
                payload["docs"] = docs
            yield payload
            return None

        response_metadata = getattr(message, "response_metadata", None) or {}
        finish_reason = response_metadata.get("finish_reason")

        if finish_reason == "tool_calls":
            yield {
                "status": "processing",
                "step": "fetching_resources",
            }
            return None

        content = self._extract_text_from_message_content(
            getattr(message, "content", "")
        )
        if content:
            yield {
                "status": "streaming",
                "step": "generating_answer",
                "content": content,
            }

    def _extract_text_from_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)

        return ""

    @log_time_and_error
    async def get_new_questions(
        self, query: str, history: List[Dict[str, str]], lang: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Gets new questions from chat model based on history.

        Args:
            query (str): The user query.
            history (list): The chat history.
            lang (str | None): UI language ISO code (used for empty-chat case).

        Returns:
            dict: The new questions.
        """
        if not history and lang:
            iso_code = lang
        elif history:
            combined = " ".join(m["content"] for m in history[-4:] if m.get("content"))
            detected = await self._detect_language(combined[:500])
            iso_code = detected.get("ISO_CODE", "en")
        else:
            detected = await self._detect_language(query)
            iso_code = detected.get("ISO_CODE", "en")

        res = await self.chat_client.completion(
            messages=[
                *history[-2:],
                {
                    "role": "user",
                    "content": prompts.GENERATE_NEW_QUESTIONS.format(language=iso_code)
                    + query,
                },
            ],
        )

        assert isinstance(res, str)

        res_list: List[str] = [r.strip() for r in res.split("%%") if len(r.strip()) > 0]
        return {"NEW_QUESTIONS": res_list}

    @log_time_and_error
    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.CHAT_MESSAGE.value,
    )
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

        res = await self.chat_client.completion(messages=messages)
        return res

    async def _create_agent(
        self,
        memory: AsyncPostgresSaver | None = None,
    ):
        if self.agent_executor:
            return self.agent_executor

        settings = get_settings()
        agent_model = ChatMistralAI(
            model_name=settings.MISTRAL_LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
        )

        self.agent_executor = create_agent(
            model=agent_model,
            tools=[
                get_resources_about_sustainability,
            ],
            middleware=[
                _PersistClearedToolUses(
                    ClearToolUsesEdit(
                        trigger=0,  # no threshold: always enforce `keep`, not a token-overflow safety net
                        clear_at_least=0,  # no minimum reclaim — clear every candidate outside `keep`
                        keep=1,  # only the most recent tool call's results stay visible to the model
                        placeholder=(
                            "[Earlier search results cleared. Call "
                            "get_resources_about_sustainability again if you need "
                            "to cite something from them.]"
                        ),
                    )
                ),
                SummarizationMiddleware(
                    model=agent_model,
                    trigger=("tokens", 32000),
                ),
                _ReinforceHardConstraints(),
            ],
            checkpointer=memory,
            system_prompt=prompts.AGENT_SYSTEM_PROMPT,
        )
        return self.agent_executor

    async def agent_message(
        self,
        query: str,
        memory: AsyncPostgresSaver | None = None,
        thread_id: Optional[uuid.UUID] = None,
        corpora: Optional[tuple[str, ...]] = None,
        sdg_filter: Optional[List[int]] = None,
        sp: SearchService | None = None,
        background_tasks: BackgroundTasks | None = None,
        streamed_ans: bool = False,
        trace_context: Optional[dict[str, Any]] = None,
    ):
        """
        Sends a chat message handled by an agent.

        Args:
            query (str): The user query.
            memory (AsyncPostgresSaver | None): The memory to use for the agent.
            thread_id (uuid.UUID): The thread ID.
            corpora (tuple[str, ...] | None): The corpora to search resources.
            sdg_filter (list[int] | None): The SDG filters to apply to the search.
            sp (SearchService | None): The search service to use for retrieving resources.
            background_tasks (BackgroundTasks | None): The background tasks to use for the search.
            streamed_ans (bool): Whether to stream the answer.

        Returns:
            str: The chat message content.
        """

        agent_executor = await self._create_agent(memory=memory)

        settings = get_settings()

        metadata: dict[str, Any] = {
            "component": TraceComponent.CHAT_AGENT.value,
            "environment": settings.ENV,
            "thread_id": str(thread_id) if thread_id else None,
            "corpora": list(corpora) if corpora else None,
            "sdg_filter": sdg_filter,
        }

        if trace_context:
            metadata.update(trace_context)

        tags = ["welearn", "chat", "agent"]
        endpoint = metadata.get("endpoint")
        if endpoint:
            tags.append(f"endpoint:{endpoint}")

        config = RunnableConfig(
            tags=tags,
            metadata=metadata,
            configurable={
                "thread_id": thread_id,
                "corpora": corpora,
                "sdg_filter": sdg_filter,
                "sp": sp,
                "background_tasks": background_tasks,
                "tool_called": [False],
            },
        )

        messages: list[BaseMessage] = [HumanMessage(content=query)]
        state: _AgentInputState = {"messages": messages}

        if streamed_ans:
            res = agent_executor.astream(
                input=cast(Any, state), config=config, stream_mode="messages"
            )
            return self.get_agent_chunks(res)

        res = await agent_executor.ainvoke(input=cast(Any, state), config=config)
        return res

    async def agent_get_history(
        self,
        thread_id: uuid.UUID,
        memory: AsyncPostgresSaver,
    ) -> list[dict[str, str]]:
        agent = await self._create_agent(memory=memory)
        config = RunnableConfig(configurable={"thread_id": thread_id})

        state = await agent.aget_state(config)
        messages = state.values.get("messages", [])

        return [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in messages
            if m.type in ("human", "ai")
        ]

    async def agent_get_latest_docs(
        self,
        thread_id: uuid.UUID,
        memory: AsyncPostgresSaver,
    ) -> Optional[List[Any]]:
        """
        Falls back to the persisted checkpoint to find the most recent tool
        call's results for a thread — needed when the current turn made no
        new tool call, since nothing streamed during it would otherwise carry
        those docs to the caller.
        """
        agent = await self._create_agent(memory=memory)
        config = RunnableConfig(configurable={"thread_id": thread_id})

        state = await agent.aget_state(config)
        return latest_tool_docs(state.values.get("messages", []))

    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.RUN_LLM_WITH_JSON_PARSING.value,
    )
    async def run_llm_with_json_parsing(
        self,
        messages: list[dict],
        model_class,
        fallback_formatter: str | None = None,
    ):
        raw = await self.chat_client.completion(messages=messages)

        if not isinstance(raw, str):
            raise ValueError("LLM response must be string")

        try:
            json_data = extract_json_from_response(raw)
            if not isinstance(json_data, dict):
                raise ValueError("Extracted JSON data is not a dictionary")
            return model_class(**json_data)
        except Exception:
            if fallback_formatter:
                return await self.json_formatter_agent(raw, fallback_formatter)
            raise

    @traceable(
        run_type=TRACE_RUN_TYPE_LLM,
        name=TraceName.SYLLABUS_FEEDBACK.value,
    )
    async def syllabus_feedback_completion(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        result = await self.chat_client.completion(
            messages=messages, max_tokens=max_tokens
        )
        if not isinstance(result, str):
            raise ValueError("Syllabus feedback response is not a string")
        return result


async def get_llm_client(request: Request):
    return request.app.state.llm


async def get_chat_service(llm_client=Depends(get_llm_client)) -> AbstractChat:
    return AbstractChat(client=llm_client)
