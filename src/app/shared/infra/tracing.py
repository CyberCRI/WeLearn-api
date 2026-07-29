from enum import Enum

TRACE_RUN_TYPE_LLM = "llm"


class TraceComponent(str, Enum):
    CHAT_NON_AGENT = "chat_non_agent"
    CHAT_AGENT = "chat_agent"


class TraceName(str, Enum):
    COMPLETION_NON_AGENT = "Completion (non-agent)"
    AZURE_COMPLETION_NON_AGENT = "Azure completion (non-agent)"
    AZURE_COMPLETION_STREAM_NON_AGENT = "Azure completion stream (non-agent)"
    COMPLETION_STREAM_NON_AGENT = "Completion stream (non-agent)"
    MISTRAL_COMPLETION_NON_AGENT = "Mistral completion (non-agent)"
    MISTRAL_COMPLETION_STREAM_NON_AGENT = "Mistral completion stream (non-agent)"
