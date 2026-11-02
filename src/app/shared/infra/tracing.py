from enum import Enum

TRACE_RUN_TYPE_LLM = "llm"


class TraceComponent(str, Enum):
    CHAT_AGENT = "chat_agent"


class TraceName(str, Enum):
    JSON_FORMATTER_AGENT = "JSON formatter agent"
    RUN_LLM_WITH_JSON_PARSING = "Run LLM with JSON parsing"
    SYLLABUS_FEEDBACK = "Syllabus feedback"
