from enum import Enum

TRACE_RUN_TYPE_LLM = "llm"


class TraceComponent(str, Enum):
    CHAT_NON_AGENT = "chat_non_agent"
    CHAT_AGENT = "chat_agent"


class TraceName(str, Enum):
    GET_NEW_QUESTIONS = "Get new questions"
    REPHRASE_MESSAGE = "Rephrase message"
    REFORMULATE_USER_QUERY = "Reformulate user query"
    CHAT_MESSAGE = "Chat message"
    JSON_FORMATTER_AGENT = "JSON formatter agent"
    RUN_LLM_WITH_JSON_PARSING = "Run LLM with JSON parsing"
    DETECT_LANG_WITH_LLM = "Detect language with LLM"
    DETECT_PAST_MESSAGE_REF = "Detect past message reference"
    SYLLABUS_FEEDBACK = "Syllabus feedback"
    JUDGE_SOURCE_GROUNDING = "Judge source grounding"
