from enum import Enum


class TraceComponent(str, Enum):
    CHAT_AGENT = "chat_agent"


class TraceName(str, Enum):
    TUTOR_SYLLABUS_GENERATION = "Tutor syllabus generation"
