from enum import Enum


class TraceComponent(str, Enum):
    CHAT_AGENT = "chat_agent"


class TraceName(str, Enum):
    FILES_SUMMARY_EXTRACTOR = "Files summary extractor"
    TUTOR_ADAPTATION = "Tutor adaptation"
    TUTOR_SYLLABUS_GENERATION = "Tutor syllabus generation"
