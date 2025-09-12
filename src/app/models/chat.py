from enum import Enum
from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from qdrant_client.models import ScoredPoint

from .documents import Document


class Context(BaseModel):
    sources: list[Document] = []
    history: list[dict] | None = []
    query: str | None = None
    subject: str | None = Field(None)


class ContextOut(BaseModel):
    sources: list[Document] = []
    history: list[dict] = []
    query: str
    subject: str | None = Field(None)


class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


ROLES = Literal["user", "system", "assistant"]


class Message(TypedDict):
    role: str
    content: str


class ReformulatedQueryResponse(BaseModel):
    """
    The schema for reformulated queries.

    Attributes:
        STANDALONE_QUESTION_EN (str): The standalone question in english.
        STANDALONE_QUESTION_FR (str): The standalone question in french.
        USER_LANGUAGE (str): The user language.
        REF_TO_PAST (bool): A reference to past messages.
    """

    STANDALONE_QUESTION_EN: str | None = None
    STANDALONE_QUESTION_FR: str | None = None
    USER_LANGUAGE: str | None = None
    QUERY_STATUS: (
        Literal["INVALID"] | Literal["VALID"] | Literal["REF_TO_PAST"] | None
    ) = None


class ReformulatedQuestionsResponse(BaseModel):
    NEW_QUESTIONS: list[str]


class AgentContext(BaseModel):
    query: str | None = None
    thread_id: str | None = None


class AgentResponse(BaseModel):
    content: str | None = None
    docs: list[ScoredPoint] | None = None


PROMPTS = Literal["STANDALONE", "NEW_QUESTIONS", "REPHRASE"]

RESPONSE_TYPE = Literal["json_object", "text"]
