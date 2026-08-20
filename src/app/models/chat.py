import uuid
from enum import Enum
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.models import ScoredPoint

from src.app.search.models.search import SDGFilter

from .documents import Document


class Context(BaseModel):
    sources: list[Document] = []
    history: list[dict] | None = []
    query: str | None = None
    subject: str | None = Field(None)
    lang: str | None = None


class ContextOut(BaseModel):
    sources: list[Document] = []
    history: list[dict] = []
    query: str
    subject: str | None = Field(None)
    conversation_id: uuid.UUID | None = Field(None)
    lang: str | None = None


class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


ROLES = Literal["user", "system", "assistant"]


class Message(TypedDict):
    role: str
    content: str


class ReformulatedQuestionsResponse(BaseModel):
    NEW_QUESTIONS: list[str]


class AgentContext(SDGFilter):
    query: str | None = Field(
        default=None,
        description="User message for the agent to answer.",
        examples=["How can communities improve clean water access?"],
    )
    thread_id: uuid.UUID | None = Field(
        default=None,
        description="Optional conversation thread id. If omitted, a new thread id is generated.",
        examples=["6fca8d02-0bc2-48a5-9c95-4942ca57e651"],
    )
    corpora: tuple[str, ...] | None = Field(
        default=None,
        description="Optional list of corpus names used by retrieval tools during agent execution.",
        examples=[["conversation"]],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "How can communities improve clean water access?",
                    "corpora": ["conversation"],
                    "sdg_filter": [6],
                },
                {
                    "query": "Give me practical school-level actions for SDG 4 and SDG 6.",
                    "thread_id": "6fca8d02-0bc2-48a5-9c95-4942ca57e651",
                    "corpora": ["conversation", "wikipedia"],
                    "sdg_filter": [4, 6],
                },
            ]
        }
    )


class TraceContext(TypedDict):
    endpoint: str
    feature: str
    environment: str
    session_id: str | None
    thread_id: str
    query_length: int
    sdg_filter: list[int] | None
    corpora: list[str] | None


class AgentResponse(BaseModel):
    content: str | None = None
    status: str | None = None
    step: str | None = None
    docs: list[ScoredPoint] | None = None
    thread_id: uuid.UUID | None = None


class UserQueryMetadata(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID


RESPONSE_TYPE = Literal["json_object", "text"]
