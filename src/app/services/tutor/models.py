from dataclasses import dataclass
from typing import Dict, List

from pydantic import BaseModel
from qdrant_client.models import ScoredPoint


class ExtractorOutput(BaseModel):
    summary: str
    themes: list[str]


class ExtractorOutputList(BaseModel):
    extracts: list[ExtractorOutput]

class SummariesOutputModel(BaseModel):
    summaries: list[str]

class TutorSearchResponse(BaseModel):
    extracts: list[ExtractorOutput]
    nb_results: int
    documents: list[ScoredPoint]


class TutorSyllabusRequest(TutorSearchResponse):
    course_title: str | None = None
    level: str | None = None
    duration: str | None = None
    description: str | None = None


class SyllabusResponseAgent(BaseModel):
    content: str
    source: str = "default"


class SyllabusResponse(BaseModel):
    syllabus: list[SyllabusResponseAgent]
    documents: list[ScoredPoint]
    extracts: list[ExtractorOutput]


class SyllabusFeedback(SyllabusResponse):
    feedback: str
    lang: str = "en"


class MessageWithAnalysis(BaseModel):
    content: Dict
    source: str = "default"


class MessageWithResources(BaseModel):
    lang: str = "en"
    content: list[ExtractorOutput] | str
    course_title: str | None = None
    level: str | None = None
    duration: str | None = None
    description: str | None = None
    themes: list[str]
    summary: list[str]
    resources: List[Dict]
    source: str = "default"


class MessageWithFeedback(BaseModel):
    content: str
    feedback: str
    source: str = "default"


@dataclass
class TaskResponse:
    task_id: str
    result: str
