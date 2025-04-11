from dataclasses import dataclass
from typing import Dict, List

from pydantic import BaseModel

from src.app.models.documents import Document


class ExtractorOutput(BaseModel):
    original_document: str
    summary: str
    themes: list[str]


class ExtractorOuputList(BaseModel):
    extracts: list[ExtractorOutput]


class TutorSearchResponse(BaseModel):
    extracts: list[ExtractorOutput]
    nb_results: int
    documents: list[Document]


class SyllabusResponse(BaseModel):
    syllabus: str
    documents: list[Document]


class Message(BaseModel):
    content: str
    source: str = "default"


class MessageWithAnalysis(BaseModel):
    content: Dict
    source: str = "default"


class MessageWithResources(BaseModel):
    content: list[ExtractorOutput] | str
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
