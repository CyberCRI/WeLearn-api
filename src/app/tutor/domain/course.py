"""
Syllabus v2: a teacher's existing course, as read from the document they upload.

The field order of CourseReading is deliberate (as in the prototype's JSON format): short
values first, then summary and objectives, then topics and sessions, so an answer that runs
long loses the parts the outline can complete.
"""

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator

ObjectiveScope = Literal["course", "programme", "unspecified"]
MaterialType = Literal["syllabus", "course_plan", "lesson", "partial", "other"]

OBJECTIVE_SCOPES: tuple[str, ...] = get_args(ObjectiveScope)
MATERIAL_TYPES: tuple[str, ...] = get_args(MaterialType)


class ReadObjective(BaseModel):
    text: str = Field(
        description="The objective, word for word, without number or bullet"
    )
    scope: ObjectiveScope = "unspecified"

    # an unknown scope is not worth losing the whole reading
    @field_validator("scope", mode="before")
    @classmethod
    def known_scope(cls, value: Any) -> Any:
        return value if value in OBJECTIVE_SCOPES else "unspecified"


class ReadTopic(BaseModel):
    title: str
    subtopics: list[str] = []
    source: str | None = Field(
        default=None,
        description="The heading or line of the document the topic comes from, word for word",
    )


class ReadSession(BaseModel):
    label: str | None = None
    title: str | None = None


class CourseOutline(BaseModel):
    """Answer of the `read_outline` prompt: one part of a long document."""

    summary: str | None = None
    topics: list[ReadTopic] = []


class CourseReading(BaseModel):
    """Answer of the `read` prompt."""

    title: str | None = None
    discipline: str | None = None
    discipline_suggested: str | None = None
    level: str | None = None
    document_language: str | None = Field(
        default=None,
        description='ISO 639-1 code of the main language of the document, e.g. "fr"',
    )
    material_type: MaterialType = "other"
    courses_found: list[str] = []
    description: str | None = None
    summary: str | None = None
    objectives: list[ReadObjective] = []
    topics: list[ReadTopic] = []
    sessions: list[ReadSession] = []

    @field_validator("material_type", mode="before")
    @classmethod
    def known_material_type(cls, value: Any) -> Any:
        return value if value in MATERIAL_TYPES else "other"


class CourseReadResponse(BaseModel):
    text: str = Field(
        description="Text extracted from the file, to check the reading word for word"
    )
    is_text_truncated: bool = Field(
        description=(
            "The document is longer than one reading: title, objectives and sessions "
            "come from its first part only (topics and summary cover every part)"
        )
    )
    is_partial: bool = Field(
        description="A part of a long document could not be read for its topics"
    )
    reading: CourseReading
