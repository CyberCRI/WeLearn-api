"""
Syllabus v2: a teacher's existing course, as read from the document they upload.

The field order of CourseReading is deliberate (as in the prototype's JSON format): short
values first, then summary and objectives, then topics and sessions, so an answer that runs
long loses the parts the outline can complete.

The model's answer is cleaned as it is parsed: every text is trimmed, an empty text is no
text, and an item without its main text (a topic without a title, an objective without its
text, a session without label nor title) is dropped.
"""

from typing import Annotated, Any, Literal, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

ObjectiveScope = Literal["course", "programme", "unspecified"]
MaterialType = Literal["syllabus", "course_plan", "lesson", "partial", "other"]

OBJECTIVE_SCOPES: tuple[str, ...] = get_args(ObjectiveScope)
MATERIAL_TYPES: tuple[str, ...] = get_args(MaterialType)

# an empty text is no text
Text = Annotated[str | None, AfterValidator(lambda value: value or None)]
Texts = Annotated[list[str], AfterValidator(lambda values: [v for v in values if v])]


class Answer(BaseModel):
    """A part of the model's answer: its texts are trimmed."""

    model_config = ConfigDict(str_strip_whitespace=True)


class ReadObjective(Answer):
    text: str = Field(
        description="The objective, word for word, without number or bullet"
    )
    scope: ObjectiveScope = "unspecified"

    # an unknown scope is not worth losing the whole reading
    @field_validator("scope", mode="before")
    @classmethod
    def known_scope(cls, value: Any) -> Any:
        return value if value in OBJECTIVE_SCOPES else "unspecified"


class ReadTopic(Answer):
    title: str
    subtopics: Texts = []
    source: Text = Field(
        default=None,
        description="The heading or line of the document the topic comes from, word for word",
    )


class ReadSession(Answer):
    # a session with only its label ("Week 8") is still a session
    label: Text = None
    title: Text = None


Objectives = Annotated[
    list[ReadObjective], AfterValidator(lambda items: [o for o in items if o.text])
]
Topics = Annotated[
    list[ReadTopic], AfterValidator(lambda items: [t for t in items if t.title])
]
Sessions = Annotated[
    list[ReadSession],
    AfterValidator(lambda items: [s for s in items if s.label or s.title]),
]


class CourseOutline(Answer):
    """Answer of the `read_outline` prompt: one part of a long document."""

    summary: Text = None
    topics: Topics = []


class CourseReading(Answer):
    """Answer of the `read` prompt."""

    title: Text = None
    discipline: Text = None
    discipline_suggested: Text = None
    level: Text = None
    document_language: Text = Field(
        default=None,
        description='ISO 639-1 code of the main language of the document, e.g. "fr"',
    )
    material_type: MaterialType = "other"
    courses_found: Texts = []
    description: Text = None
    summary: Text = None
    objectives: Objectives = []
    topics: Topics = []
    sessions: Sessions = []

    @field_validator("material_type", mode="before")
    @classmethod
    def known_material_type(cls, value: Any) -> Any:
        return value if value in MATERIAL_TYPES else "other"


# not an Answer: the text goes back as it was extracted, for the word-for-word check
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
