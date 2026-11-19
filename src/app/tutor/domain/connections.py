"""
Syllabus v2: sustainability connections for a course the teacher has confirmed.

`ConfirmedCourse` is the course as the teacher validated it on the review screen (their
corrections included, degree-level objectives left out). The client sends it back with each
call: the endpoints keep no state.
"""

from typing import Annotated, Literal, get_args

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, model_validator

from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.tutor.domain.course import Answer, Text, Texts
from src.app.tutor.domain.models import TutorLang

# a small set the teacher can weigh: a few ideas, a few resources each
MAX_IDEAS = 4
MAX_RESOURCES = 3


class CourseTopic(BaseModel):
    title: str
    subtopics: list[str] = []


class ConfirmedCourse(BaseModel):
    title: str | None = None
    discipline: str | None = None
    level: str | None = None
    overview: str | None = None
    topics: list[CourseTopic] = []
    objectives: list[str] = []
    competencies: list[str] = []
    sessions: list[str] = Field(
        default=[], description="The course outline: sessions, weeks or chapters"
    )

    @model_validator(mode="after")
    def says_something(self) -> "ConfirmedCourse":
        if not any(
            [
                self.title,
                self.overview,
                self.topics,
                self.objectives,
                self.competencies,
                self.sessions,
            ]
        ):
            raise ValueError("The course is empty")
        return self


# ---- the `ideas` prompt's answer ----


class IdeaAnswer(Answer):
    title: Text = None
    anchor: Text = Field(
        default=None,
        description="The course part concerned, copied exactly from the course",
    )
    anchor_label: Text = Field(
        default=None, description="2 to 5 words naming what this course part covers"
    )
    why: Text = None
    use: Text = Field(
        default=None,
        description="How the connection uses the course's concept in teaching",
    )
    queries: Texts = Field(default=[], description="Two search questions")


def _complete(ideas: list[IdeaAnswer]) -> list[IdeaAnswer]:
    """An idea without a title, a reason or a search question can't be used."""
    return [idea for idea in ideas if idea.title and idea.why and idea.queries]


class IdeasAnswer(Answer):
    """Answer of the `ideas` prompt."""

    ideas: Annotated[list[IdeaAnswer], AfterValidator(_complete)] = []
    none_reason: Text = Field(
        default=None,
        description="Why no connection is strong enough, when no connection is given",
    )

    # an empty list is a judgement only with its reason: otherwise the answer is unusable
    # (asked again), never shown as "no connection found"
    @model_validator(mode="after")
    def ideas_or_reason(self) -> "IdeasAnswer":
        if not self.ideas and not self.none_reason:
            raise ValueError("No usable idea and no reason for none")
        return self


# ---- POST /tutor/course/connection_ideas ----


class ConnectionIdeasRequest(BaseModel):
    course: ConfirmedCourse
    lang: TutorLang = "en"


class ConnectionIdea(BaseModel):
    id: str = Field(description="Given by the server: idea-1, idea-2…")
    title: str
    anchor: str | None = Field(
        description=(
            "The course part the connection builds on, as the teacher wrote it "
            "(a topic, subtopic, objective, competency or session); null when the "
            "model named no part of this course"
        )
    )
    anchor_label: str | None = Field(
        description="A few words naming the course part; null without an anchor"
    )
    why: str
    use: str | None
    queries: list[str] = Field(
        min_length=1, max_length=2, description="The searches for its resources"
    )


class ConnectionIdeasResponse(BaseModel):
    ideas: list[ConnectionIdea] = Field(max_length=MAX_IDEAS)
    none_reason: str | None = Field(
        description="Why no connection is proposed, for the teacher; set only when ideas is empty"
    )


# ---- the `selection` prompt's answer ----

Fit = Literal["genuine", "analogy", "off_discipline"]
Relevance = Literal["direct", "context"]
FITS: tuple[str, ...] = get_args(Fit)
RELEVANCES: tuple[str, ...] = get_args(Relevance)


def known(values: tuple[str, ...]) -> BeforeValidator:
    """An unknown value is no value: not worth losing the whole answer."""
    return BeforeValidator(lambda value: value if value in values else None)


class SelectedResource(Answer):
    # every field may be missing: one bad item is left out, not the whole answer
    document: int | None = Field(default=None, description='The "n" of the document')
    subject: Text = None
    concept_terms: Texts = []
    sustainability_terms: Texts = []
    same_sense: bool | None = None
    establishes: bool | None = None
    relevance: Annotated[Relevance | None, known(RELEVANCES)] = None
    usefulness: Text = None
    evidence: Text = Field(
        default=None, description="One sentence copied word for word from the document"
    )


class SelectionAnswer(Answer):
    """Answer of the `selection` prompt."""

    fit: Annotated[Fit | None, known(FITS)] = None
    fit_reason: Text = None
    keep: bool
    concept: Text = None
    why: Text = None
    resources: list[SelectedResource] = []


# ---- POST /tutor/course/connection_evidence ----


class ConnectionEvidenceRequest(BaseModel):
    course: ConfirmedCourse
    idea: ConnectionIdea
    lang: TutorLang = "en"


class EvidencedResource(BaseModel):
    document: EnhancedDocumentSlice
    usefulness: str = Field(description="What it brings to the connection in class")
    evidence: str = Field(
        description="The sentence it was kept for, found word for word in its text"
    )


EvidenceStatus = Literal["supported", "unsupported", "forced", "unchecked"]


class ConnectionEvidenceResponse(BaseModel):
    status: EvidenceStatus = Field(
        description=(
            "supported: at least one resource supports the connection directly; "
            "unsupported: none does (no candidate, or none passed the checks); "
            "forced: the connection itself was judged an analogy or off the discipline; "
            "unchecked: the resources could not be judged (ask again)"
        )
    )
    concept: str | None = Field(
        description="The course's concept or mechanism the connection rests on"
    )
    why: str = Field(
        description=(
            "Why the connection matters for the course: the judgement's words when a "
            "resource supports it, otherwise the idea's own"
        )
    )
    resources: list[EvidencedResource] = Field(
        description="Direct support, best first (at most 3 with background)"
    )
    background: list[EvidencedResource] = Field(
        description="Context on the sustainability issue only, never counted as support"
    )
    candidate_count: int = Field(description="Search results judged")
