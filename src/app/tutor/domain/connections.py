"""
Syllabus v2: sustainability connections for a course the teacher has confirmed.

`ConfirmedCourse` is the course as the teacher validated it on the review screen (their
corrections included, degree-level objectives left out). The client sends it back with each
call: the endpoints keep no state.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator

from src.app.tutor.domain.course import Answer, Text, Texts
from src.app.tutor.domain.models import TutorLang

# a small set the teacher can weigh
MAX_IDEAS = 4


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
