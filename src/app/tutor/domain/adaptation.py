"""
Syllabus v2: one concrete change to the course for a connection the teacher chose
(`POST /tutor/course/adaptation`), and the check of its claims.

The client sends the connection back with the resources `connection_evidence` kept for it:
the endpoint keeps no state.
"""

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field

from src.app.tutor.domain.connections import (
    MAX_RESOURCES,
    ConfirmedCourse,
    EvidencedResource,
    known,
)
from src.app.tutor.domain.course import Answer, Text
from src.app.tutor.domain.greencomp import GreenCompId
from src.app.tutor.domain.models import TutorLang

AdaptationKind = Literal["objective", "topic", "teaching", "assessment"]
AdaptationSupport = Literal["strong", "partial", "none"]
ClaimStatus = Literal["established", "not_established", "incorrect"]
ChangeFit = Literal["genuine", "forced"]
ResourceRole = Literal["direct", "context"]

KINDS: tuple[str, ...] = get_args(AdaptationKind)
SUPPORTS: tuple[str, ...] = get_args(AdaptationSupport)
CLAIM_STATUSES: tuple[str, ...] = get_args(ClaimStatus)
CHANGE_FITS: tuple[str, ...] = get_args(ChangeFit)


# ---- the `adaptation` prompt's answer ----


class EvidenceQuote(Answer):
    document: int | None = Field(default=None, description='The "n" of the document')
    quote: Text = Field(
        default=None, description="One sentence copied word for word from the document"
    )


class GreenCompAnswer(Answer):
    id: Text = None
    how: Text = None


class GuidanceAnswer(Answer):
    learning: Text = None
    greencomp: GreenCompAnswer | None = None
    teaching: Text = None
    check: Text = None


class AdaptationAnswer(Answer):
    """Answer of the `adaptation` prompt."""

    kind: Annotated[AdaptationKind | None, known(KINDS)] = None
    original: Text = Field(
        default=None,
        description="The course objective, topic or session concerned, copied exactly",
    )
    suggested: Text = None
    why: Text = None
    documents: list[int] = Field(default=[], description="The documents it relies on")
    evidence: list[EvidenceQuote] = []
    support: Annotated[AdaptationSupport | None, known(SUPPORTS)] = None
    guidance: GuidanceAnswer = GuidanceAnswer()


# ---- the `verification` prompt's answer ----


class ClaimAnswer(Answer):
    claim: Text = None
    status: Annotated[ClaimStatus | None, known(CLAIM_STATUSES)] = None
    document: int | None = None
    quote: Text = None
    problem: Text = Field(
        default=None,
        description="For a claim not established or wrong: what is wrong, for the teacher",
    )


class VerificationAnswer(Answer):
    """Answer of the `verification` prompt."""

    # required: an answer without its list of claims has checked nothing
    claims: list[ClaimAnswer]
    fit: Annotated[ChangeFit | None, known(CHANGE_FITS)] = None
    fit_problem: Text = None


# ---- POST /tutor/course/adaptation ----


class ChosenConnection(BaseModel):
    title: str
    anchor: str | None = Field(
        description="The course part the connection builds on, as the teacher wrote it"
    )
    why: str
    resources: list[EvidencedResource] = Field(
        default=[], max_length=MAX_RESOURCES, description="Direct support"
    )
    background: list[EvidencedResource] = Field(
        default=[], max_length=MAX_RESOURCES, description="Context only"
    )


class AdaptationRequest(BaseModel):
    course: ConfirmedCourse
    connection: ChosenConnection
    lang: TutorLang = "en"


class AdaptationResource(EvidencedResource):
    role: ResourceRole = Field(
        description="direct: supports the connection; context: background only"
    )
    quote: str | None = Field(
        description="The sentence of it that backs the change, found word for word"
    )


class GreenCompLink(BaseModel):
    id: GreenCompId
    how: str


class AdaptationGuidance(BaseModel):
    """Optional notes for the teacher, never part of the syllabus change."""

    learning: str | None = Field(description="What students could do thanks to it")
    greencomp: GreenCompLink | None
    teaching: str | None = Field(description="A teaching approach, in one sentence")
    check: str | None = Field(description="How the teacher might check the learning")


class AdaptationChecks(BaseModel):
    status: Literal["checked", "unchecked"] = Field(
        description="unchecked: the check gave no usable answer (support partial at most)"
    )
    problems: list[str] = Field(
        description="Claims not established by the resources, or wrong, for the teacher"
    )
    unbacked_names: list[str] = Field(
        description="Names in the change that no direct resource nor the course contains"
    )


class AdaptationProposal(BaseModel):
    kind: AdaptationKind
    original: str | None = Field(
        description=(
            "The course part changed, as the teacher wrote it; null when the change only "
            "adds something"
        )
    )
    suggested: str
    why: str
    support: AdaptationSupport = Field(
        description=(
            "strong only with a sentence of a direct resource found word for word, and "
            "never above what the check found"
        )
    )
    resources: list[AdaptationResource] = Field(
        description="The connection's resources the change relies on"
    )
    guidance: AdaptationGuidance
    checks: AdaptationChecks | None = Field(
        description="The check of its claims; null without any resource to check against"
    )
