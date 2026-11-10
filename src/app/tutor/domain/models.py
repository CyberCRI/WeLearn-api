import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.app.baml_client import types
from src.app.search.domain.models import MAX_SEARCH_INPUTS
from src.app.shared.domain.documents import EnhancedDocumentSlice

# languages the tutor writes in; the value goes into the system prompts
TutorLang = Literal["en", "fr"]
# the MAS tutor client sends the language name, as in `output_language`
MasFileLang = Literal["en", "fr", "english", "french"]


class SummariesList(BaseModel):
    # each summary is one parallel search
    summaries: list[str] = Field(max_length=MAX_SEARCH_INPUTS)


class ExtractorOutput(BaseModel):
    summary: str
    themes: list[Dict]


class ExtractorOutputList(BaseModel):
    extracts: list[ExtractorOutput]


class TutorSearchResponse(BaseModel):
    extracts: list[ExtractorOutput]
    nb_results: int
    documents: list[EnhancedDocumentSlice]


class TutorSyllabusRequest(TutorSearchResponse):
    course_title: str | None = None
    discipline: int | None = None
    level: str | None = None
    duration: str | None = None
    description: str | None = None


class SyllabusResponseAgent(BaseModel):
    content: str
    source: str = "default"


class SyllabusResponse(BaseModel):
    syllabus: list[SyllabusResponseAgent]
    documents: list[EnhancedDocumentSlice]
    extracts: list[ExtractorOutput]
    syllabus_message_id: uuid.UUID | None = None


class SyllabusFeedback(SyllabusResponse):
    feedback: str
    lang: TutorLang = "en"


class SyllabusUserUpdate(BaseModel):
    syllabus: str


@dataclass
class Limits:
    """Target counts for a syllabus, derived from the course duration."""

    sessions: int | None
    objectives: int
    outcomes: int
    competencies: int


@dataclass
class RenderedCompetency:
    """A competency line of the syllabus; code and name are set for GreenComp only."""

    text: str
    los: str
    code: str = ""
    name: str = ""


class DraftOutcome(BaseModel):
    text: str = Field(
        description="Student-centred, measurable outcome starting with an action verb, "
        "including how it is demonstrated (the class activity or assessment)."
    )
    objective_numbers: list[int] = Field(
        description="1-based numbers of the learning objectives this outcome serves."
    )


class DraftCompetency(BaseModel):
    text: str = Field(description="Transferable competency developed by the course.")
    greencomp_code: str | None = Field(
        default=None,
        description="Code of the GreenComp competency this corresponds to, like '2.1'; "
        "null only if it has no GreenComp equivalent.",
    )
    outcome_numbers: list[int] = Field(
        description="1-based numbers of the learning outcomes that develop it."
    )


class DraftAssessment(BaseModel):
    method: str = Field(description="Assessment method and what students produce.")
    weight: str = Field(description="Share of the final grade, e.g. '40%'.")
    outcome_numbers: list[int] = Field(
        description="1-based numbers of the learning outcomes it evaluates."
    )


class DraftSession(BaseModel):
    topics: str = Field(description="Topics covered in this session.")
    outcome_numbers: list[int] = Field(
        description="1-based numbers of the learning outcomes targeted."
    )
    class_plan: str = Field(
        description="Student-centred plan for the session: the active-learning "
        "activities students do, with timing; teacher input kept short. Never a "
        "placeholder or ellipsis."
    )


class DraftSyllabus(BaseModel):
    """Structured syllabus returned by the LLM; references are added in code."""

    course_title: str
    description: str = Field(
        description="Course overview addressed to students: content, relevance to the "
        "academic program and to sustainability challenges. No comments about how the "
        "syllabus was designed."
    )
    objectives: list[str] = Field(
        description="Broad, teacher-centred learning objectives."
    )
    outcomes: list[DraftOutcome]
    competencies: list[DraftCompetency]
    assessment: list[DraftAssessment]
    schedule: list[DraftSession]


class MessageWithAnalysis(BaseModel):
    content: Dict
    source: str = "default"


class MessageWithResources(BaseModel):
    lang: str = "en"
    content: list[ExtractorOutput] | str
    course_title: str | None = None
    discipline: int | None = None
    level: str | None = None
    duration: str | None = None
    description: str | None = None
    themes: list[Dict]
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


class CourseDescriptionRequest(BaseModel):
    mode: str
    course_metadata: types.CourseMetadata
    context_text: str


class LearningObjectivesRequest(BaseModel):
    description: str
    course_metadata: types.CourseMetadata
    mode: str
    context_text: str


class SustainabilityIntegrationRequest(BaseModel):
    connections: list[Dict]
    suggested_objectives: types.LearningObjectives
    integration_strategy: str
    key_resources: list[str]


class LearningOutcomesRequest(BaseModel):
    description: str
    objectives: types.LearningObjectives
    course_metadata: types.CourseMetadata
    sustainability_map: types.SustainabilityIntegration


class IntegrateSustainabilityRequest(BaseModel):
    description: str
    objectives: types.LearningObjectives
    course_metadata: types.CourseMetadata
    sdg_resources: List[types.Document]
    mode: str


class CompetencyMappingRequest(BaseModel):
    outcomes: List[types.LearningOutcome]
    output_language: str
    framework: Optional[Dict[str, str]] = None


class CourseLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    M1 = "M1"
    M2 = "M2"


class ValidationIssue(BaseModel):
    """Single validation issue with fix applied"""

    description: str
    fix_applied: Optional[str] = None  # What was done to fix it


class OrchestratorState(BaseModel):
    """Complete state of the orchestrator"""

    mode: str
    metadata: types.CourseMetadata
    rag_resources: Optional[List[types.Document]] = None
    provided_objectives: Optional[List[str]] = None

    description: Optional[types.CourseDescription] = None
    objectives: Optional[types.LearningObjectives] = None
    sustainability_integration: Optional[types.SustainabilityIntegration] = None
    outcomes: Optional[types.LearningOutcomes] = None
    competencies: Optional[types.CompetencyMappings] = None
    sessions: Optional[List[types.SessionPlan]] = None
    activities: Optional[Dict[int, List[types.ActivityGuide]]] = None

    retrieved_resources: List[types.Document] = []
    validation_history: List[types.ValidationReport] = []
    current_phase: str = "input"


class SyllabusValidation(BaseModel):
    passed: bool
    major_issues: List[str]
    minor_issues: List[str]
    suggestions: List[str]


class SyllabusGenerationResponse(BaseModel):
    """
    `/tutor/api/generate` output: the BAML outputs the orchestrator stores.
    """

    status: Literal["success", "error"]
    message: Optional[str] = None
    validation: SyllabusValidation
    description: Optional[types.CourseDescription] = None
    objectives: Optional[types.LearningObjectives] = None
    outcomes: Optional[types.LearningOutcomes] = None
    competencies: Optional[types.CompetencyMappings] = None
    sustainability: Optional[types.SustainabilityIntegration] = None
    sessions: Optional[List[types.SessionPlan]] = None


# Checkpoint Models for Interactive Flow
class CheckpointData(BaseModel):
    """Data for a checkpoint in the orchestration flow"""

    phase: str  # "description", "framework", "sessions", "session_N", "complete"
    data: Dict[str, Any]
    actions: List[str]  # Available actions like ["accept", "edit", "regenerate"]


class UserAction(BaseModel):
    """User action at a checkpoint"""

    action_type: str  # "accept", "edit", "regenerate", "restart"
    edited_data: Optional[Dict[str, Any]] = None  # For direct edits
    feedback: Optional[str] = None  # For regeneration with feedback


class UserInput(BaseModel):
    """Initial user input"""

    metadata: types.CourseMetadata
    mode: Optional[str] = None
    documents: Optional[List[bytes]] = None
    documents_summaries: Optional[List[str]] = None
    rag_resources: Optional[List[types.Document]] = None
    provided_objectives: Optional[List[str]] = None
    provided_description: Optional[str] = None
    objective_processing: Optional[str] = None  # "augment" or "transform" for Mode 2B


class SyllabusGenerationRequest(BaseModel):
    mode: str
    course_metadata: types.CourseMetadata
    rag_resources: Optional[List[types.Document]] = None
    provided_objectives: Optional[List[str]] = None
    provided_description: Optional[str] = None
    context: Optional[types.FileSummary | str] = None
