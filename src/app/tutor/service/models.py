import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from qdrant_client.models import ScoredPoint

from src.app.baml_client import types


class SummariesList(BaseModel):
    summaries: list[str]


class ExtractorOutput(BaseModel):
    summary: str
    themes: list[Dict]


class ExtractorOutputList(BaseModel):
    extracts: list[ExtractorOutput]


class TutorSearchResponse(BaseModel):
    extracts: list[ExtractorOutput]
    nb_results: int
    documents: list[ScoredPoint]


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
    documents: list[ScoredPoint]
    extracts: list[ExtractorOutput]
    syllabus_message_id: uuid.UUID | None = None


class SyllabusFeedback(SyllabusResponse):
    feedback: str
    lang: str = "en"


class SyllabusUserUpdate(BaseModel):
    syllabus: str


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


# create enum sessionmode with values PRESENTIEL, DISTANCIEL, HYBRIDE


class SessionMode(str, Enum):
    PRESENTIEL = "PRESENTIEL"
    DISTANCIEL = "DISTANCIEL"
    HYBRIDE = "HYBRIDE"


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


class Document(BaseModel):
    text: str
    metadata: Dict[str, Any]
    relevance_score: float | None = None
    url: Optional[str] = None  # Link to resource


class CourseDescription(BaseModel):
    text: str
    word_count: int


class SustainabilityConnection(BaseModel):
    objective_number: int
    sdg_themes: List[str]
    connection_explanation: str
    key_resources: List[types.Document]


class SustainabilityIntegration(BaseModel):
    connections: List[SustainabilityConnection]
    suggested_objectives: list[types.LearningObjective] | None = None
    integration_strategy: str
    resources_used: List[types.Document]


class IntegrateSustainabilityRequest(BaseModel):
    description: str
    objectives: types.LearningObjectives
    course_metadata: types.CourseMetadata
    sdg_resources: List[types.Document]
    mode: str


class LearningOutcome(BaseModel):
    number: int
    text: str
    related_objectives: List[int]
    assessment_method: str


class LearningOutcomes(BaseModel):
    outcomes: List[LearningOutcome]


class CompetencyMappingRequest(BaseModel):
    outcomes: List[LearningOutcome]
    output_language: str
    framework: Optional[Dict[str, str]] = None


class CompetencyMapping(BaseModel):
    outcome_number: int
    greencomp_competencies: List[str]
    rationale: str


class CompetencyMappings(BaseModel):
    mappings: List[CompetencyMapping]


class ValidationReport(BaseModel):
    passed: bool
    issues: List[str]
    suggestions: List[str]
    severity: str


class SessionPlan(BaseModel):
    session_number: int
    objectives: List[int]
    type: str
    mode: SessionMode
    duration: float
    class_size: int


class ActivityTemplate(BaseModel):
    id: str
    name: str
    description: str
    compatible_types: List[str]
    compatible_modes: List[SessionMode]
    min_duration: float
    max_duration: float
    min_size: int
    max_size: int
    estimated_duration: float


class ActivityGuide(BaseModel):
    activity_name: str
    description: str
    learning_objectives: List[int]
    steps: List[str]
    teacher_role: str
    student_role: str
    resources_needed: List[str]
    timing_breakdown: Dict[str, float]
    evaluation_method: str
    sustainability_integration: str


class ActivityRecommendations(BaseModel):
    session_number: int
    recommended_activities: List[ActivityTemplate]
    rationale: str


class CourseLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    M1 = "M1"
    M2 = "M2"


# Core Models
class CourseMetadata(BaseModel):
    """Course metadata provided by user"""

    discipline: str
    topic: str
    level: str  # Free-form to allow flexibility
    num_sessions: int
    num_objectives: Optional[int] = None  # Target number of objectives
    session_duration: float
    session_type: str  # Free-form (CM, TD, TP, Séminaire, etc.)
    session_mode: SessionMode
    class_size: int
    output_language: str = "Français"  # Default to French


class LearningObjective(BaseModel):
    """Single learning objective (teacher-centered: what will be taught/covered)"""

    number: int
    text: str


class LearningObjectives(BaseModel):
    """Collection of learning objectives"""

    objectives: List[LearningObjective]


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

    description: Optional[CourseDescription] = None
    objectives: Optional[LearningObjectives] = None
    sustainability_integration: Optional[SustainabilityIntegration] = None
    outcomes: Optional[LearningOutcomes] = None
    competencies: Optional[CompetencyMappings] = None
    sessions: Optional[List[SessionPlan]] = None
    activities: Optional[Dict[int, List[ActivityGuide]]] = None

    retrieved_resources: List[types.Document] = []
    validation_history: List[ValidationReport] = []
    current_phase: str = "input"


class SyllabusOutput(BaseModel):
    """Final complete syllabus output"""

    description: Optional[CourseDescription] = None
    objectives: Optional[LearningObjectives] = None
    outcomes: Optional[LearningOutcomes] = None
    competencies: Optional[CompetencyMappings] = None
    sustainability: Optional[SustainabilityIntegration] = None
    sessions: Optional[List[SessionPlan]] = None
    # activities: Optional[Dict[int, List[ActivityGuide]]] = None


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
