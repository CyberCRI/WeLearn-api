"""
Agent wrappers - Python interface to BAML agents
"""

from .activity_guide_generator_agent import ActivityGuideGeneratorAgent
from .activity_recommendation_agent import ActivityRecommendationAgent
from .base import AgentError, BaseAgent
from .competency_mapping_agent import CompetencyMappingAgent
from .course_description_agent import CourseDescriptionAgent
from .learning_objectives_agent import LearningObjectivesAgent
from .learning_outcomes_agent import LearningOutcomesAgent
from .pedagogical_engineer_agent import PedagogicalEngineerAgent
from .sustainability_integration_agent import SustainabilityIntegrationAgent

__all__ = [
    "BaseAgent",
    "AgentError",
    "CourseDescriptionAgent",
    "LearningObjectivesAgent",
    "SustainabilityIntegrationAgent",
    "LearningOutcomesAgent",
    "CompetencyMappingAgent",
    "PedagogicalEngineerAgent",
    "ActivityRecommendationAgent",
    "ActivityGuideGeneratorAgent",
]
