"""Learning Outcomes Agent wrapper"""

from src.app.baml_client.async_client import b, types

from .base import BaseAgent


class LearningOutcomesAgent(BaseAgent):
    """Generates learning outcomes"""

    def __init__(self):
        super().__init__("learning_outcomes")

    async def generate(
        self,
        objectives: types.LearningObjectives,
        sustainability_map: types.SustainabilityIntegration,
        metadata: types.CourseMetadata,
        output_language: str = "Français",
    ) -> types.LearningOutcomes:
        """Generate learning outcomes"""
        self.logger.info("Generating learning outcomes")

        async def _call_baml():
            result = await b.GenerateLearningOutcomes(
                objectives=objectives,
                sustainability_map=sustainability_map,
                metadata=metadata,
                output_language=output_language,
            )
            # Convert BAML types to Pydantic models
            return types.LearningOutcomes(
                outcomes=[
                    types.LearningOutcome(
                        number=outcome.number,
                        text=outcome.text,
                        related_objectives=outcome.related_objectives,
                        assessment_method=outcome.assessment_method,
                    )
                    for outcome in result.outcomes
                ]
            )

        return await self.with_retry(_call_baml)
