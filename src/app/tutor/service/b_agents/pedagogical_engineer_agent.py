"""Pedagogical Engineer Agent wrapper"""

from src.app.baml_client.async_client import b, types

from .base import BaseAgent


class PedagogicalEngineerAgent(BaseAgent):
    """Validates pedagogical framework"""

    def __init__(self):
        super().__init__("pedagogical_engineer")

    async def validate(
        self,
        description: str,
        objectives: types.LearningObjectives,
        outcomes: types.LearningOutcomes,
        competencies: types.CompetencyMappings,
        sustainability_map: types.SustainabilityIntegration,
        metadata: types.CourseMetadata,
        output_language: str = "Français",
    ) -> types.ValidationReport:
        """Validate framework"""
        self.logger.info("Validating pedagogical framework")

        async def _call_baml():
            result = await b.ValidatePedagogicalFramework(
                description=description,
                objectives=objectives,
                outcomes=outcomes,
                competencies=competencies,
                sustainability_map=sustainability_map,
                metadata=metadata,
                output_language=output_language,
            )
            # Convert BAML types to Pydantic models
            return types.ValidationReport(
                passed=result.passed,
                issues=result.issues,
                suggestions=result.suggestions,
                severity=result.severity,
            )

        return await self.with_retry(_call_baml)
