"""Competency Mapping Agent wrapper"""

from typing import Dict

from src.app.baml_client.async_client import b, types

from .base import BaseAgent


class CompetencyMappingAgent(BaseAgent):
    """Maps outcomes to GreenComp competencies"""

    def __init__(self):
        super().__init__("competency_mapping")

    async def map_competencies(
        self,
        outcomes: types.LearningOutcomes,
        greencomp_framework: Dict[str, str],
        output_language: str = "Français",
    ) -> types.CompetencyMappings:
        """Map competencies"""
        self.logger.info("Mapping competencies")

        async def _call_baml():
            result = await b.MapCompetencies(
                outcomes=outcomes,
                greencomp_framework=greencomp_framework,
                output_language=output_language,
            )
            # Convert BAML types to Pydantic models
            return types.CompetencyMappings(
                mappings=[
                    types.CompetencyMapping(
                        outcome_number=mapping.outcome_number,
                        greencomp_competencies=mapping.greencomp_competencies,
                        rationale=mapping.rationale,
                    )
                    for mapping in result.mappings
                ]
            )

        return await self.with_retry(_call_baml)
