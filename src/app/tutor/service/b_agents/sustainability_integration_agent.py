"""Sustainability Integration Agent wrapper"""

from typing import List

from src.app.baml_client.async_client import b, types

from .base import BaseAgent


class SustainabilityIntegrationAgent(BaseAgent):
    """Creates sustainability connections"""

    def __init__(self):
        super().__init__("sustainability_integration")

    async def integrate(
        self,
        description: str,
        objectives: types.LearningObjectives,
        sdg_resources: List[types.Document],
        metadata: types.CourseMetadata,
        mode: str,
        output_language: str = "Français",
    ) -> types.SustainabilityIntegration:
        """Create sustainability integration"""
        self.logger.info(f"Integrating sustainability (mode={mode})")

        async def _call_baml():
            result = await b.IntegrateSustainability(
                description=description,
                objectives=objectives,
                sdg_resources=sdg_resources,
                metadata=metadata,
                mode=mode,
                output_language=output_language,
            )
            # Convert BAML types to Pydantic models
            return types.SustainabilityIntegration(
                connections=[
                    types.SustainabilityConnection(
                        objective_number=conn.objective_number,
                        sdg_themes=conn.sdg_themes,
                        connection_explanation=conn.connection_explanation,
                        key_resources=conn.key_resources,
                    )
                    for conn in result.connections
                ],
                suggested_objectives=(
                    [
                        types.LearningObjective(
                            number=obj.number,
                            text=obj.text,
                            bloom_level=obj.bloom_level,
                        )
                        for obj in result.suggested_objectives
                    ]
                    if result.suggested_objectives
                    else None
                ),
                integration_strategy=result.integration_strategy,
                resources_used=result.resources_used,
            )

        return await self.with_retry(_call_baml)
