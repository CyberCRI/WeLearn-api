"""Learning Objectives Agent wrapper"""

from src.app.baml_client.async_client import b, types

from .base import BaseAgent


class LearningObjectivesAgent(BaseAgent):
    """Generates learning objectives"""

    def __init__(self):
        super().__init__("learning_objectives")

    async def generate(
        self,
        description: str,
        context_text: str,
        metadata: types.CourseMetadata,
        mode: str,
        output_language: str = "Français",
    ) -> types.LearningObjectives:
        """Generate learning objectives"""
        self.logger.info(f"Generating learning objectives (mode={mode})")

        async def _call_baml():
            result = await b.GenerateLearningObjectives(
                description=description,
                context_text=context_text,
                metadata=metadata,
                mode=mode,
                output_language=output_language,
            )
            # Convert BAML types to Pydantic models and ensure sequential numbering
            objectives = []
            for i, obj in enumerate(result.objectives, start=1):
                objectives.append(
                    types.LearningObjective(
                        number=i,  # Force sequential numbering starting from 1
                        text=obj.text,
                    )
                )
            return types.LearningObjectives(objectives=objectives)

        return await self.with_retry(_call_baml)
