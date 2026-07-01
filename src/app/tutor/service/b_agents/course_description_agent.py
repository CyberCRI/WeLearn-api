"""Course Description Agent wrapper"""

from src.app.baml_client.async_client import b, types

from .base import BaseAgent


class CourseDescriptionAgent(BaseAgent):
    """Generates course descriptions"""

    def __init__(self):
        super().__init__("course_description")

    async def generate(
        self,
        mode: str,
        metadata: types.CourseMetadata,
        context_text: str = "",
        output_language: str = "Français",
    ) -> types.CourseDescription:
        """
        Generate course description

        Args:
            mode: Input mode (mode_1, mode_2a, mode_2b, mode_3)
            metadata: Course metadata
            context_text: Contextual text (documents, description)
            output_language: Output language

        Returns:
            CourseDescription object
        """
        self.logger.info(f"Generating course description (mode={mode})")

        async def _call_baml():
            result = await b.GenerateCourseDescription(
                mode=mode,
                metadata=metadata,
                context_text=context_text,
                output_language=output_language,
            )
            # Convert BAML types back to Pydantic models
            return types.CourseDescription(
                text=result.text, word_count=result.word_count
            )

        return await self.with_retry(_call_baml)


# Additional agent wrappers follow similar pattern
