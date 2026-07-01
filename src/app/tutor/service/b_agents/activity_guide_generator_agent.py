"""Activity Guide Generator Agent wrapper"""

from src.app.baml_client.async_client import b, types

from .base import BaseAgent


class ActivityGuideGeneratorAgent(BaseAgent):
    """Generates personalized activity guides"""

    def __init__(self):
        super().__init__("activity_guide_generator")

    async def generate(
        self,
        activity_template: types.ActivityTemplate,
        session: types.SessionPlan,
        metadata: types.CourseMetadata,
        sustainability_map: types.SustainabilityIntegration,
        output_language: str = "Français",
    ) -> types.ActivityGuide:
        """Generate activity guide"""
        self.logger.info(f"Generating guide for activity {activity_template.name}")

        async def _call_baml():
            result = await b.GenerateActivityGuide(
                activity_template=activity_template,
                session=session,
                metadata=metadata,
                sustainability_map=sustainability_map,
                output_language=output_language,
            )
            # Convert BAML types to Pydantic models
            return types.ActivityGuide(
                activity_name=result.activity_name,
                description=result.description,
                learning_objectives=result.learning_objectives,
                steps=result.steps,
                teacher_role=result.teacher_role,
                student_role=result.student_role,
                resources_needed=result.resources_needed,
                timing_breakdown=result.timing_breakdown,
                evaluation_method=result.evaluation_method,
                sustainability_integration=result.sustainability_integration,
            )

        return await self.with_retry(_call_baml)
