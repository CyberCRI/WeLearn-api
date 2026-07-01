"""Activity Recommendation Agent wrapper"""

from typing import List

from src.app.baml_client.async_client import b, types
from src.app.tutor.tools import (
    get_suitable_methods_for_outcome_text,
    load_greencomp_methods,
)

from .base import BaseAgent


class ActivityRecommendationAgent(BaseAgent):
    """Recommends activities for sessions using GreenComp learning methods guidance"""

    def __init__(self):
        super().__init__("activity_recommendation")
        # Pre-load GreenComp methods for reference
        self.greencomp_methods = load_greencomp_methods()

    async def recommend(
        self,
        session: types.SessionPlan,
        learning_outcomes: types.LearningOutcomes,
        filtered_activities: List[types.ActivityTemplate],
        output_language: str = "Français",
    ) -> types.ActivityRecommendations:
        """
        Recommend activities based on learning outcomes

        Args:
            session: Session plan with objectives
            learning_outcomes: All learning outcomes (filtered to session objectives in prompt)
            filtered_activities: Pre-filtered activity templates
            output_language: Output language

        Returns:
            ActivityRecommendations with selected activities and rationale
        """
        self.logger.info(
            f"Recommending activities for session {session.session_number} "
            f"based on learning outcomes"
        )

        # Get GreenComp-recommended methods for session outcomes
        # This helps guide activity selection based on pedagogical best practices
        session_outcomes = [
            out
            for out in learning_outcomes.outcomes
            if any(obj in session.objectives for obj in out.related_objectives)
        ]

        greencomp_suggestions = []
        for outcome in session_outcomes[:3]:  # Top 3 outcomes
            methods = get_suitable_methods_for_outcome_text(outcome.text)
            if methods:
                greencomp_suggestions.extend([m["name"] for m in methods[:2]])

        self.logger.info(f"GreenComp suggests methods: {greencomp_suggestions}")

        async def _call_baml():
            # Pass GreenComp method suggestions to guide activity selection
            result = await b.RecommendActivities(
                session=session,
                learning_outcomes=learning_outcomes,
                filtered_activities=filtered_activities,
                greencomp_methods=greencomp_suggestions,
                output_language=output_language,
            )
            # Convert BAML types to Pydantic models
            return types.ActivityRecommendations(
                session_number=result.session_number,
                recommended_activities=[
                    types.ActivityTemplate(
                        id=activity.id,
                        name=activity.name,
                        description=activity.description,
                        compatible_types=activity.compatible_types,
                        compatible_modes=activity.compatible_modes,
                        min_duration=activity.min_duration,
                        max_duration=activity.max_duration,
                        min_size=activity.min_size,
                        max_size=activity.max_size,
                        estimated_duration=activity.estimated_duration,
                    )
                    for activity in result.recommended_activities
                ],
                rationale=result.rationale,
            )

        return await self.with_retry(_call_baml)
