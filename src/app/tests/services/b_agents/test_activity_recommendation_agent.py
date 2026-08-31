import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.baml_client.types import SessionMode
from src.app.tutor.service.b_agents.activity_recommendation_agent import (
    ActivityRecommendationAgent,
)


class TestActivityRecommendationAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = ActivityRecommendationAgent()
        self.session = types.SessionPlan(
            session_number=2,
            objectives=[1, 2],
            type="fake type",
            mode=SessionMode.PRESENTIEL,
            duration=1.5,
            class_size=20,
        )

        self.learning_outcomes = types.LearningOutcomes(
            outcomes=[
                types.LearningOutcome(
                    number=1,
                    text="Understand sustainable development concepts.",
                    related_objectives=[1],
                    assessment_method="Written examination",
                ),
                types.LearningOutcome(
                    number=2,
                    text="Apply sustainability concepts to practical problems.",
                    related_objectives=[2],
                    assessment_method="Practical assignment",
                ),
                types.LearningOutcome(
                    number=3,
                    text="Analyze sustainability challenges.",
                    related_objectives=[1, 2],
                    assessment_method="Project evaluation",
                ),
            ]
        )

        self.filtered_activities = [
            types.ActivityTemplate(
                name="Group discussion",
                id="activity-1",
                description="A collaborative group discussion.",
                compatible_types=["seminaire"],
                compatible_modes=[SessionMode.PRESENTIEL],
                min_duration=1.0,
                max_duration=3.0,
                min_size=5,
                max_size=30,
                estimated_duration=2.0,
            ),
            types.ActivityTemplate(
                name="Case study",
                id="activity-2",
                description="A sustainability case study.",
                compatible_types=["seminaire"],
                compatible_modes=[SessionMode.PRESENTIEL],
                min_duration=1.0,
                max_duration=3.0,
                min_size=5,
                max_size=30,
                estimated_duration=2.0,
            ),
        ]

        self.activity_recommendations = types.ActivityRecommendations(
            session_number=2,
            recommended_activities=[
                types.ActivityTemplate(
                    name="Group discussion",
                    id="activity-1",
                    description="A collaborative group discussion.",
                    compatible_types=["seminaire"],
                    compatible_modes=[SessionMode.PRESENTIEL],
                    min_duration=1.0,
                    max_duration=3.0,
                    min_size=5,
                    max_size=30,
                    estimated_duration=2.0,
                ),
                types.ActivityTemplate(
                    name="Case study",
                    id="activity-2",
                    description="A sustainability case study.",
                    compatible_types=["seminaire"],
                    compatible_modes=[SessionMode.PRESENTIEL],
                    min_duration=1.0,
                    max_duration=3.0,
                    min_size=5,
                    max_size=30,
                    estimated_duration=2.0,
                ),
            ],
            rationale="These activities support the learning outcomes.",
        )

        self.greencomp_methods = [
            {"name": "Collaborative learning"},
            {"name": "Critical thinking"},
        ]

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_calls_baml(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=self.greencomp_methods,
        ):
            result = await self.agent.recommend(
                session=self.session,
                learning_outcomes=self.learning_outcomes,
                filtered_activities=self.filtered_activities,
                output_language="Français",
            )

        mock_recommend.assert_awaited_once_with(
            session=self.session,
            learning_outcomes=self.learning_outcomes,
            filtered_activities=self.filtered_activities,
            greencomp_methods=[
                "Collaborative learning",
                "Critical thinking",
                "Collaborative learning",
                "Critical thinking",
                "Collaborative learning",
                "Critical thinking",
            ],
            output_language="Français",
        )

        self.assertEqual(
            result,
            self.activity_recommendations,
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_uses_default_language(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=[],
        ):
            await self.agent.recommend(
                session=self.session,
                learning_outcomes=self.learning_outcomes,
                filtered_activities=self.filtered_activities,
            )

        mock_recommend.assert_awaited_once_with(
            session=self.session,
            learning_outcomes=self.learning_outcomes,
            filtered_activities=self.filtered_activities,
            greencomp_methods=[],
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_returns_activity_recommendations(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=[],
        ):
            result = await self.agent.recommend(
                session=self.session,
                learning_outcomes=self.learning_outcomes,
                filtered_activities=self.filtered_activities,
            )

        self.assertIsInstance(
            result,
            types.ActivityRecommendations,
        )

        self.assertEqual(
            result.session_number,
            self.activity_recommendations.session_number,
        )

        self.assertEqual(
            len(result.recommended_activities),
            len(self.activity_recommendations.recommended_activities),
        )

        for result_activity, expected_activity in zip(
            result.recommended_activities,
            self.activity_recommendations.recommended_activities,
        ):
            self.assertEqual(
                result_activity.id,
                expected_activity.id,
            )

            self.assertEqual(
                result_activity.name,
                expected_activity.name,
            )

            self.assertEqual(
                result_activity.description,
                expected_activity.description,
            )

            self.assertEqual(
                result_activity.compatible_types,
                expected_activity.compatible_types,
            )

            self.assertEqual(
                result_activity.compatible_modes,
                expected_activity.compatible_modes,
            )

            self.assertEqual(
                result_activity.min_duration,
                expected_activity.min_duration,
            )

            self.assertEqual(
                result_activity.max_duration,
                expected_activity.max_duration,
            )

            self.assertEqual(
                result_activity.min_size,
                expected_activity.min_size,
            )

            self.assertEqual(
                result_activity.max_size,
                expected_activity.max_size,
            )

            self.assertEqual(
                result_activity.estimated_duration,
                expected_activity.estimated_duration,
            )

        self.assertEqual(
            result.rationale,
            self.activity_recommendations.rationale,
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_uses_greencomp_methods(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=self.greencomp_methods,
        ) as mock_get_methods:

            await self.agent.recommend(
                session=self.session,
                learning_outcomes=self.learning_outcomes,
                filtered_activities=self.filtered_activities,
            )

        self.assertEqual(
            mock_get_methods.call_count,
            3,
        )

        expected_texts = [
            "Understand sustainable development concepts.",
            "Apply sustainability concepts to practical problems.",
            "Analyze sustainability challenges.",
        ]

        actual_texts = [call.args[0] for call in mock_get_methods.call_args_list]

        self.assertEqual(
            actual_texts,
            expected_texts,
        )

        mock_recommend.assert_awaited_once_with(
            session=self.session,
            learning_outcomes=self.learning_outcomes,
            filtered_activities=self.filtered_activities,
            greencomp_methods=[
                "Collaborative learning",
                "Critical thinking",
                "Collaborative learning",
                "Critical thinking",
                "Collaborative learning",
                "Critical thinking",
            ],
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_only_uses_outcomes_related_to_session_objectives(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        unrelated_outcome = types.LearningOutcome(
            number=4,
            text="This outcome is unrelated to the session.",
            related_objectives=[99],
            assessment_method="Exam",
        )

        learning_outcomes = types.LearningOutcomes(
            outcomes=[
                *self.learning_outcomes.outcomes,
                unrelated_outcome,
            ]
        )

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=self.greencomp_methods,
        ) as mock_get_methods:

            await self.agent.recommend(
                session=self.session,
                learning_outcomes=learning_outcomes,
                filtered_activities=self.filtered_activities,
            )

        self.assertEqual(
            mock_get_methods.call_count,
            3,
        )

        mock_get_methods.assert_any_call("Understand sustainable development concepts.")

        mock_get_methods.assert_any_call(
            "Apply sustainability concepts to practical problems."
        )

        mock_get_methods.assert_any_call("Analyze sustainability challenges.")

        mock_recommend.assert_awaited_once()

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_limits_greencomp_processing_to_three_outcomes(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        outcomes = [
            types.LearningOutcome(
                number=i,
                text=f"Learning outcome {i}",
                related_objectives=[1],
                assessment_method="Exam",
            )
            for i in range(1, 6)
        ]

        learning_outcomes = types.LearningOutcomes(outcomes=outcomes)

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=self.greencomp_methods,
        ) as mock_get_methods:

            await self.agent.recommend(
                session=self.session,
                learning_outcomes=learning_outcomes,
                filtered_activities=self.filtered_activities,
            )

        self.assertEqual(
            mock_get_methods.call_count,
            3,
        )

        expected_texts = [
            "Learning outcome 1",
            "Learning outcome 2",
            "Learning outcome 3",
        ]

        actual_texts = [call.args[0] for call in mock_get_methods.call_args_list]

        self.assertEqual(
            actual_texts,
            expected_texts,
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_ignores_empty_greencomp_methods(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=[],
        ):
            await self.agent.recommend(
                session=self.session,
                learning_outcomes=self.learning_outcomes,
                filtered_activities=self.filtered_activities,
            )

        mock_recommend.assert_awaited_once_with(
            session=self.session,
            learning_outcomes=self.learning_outcomes,
            filtered_activities=self.filtered_activities,
            greencomp_methods=[],
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_with_custom_language(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.return_value = self.activity_recommendations

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=[],
        ):
            await self.agent.recommend(
                session=self.session,
                learning_outcomes=self.learning_outcomes,
                filtered_activities=self.filtered_activities,
                output_language="English",
            )

        mock_recommend.assert_awaited_once_with(
            session=self.session,
            learning_outcomes=self.learning_outcomes,
            filtered_activities=self.filtered_activities,
            greencomp_methods=[],
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.load_greencomp_methods",
        return_value=[],
    )
    @patch(
        "src.app.tutor.service.b_agents.activity_recommendation_agent.b.RecommendActivities",
        new_callable=AsyncMock,
    )
    async def test_recommend_propagates_exception(
        self,
        mock_recommend,
        mock_load_methods,
    ):
        mock_recommend.side_effect = RuntimeError("BAML recommendation failed")

        with patch(
            "src.app.tutor.service.b_agents.activity_recommendation_agent.get_suitable_methods_for_outcome_text",
            return_value=[],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "BAML recommendation failed",
            ):
                await self.agent.recommend(
                    session=self.session,
                    learning_outcomes=self.learning_outcomes,
                    filtered_activities=self.filtered_activities,
                )
