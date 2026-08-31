import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.baml_client.types import SessionMode
from src.app.tutor.service.b_agents.learning_outcomes_agent import LearningOutcomesAgent


class TestLearningOutcomesAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = LearningOutcomesAgent()

        self.objectives = types.LearningObjectives(
            objectives=[
                types.LearningObjective(
                    number=1,
                    text="Understand the fundamentals of computer science.",
                ),
                types.LearningObjective(
                    number=2,
                    text="Apply fundamental computer science concepts.",
                ),
                types.LearningObjective(
                    number=3,
                    text="Analyze basic computer science problems.",
                ),
            ]
        )

        self.sustainability_map = types.SustainabilityIntegration(
            connections=[
                types.SustainabilityConnection(
                    objective_number=1,
                    sdg_themes=["fake themes"],
                    connection_explanation="fake explanation",
                    key_resources=[
                        types.Document(
                            text="fake text",
                            corpus="toto",
                            description="fake description",
                            sdg=[1],
                            url="fake url",
                            title="fake title",
                        )
                    ],
                )
            ],
            integration_strategy="fake strategy",
            resources_used=[
                types.Document(
                    text="fake text",
                    corpus="toto",
                    description="fake description",
                    sdg=[1],
                    url="fake url",
                    title="fake title",
                )
            ],
        )

        self.metadata = types.CourseMetadata(
            output_language="en",
            discipline="computer science",
            level="licence",
            topic="fake topic",
            num_sessions=4,
            session_type="seminaire",
            session_mode=SessionMode.PRESENTIEL,
            class_size=10,
            session_duration=1.5,
        )

        self.learning_outcomes = types.LearningOutcomes(
            outcomes=[
                types.LearningOutcome(
                    number=1,
                    text="Explain fundamental computer science concepts.",
                    related_objectives=[1],
                    assessment_method="Written examination",
                ),
                types.LearningOutcome(
                    number=2,
                    text="Apply computer science concepts to practical problems.",
                    related_objectives=[1, 2],
                    assessment_method="Practical assignment",
                ),
                types.LearningOutcome(
                    number=3,
                    text="Analyze and solve basic computer science problems.",
                    related_objectives=[2, 3],
                    assessment_method="Project evaluation",
                ),
            ]
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_outcomes_agent.b.GenerateLearningOutcomes",
        new_callable=AsyncMock,
    )
    async def test_generate_calls_baml(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_outcomes

        result = await self.agent.generate(
            objectives=self.objectives,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="Français",
        )

        mock_generate.assert_awaited_once_with(
            objectives=self.objectives,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="Français",
        )

        self.assertEqual(
            result,
            self.learning_outcomes,
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_outcomes_agent.b.GenerateLearningOutcomes",
        new_callable=AsyncMock,
    )
    async def test_generate_uses_default_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_outcomes

        await self.agent.generate(
            objectives=self.objectives,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
        )

        mock_generate.assert_awaited_once_with(
            objectives=self.objectives,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_outcomes_agent.b.GenerateLearningOutcomes",
        new_callable=AsyncMock,
    )
    async def test_generate_returns_learning_outcomes(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_outcomes

        result = await self.agent.generate(
            objectives=self.objectives,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
        )

        self.assertIsInstance(
            result,
            types.LearningOutcomes,
        )

        self.assertEqual(
            len(result.outcomes),
            len(self.learning_outcomes.outcomes),
        )

        for result_outcome, expected_outcome in zip(
            result.outcomes,
            self.learning_outcomes.outcomes,
        ):
            self.assertEqual(
                result_outcome.number,
                expected_outcome.number,
            )

            self.assertEqual(
                result_outcome.text,
                expected_outcome.text,
            )

            self.assertEqual(
                result_outcome.related_objectives,
                expected_outcome.related_objectives,
            )

            self.assertEqual(
                result_outcome.assessment_method,
                expected_outcome.assessment_method,
            )

    @patch(
        "src.app.tutor.service.b_agents.learning_outcomes_agent.b.GenerateLearningOutcomes",
        new_callable=AsyncMock,
    )
    async def test_generate_with_custom_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_outcomes

        await self.agent.generate(
            objectives=self.objectives,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="English",
        )

        mock_generate.assert_awaited_once_with(
            objectives=self.objectives,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_outcomes_agent.b.GenerateLearningOutcomes",
        new_callable=AsyncMock,
    )
    async def test_generate_propagates_exception(
        self,
        mock_generate,
    ):
        mock_generate.side_effect = RuntimeError("BAML generation failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "BAML generation failed",
        ):
            await self.agent.generate(
                objectives=self.objectives,
                sustainability_map=self.sustainability_map,
                metadata=self.metadata,
            )
