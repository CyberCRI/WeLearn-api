import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.baml_client.types import SessionMode
from src.app.tutor.service.b_agents.learning_objectives_agent import (
    LearningObjectivesAgent,
)


class TestLearningObjectivesAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = LearningObjectivesAgent()

        self.description = (
            "This course introduces students to the fundamentals "
            "of computer science."
        )

        self.context_text = "This is fake contextual text."

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

        self.learning_objectives = types.LearningObjectives(
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

    @patch(
        "src.app.tutor.service.b_agents.learning_objectives_agent.b.GenerateLearningObjectives",
        new_callable=AsyncMock,
    )
    async def test_generate_calls_baml(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_objectives

        result = await self.agent.generate(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
            output_language="Français",
        )

        mock_generate.assert_awaited_once_with(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
            output_language="Français",
        )

        self.assertEqual(
            result,
            self.learning_objectives,
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_objectives_agent.b.GenerateLearningObjectives",
        new_callable=AsyncMock,
    )
    async def test_generate_uses_default_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_objectives

        await self.agent.generate(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
        )

        mock_generate.assert_awaited_once_with(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_objectives_agent.b.GenerateLearningObjectives",
        new_callable=AsyncMock,
    )
    async def test_generate_returns_learning_objectives(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_objectives

        result = await self.agent.generate(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
        )

        self.assertIsInstance(
            result,
            types.LearningObjectives,
        )

        self.assertEqual(
            len(result.objectives),
            len(self.learning_objectives.objectives),
        )

        for result_objective, expected_objective in zip(
            result.objectives,
            self.learning_objectives.objectives,
        ):
            self.assertEqual(
                result_objective.number,
                expected_objective.number,
            )
            self.assertEqual(
                result_objective.text,
                expected_objective.text,
            )

    @patch(
        "src.app.tutor.service.b_agents.learning_objectives_agent.b.GenerateLearningObjectives",
        new_callable=AsyncMock,
    )
    async def test_generate_uses_sequential_numbering(
        self,
        mock_generate,
    ):
        mock_generate.return_value = types.LearningObjectives(
            objectives=[
                types.LearningObjective(
                    number=10,
                    text="First objective",
                ),
                types.LearningObjective(
                    number=25,
                    text="Second objective",
                ),
                types.LearningObjective(
                    number=99,
                    text="Third objective",
                ),
            ]
        )

        result = await self.agent.generate(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
        )

        self.assertEqual(
            [objective.number for objective in result.objectives],
            [1, 2, 3],
        )

        self.assertEqual(
            [objective.text for objective in result.objectives],
            [
                "First objective",
                "Second objective",
                "Third objective",
            ],
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_objectives_agent.b.GenerateLearningObjectives",
        new_callable=AsyncMock,
    )
    async def test_generate_with_custom_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.learning_objectives

        await self.agent.generate(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
            output_language="English",
        )

        mock_generate.assert_awaited_once_with(
            description=self.description,
            context_text=self.context_text,
            metadata=self.metadata,
            mode="mode_1",
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.learning_objectives_agent.b.GenerateLearningObjectives",
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
                description=self.description,
                context_text=self.context_text,
                metadata=self.metadata,
                mode="mode_1",
            )
