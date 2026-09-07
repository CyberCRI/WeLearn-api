import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.baml_client.types import SessionMode
from src.app.tutor.service.b_agents.activity_guide_generator_agent import (
    ActivityGuideGeneratorAgent,
)


class TestActivityGuideGeneratorAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = ActivityGuideGeneratorAgent()

        # Replace these with the required fields for your actual BAML types.
        self.activity_template = types.ActivityTemplate(
            name="Débat sur le développement durable",
            id="fakeid",
            description="fake description",
            compatible_types=["fake types"],
            compatible_modes=[SessionMode.PRESENTIEL],
            min_duration=1.5,
            max_duration=3,
            min_size=20,
            max_size=30,
            estimated_duration=2,
        )

        self.session = types.SessionPlan(
            session_number=2,
            objectives=[1],
            type="fake type",
            mode=SessionMode.PRESENTIEL,
            duration=1.5,
            class_size=20,
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

        self.activity_guide = types.ActivityGuide(
            activity_name="Débat sur le développement durable",
            description="Une activité de débat.",
            learning_objectives=[1],
            steps=[
                "Former les groupes",
                "Préparer les arguments",
                "Réaliser le débat",
            ],
            teacher_role="Faciliter le débat",
            student_role="Participer au débat",
            resources_needed=["Documents de référence"],
            timing_breakdown={"toto": 30},
            evaluation_method="Évaluation des arguments",
            sustainability_integration="Lien avec le développement durable",
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_guide_generator_agent.b.GenerateActivityGuide",
        new_callable=AsyncMock,
    )
    async def test_generate_calls_baml(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.activity_guide

        result = await self.agent.generate(
            activity_template=self.activity_template,
            session=self.session,
            metadata=self.metadata,
            sustainability_map=self.sustainability_map,
            output_language="Français",
        )

        mock_generate.assert_awaited_once_with(
            activity_template=self.activity_template,
            session=self.session,
            metadata=self.metadata,
            sustainability_map=self.sustainability_map,
            output_language="Français",
        )

        self.assertEqual(result, self.activity_guide)

    @patch(
        "src.app.tutor.service.b_agents.activity_guide_generator_agent.b.GenerateActivityGuide",
        new_callable=AsyncMock,
    )
    async def test_generate_uses_default_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.activity_guide

        await self.agent.generate(
            activity_template=self.activity_template,
            session=self.session,
            metadata=self.metadata,
            sustainability_map=self.sustainability_map,
        )

        mock_generate.assert_awaited_once_with(
            activity_template=self.activity_template,
            session=self.session,
            metadata=self.metadata,
            sustainability_map=self.sustainability_map,
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_guide_generator_agent.b.GenerateActivityGuide",
        new_callable=AsyncMock,
    )
    async def test_generate_returns_activity_guide(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.activity_guide

        result = await self.agent.generate(
            activity_template=self.activity_template,
            session=self.session,
            metadata=self.metadata,
            sustainability_map=self.sustainability_map,
        )

        self.assertIsInstance(result, types.ActivityGuide)
        self.assertEqual(
            result.activity_name,
            self.activity_guide.activity_name,
        )
        self.assertEqual(
            result.description,
            self.activity_guide.description,
        )
        self.assertEqual(
            result.learning_objectives,
            self.activity_guide.learning_objectives,
        )
        self.assertEqual(
            result.steps,
            self.activity_guide.steps,
        )
        self.assertEqual(
            result.teacher_role,
            self.activity_guide.teacher_role,
        )
        self.assertEqual(
            result.student_role,
            self.activity_guide.student_role,
        )
        self.assertEqual(
            result.resources_needed,
            self.activity_guide.resources_needed,
        )
        self.assertEqual(
            result.timing_breakdown,
            self.activity_guide.timing_breakdown,
        )
        self.assertEqual(
            result.evaluation_method,
            self.activity_guide.evaluation_method,
        )
        self.assertEqual(
            result.sustainability_integration,
            self.activity_guide.sustainability_integration,
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_guide_generator_agent.b.GenerateActivityGuide",
        new_callable=AsyncMock,
    )
    async def test_generate_with_custom_language(
        self,
        mock_generate,
    ):
        mock_generate.return_value = self.activity_guide

        await self.agent.generate(
            activity_template=self.activity_template,
            session=self.session,
            metadata=self.metadata,
            sustainability_map=self.sustainability_map,
            output_language="English",
        )

        mock_generate.assert_awaited_once_with(
            activity_template=self.activity_template,
            session=self.session,
            metadata=self.metadata,
            sustainability_map=self.sustainability_map,
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.activity_guide_generator_agent.b.GenerateActivityGuide",
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
                activity_template=self.activity_template,
                session=self.session,
                metadata=self.metadata,
                sustainability_map=self.sustainability_map,
            )
