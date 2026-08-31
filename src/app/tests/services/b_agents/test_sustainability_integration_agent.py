import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.baml_client.types import SessionMode
from src.app.tutor.service.b_agents.sustainability_integration_agent import (
    SustainabilityIntegrationAgent,
)


class TestSustainabilityIntegrationAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = SustainabilityIntegrationAgent()

        self.description = (
            "This course introduces students to the fundamentals "
            "of computer science."
        )

        self.objectives = types.LearningObjectives(
            objectives=[
                types.LearningObjective(
                    number=1,
                    text="Understand the fundamentals of computer science.",
                    bloom_level="Understand",
                ),
                types.LearningObjective(
                    number=2,
                    text="Apply fundamental computer science concepts.",
                    bloom_level="Apply",
                ),
            ]
        )

        self.sdg_resources = [
            types.Document(
                text="Sustainable development resource",
                corpus="toto",
                description="fake description",
                sdg=[4],
                url="fake url",
                title="Sustainable Development Goal 4",
            ),
            types.Document(
                text="Climate action resource",
                corpus="toto",
                description="fake description",
                sdg=[13],
                url="fake url 2",
                title="Climate Action",
            ),
        ]

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

        self.sustainability_integration = types.SustainabilityIntegration(
            connections=[
                types.SustainabilityConnection(
                    objective_number=1,
                    sdg_themes=[
                        "Quality Education",
                        "Sustainable Development",
                    ],
                    connection_explanation=(
                        "This objective connects to sustainable "
                        "development through education."
                    ),
                    key_resources=[
                        self.sdg_resources[0],
                    ],
                ),
                types.SustainabilityConnection(
                    objective_number=2,
                    sdg_themes=[
                        "Climate Action",
                    ],
                    connection_explanation=(
                        "This objective supports climate awareness."
                    ),
                    key_resources=[
                        self.sdg_resources[1],
                    ],
                ),
            ],
            suggested_objectives=[
                types.LearningObjective(
                    number=3,
                    text="Evaluate the environmental impact of technology.",
                    bloom_level="Evaluate",
                ),
            ],
            integration_strategy=(
                "Integrate sustainability concepts throughout " "the course."
            ),
            resources_used=self.sdg_resources,
        )

    @patch(
        "src.app.tutor.service.b_agents.sustainability_integration_agent.b.IntegrateSustainability",
        new_callable=AsyncMock,
    )
    async def test_integrate_calls_baml(
        self,
        mock_integrate,
    ):
        mock_integrate.return_value = self.sustainability_integration

        result = await self.agent.integrate(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
            output_language="Français",
        )

        mock_integrate.assert_awaited_once_with(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
            output_language="Français",
        )

        self.assertEqual(
            result,
            self.sustainability_integration,
        )

    @patch(
        "src.app.tutor.service.b_agents.sustainability_integration_agent.b.IntegrateSustainability",
        new_callable=AsyncMock,
    )
    async def test_integrate_uses_default_language(
        self,
        mock_integrate,
    ):
        mock_integrate.return_value = self.sustainability_integration

        await self.agent.integrate(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
        )

        mock_integrate.assert_awaited_once_with(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.sustainability_integration_agent.b.IntegrateSustainability",
        new_callable=AsyncMock,
    )
    async def test_integrate_returns_sustainability_integration(
        self,
        mock_integrate,
    ):
        mock_integrate.return_value = self.sustainability_integration

        result = await self.agent.integrate(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
        )

        self.assertIsInstance(
            result,
            types.SustainabilityIntegration,
        )

        self.assertEqual(
            len(result.connections),
            len(self.sustainability_integration.connections),
        )

        for result_connection, expected_connection in zip(
            result.connections,
            self.sustainability_integration.connections,
        ):
            self.assertEqual(
                result_connection.objective_number,
                expected_connection.objective_number,
            )

            self.assertEqual(
                result_connection.sdg_themes,
                expected_connection.sdg_themes,
            )

            self.assertEqual(
                result_connection.connection_explanation,
                expected_connection.connection_explanation,
            )

            self.assertEqual(
                result_connection.key_resources,
                expected_connection.key_resources,
            )

        self.assertEqual(
            result.suggested_objectives,
            self.sustainability_integration.suggested_objectives,
        )

        self.assertEqual(
            result.integration_strategy,
            self.sustainability_integration.integration_strategy,
        )

        self.assertEqual(
            result.resources_used,
            self.sustainability_integration.resources_used,
        )

    @patch(
        "src.app.tutor.service.b_agents.sustainability_integration_agent.b.IntegrateSustainability",
        new_callable=AsyncMock,
    )
    async def test_integrate_with_custom_language(
        self,
        mock_integrate,
    ):
        mock_integrate.return_value = self.sustainability_integration

        await self.agent.integrate(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
            output_language="English",
        )

        mock_integrate.assert_awaited_once_with(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.sustainability_integration_agent.b.IntegrateSustainability",
        new_callable=AsyncMock,
    )
    async def test_integrate_with_no_suggested_objectives(
        self,
        mock_integrate,
    ):
        mock_integrate.return_value = types.SustainabilityIntegration(
            connections=[
                types.SustainabilityConnection(
                    objective_number=1,
                    sdg_themes=["Quality Education"],
                    connection_explanation="fake explanation",
                    key_resources=[
                        self.sdg_resources[0],
                    ],
                ),
            ],
            suggested_objectives=None,
            integration_strategy="fake strategy",
            resources_used=self.sdg_resources,
        )

        result = await self.agent.integrate(
            description=self.description,
            objectives=self.objectives,
            sdg_resources=self.sdg_resources,
            metadata=self.metadata,
            mode="mode_1",
        )

        self.assertIsNone(
            result.suggested_objectives,
        )

    @patch(
        "src.app.tutor.service.b_agents.sustainability_integration_agent.b.IntegrateSustainability",
        new_callable=AsyncMock,
    )
    async def test_integrate_propagates_exception(
        self,
        mock_integrate,
    ):
        mock_integrate.side_effect = RuntimeError("BAML integration failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "BAML integration failed",
        ):
            await self.agent.integrate(
                description=self.description,
                objectives=self.objectives,
                sdg_resources=self.sdg_resources,
                metadata=self.metadata,
                mode="mode_1",
            )
