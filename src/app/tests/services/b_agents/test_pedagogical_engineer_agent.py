import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.baml_client.types import SessionMode
from src.app.tutor.service.b_agents.pedagogical_engineer_agent import (
    PedagogicalEngineerAgent,
)


class TestPedagogicalEngineerAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = PedagogicalEngineerAgent()

        self.description = (
            "This course introduces students to the fundamentals "
            "of computer science."
        )

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
            ]
        )

        self.outcomes = types.LearningOutcomes(
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
            ]
        )

        self.competencies = types.CompetencyMappings(
            mappings=[
                types.CompetencyMapping(
                    outcome_number=1,
                    greencomp_competencies=["fake comp"],
                    rationale="fake rationale",
                )
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

        self.validation_report = types.ValidationReport(
            passed=True,
            issues=["fake issue"],
            suggestions=["fake suggestion"],
            severity="low",
        )

    @patch(
        "src.app.tutor.service.b_agents.pedagogical_engineer_agent.b.ValidatePedagogicalFramework",
        new_callable=AsyncMock,
    )
    async def test_validate_calls_baml(
        self,
        mock_validate,
    ):
        mock_validate.return_value = self.validation_report

        result = await self.agent.validate(
            description=self.description,
            objectives=self.objectives,
            outcomes=self.outcomes,
            competencies=self.competencies,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="Français",
        )

        mock_validate.assert_awaited_once_with(
            description=self.description,
            objectives=self.objectives,
            outcomes=self.outcomes,
            competencies=self.competencies,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="Français",
        )

        self.assertEqual(
            result,
            self.validation_report,
        )

    @patch(
        "src.app.tutor.service.b_agents.pedagogical_engineer_agent.b.ValidatePedagogicalFramework",
        new_callable=AsyncMock,
    )
    async def test_validate_uses_default_language(
        self,
        mock_validate,
    ):
        mock_validate.return_value = self.validation_report

        await self.agent.validate(
            description=self.description,
            objectives=self.objectives,
            outcomes=self.outcomes,
            competencies=self.competencies,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
        )

        mock_validate.assert_awaited_once_with(
            description=self.description,
            objectives=self.objectives,
            outcomes=self.outcomes,
            competencies=self.competencies,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.pedagogical_engineer_agent.b.ValidatePedagogicalFramework",
        new_callable=AsyncMock,
    )
    async def test_validate_returns_validation_report(
        self,
        mock_validate,
    ):
        mock_validate.return_value = self.validation_report

        result = await self.agent.validate(
            description=self.description,
            objectives=self.objectives,
            outcomes=self.outcomes,
            competencies=self.competencies,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
        )

        self.assertIsInstance(
            result,
            types.ValidationReport,
        )

        self.assertEqual(
            result.passed,
            self.validation_report.passed,
        )

        self.assertEqual(
            result.issues,
            self.validation_report.issues,
        )

        self.assertEqual(
            result.suggestions,
            self.validation_report.suggestions,
        )

        self.assertEqual(
            result.severity,
            self.validation_report.severity,
        )

    @patch(
        "src.app.tutor.service.b_agents.pedagogical_engineer_agent.b.ValidatePedagogicalFramework",
        new_callable=AsyncMock,
    )
    async def test_validate_with_custom_language(
        self,
        mock_validate,
    ):
        mock_validate.return_value = self.validation_report

        await self.agent.validate(
            description=self.description,
            objectives=self.objectives,
            outcomes=self.outcomes,
            competencies=self.competencies,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="English",
        )

        mock_validate.assert_awaited_once_with(
            description=self.description,
            objectives=self.objectives,
            outcomes=self.outcomes,
            competencies=self.competencies,
            sustainability_map=self.sustainability_map,
            metadata=self.metadata,
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.pedagogical_engineer_agent.b.ValidatePedagogicalFramework",
        new_callable=AsyncMock,
    )
    async def test_validate_propagates_exception(
        self,
        mock_validate,
    ):
        mock_validate.side_effect = RuntimeError("BAML validation failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "BAML validation failed",
        ):
            await self.agent.validate(
                description=self.description,
                objectives=self.objectives,
                outcomes=self.outcomes,
                competencies=self.competencies,
                sustainability_map=self.sustainability_map,
                metadata=self.metadata,
            )
