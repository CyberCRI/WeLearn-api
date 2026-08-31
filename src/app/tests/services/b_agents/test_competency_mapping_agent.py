import unittest
from unittest.mock import AsyncMock, patch

from src.app.baml_client.async_client import types
from src.app.tutor.service.b_agents.competency_mapping_agent import (
    CompetencyMappingAgent,
)


class TestCompetencyMappingAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = CompetencyMappingAgent()

        self.outcomes = types.LearningOutcomes(
            outcomes=[
                types.LearningOutcome(
                    number=1,
                    text="Understand the fundamentals of computer science.",
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

        self.greencomp_framework = {
            "Critical thinking": "Ability to analyze and evaluate information",
            "Collaboration": "Ability to work effectively with others",
            "Systems thinking": "Ability to understand complex systems",
        }

        self.competency_mappings = types.CompetencyMappings(
            mappings=[
                types.CompetencyMapping(
                    outcome_number=1,
                    greencomp_competencies=[
                        "Critical thinking",
                        "Systems thinking",
                    ],
                    rationale=(
                        "This outcome requires students to analyze "
                        "and evaluate complex information."
                    ),
                ),
                types.CompetencyMapping(
                    outcome_number=2,
                    greencomp_competencies=[
                        "Collaboration",
                        "Critical thinking",
                    ],
                    rationale=(
                        "This outcome requires students to apply concepts "
                        "and collaborate on practical problems."
                    ),
                ),
            ]
        )

    @patch(
        "src.app.tutor.service.b_agents.competency_mapping_agent.b.MapCompetencies",
        new_callable=AsyncMock,
    )
    async def test_map_competencies_calls_baml(
        self,
        mock_map,
    ):
        mock_map.return_value = self.competency_mappings

        result = await self.agent.map_competencies(
            outcomes=self.outcomes,
            greencomp_framework=self.greencomp_framework,
            output_language="Français",
        )

        mock_map.assert_awaited_once_with(
            outcomes=self.outcomes,
            greencomp_framework=self.greencomp_framework,
            output_language="Français",
        )

        self.assertEqual(
            result,
            self.competency_mappings,
        )

    @patch(
        "src.app.tutor.service.b_agents.competency_mapping_agent.b.MapCompetencies",
        new_callable=AsyncMock,
    )
    async def test_map_competencies_uses_default_language(
        self,
        mock_map,
    ):
        mock_map.return_value = self.competency_mappings

        await self.agent.map_competencies(
            outcomes=self.outcomes,
            greencomp_framework=self.greencomp_framework,
        )

        mock_map.assert_awaited_once_with(
            outcomes=self.outcomes,
            greencomp_framework=self.greencomp_framework,
            output_language="Français",
        )

    @patch(
        "src.app.tutor.service.b_agents.competency_mapping_agent.b.MapCompetencies",
        new_callable=AsyncMock,
    )
    async def test_map_competencies_returns_competency_mappings(
        self,
        mock_map,
    ):
        mock_map.return_value = self.competency_mappings

        result = await self.agent.map_competencies(
            outcomes=self.outcomes,
            greencomp_framework=self.greencomp_framework,
        )

        self.assertIsInstance(
            result,
            types.CompetencyMappings,
        )

        self.assertEqual(
            len(result.mappings),
            len(self.competency_mappings.mappings),
        )

        for result_mapping, expected_mapping in zip(
            result.mappings,
            self.competency_mappings.mappings,
        ):
            self.assertEqual(
                result_mapping.outcome_number,
                expected_mapping.outcome_number,
            )

            self.assertEqual(
                result_mapping.greencomp_competencies,
                expected_mapping.greencomp_competencies,
            )

            self.assertEqual(
                result_mapping.rationale,
                expected_mapping.rationale,
            )

    @patch(
        "src.app.tutor.service.b_agents.competency_mapping_agent.b.MapCompetencies",
        new_callable=AsyncMock,
    )
    async def test_map_competencies_with_custom_language(
        self,
        mock_map,
    ):
        mock_map.return_value = self.competency_mappings

        await self.agent.map_competencies(
            outcomes=self.outcomes,
            greencomp_framework=self.greencomp_framework,
            output_language="English",
        )

        mock_map.assert_awaited_once_with(
            outcomes=self.outcomes,
            greencomp_framework=self.greencomp_framework,
            output_language="English",
        )

    @patch(
        "src.app.tutor.service.b_agents.competency_mapping_agent.b.MapCompetencies",
        new_callable=AsyncMock,
    )
    async def test_map_competencies_propagates_exception(
        self,
        mock_map,
    ):
        mock_map.side_effect = RuntimeError("BAML competency mapping failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "BAML competency mapping failed",
        ):
            await self.agent.map_competencies(
                outcomes=self.outcomes,
                greencomp_framework=self.greencomp_framework,
            )
