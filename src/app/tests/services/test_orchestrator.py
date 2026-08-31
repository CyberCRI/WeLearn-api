"""Unit tests for src.app.tutor.orchestrator.

Run from the project root:
    python -m unittest -v unittest.py

The tests mock every agent constructed by SyllabusOrchestrator, so no LLM/API
calls are made.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import src.app.tutor.service.orchestrator as orch
from src.app.baml_client.types import CourseMetadata, SessionMode
from src.app.tutor.service.models import UserInput


class FakeAgent:
    def __init__(self, total_tokens=0):
        self.total_tokens = total_tokens


class FakeCourseDescriptionAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.generate = AsyncMock(
            return_value=SimpleNamespace(
                text="A test course description.",
                word_count=5,
            )
        )


class FakeLearningObjectivesAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.generate = AsyncMock(
            return_value=SimpleNamespace(
                objectives=[
                    SimpleNamespace(text="Understand the core concepts."),
                    SimpleNamespace(text="Apply the concepts in practice."),
                ]
            )
        )


class FakeSustainabilityAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.integrate = AsyncMock(
            return_value=SimpleNamespace(suggested_objectives=[])
        )


class FakeLearningOutcomesAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.generate = AsyncMock(
            return_value=SimpleNamespace(
                outcomes=[SimpleNamespace(text="Demonstrate understanding.")]
            )
        )


class FakeCompetencyMappingAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.map_competencies = AsyncMock(
            return_value=SimpleNamespace(
                mappings=[
                    SimpleNamespace(
                        code="C4",
                        name="Systems thinking",
                    )
                ]
            )
        )


class FakePedagogicalEngineerAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.validate = AsyncMock(
            return_value=SimpleNamespace(
                passed=True,
                severity=None,
                suggestions=[],
                issues=[],
            )
        )


class TestSyllabusOrchestrator(unittest.IsolatedAsyncioTestCase):
    """Unit tests with all external agents replaced by mocks."""

    def setUp(self):
        self.description_agent = FakeCourseDescriptionAgent()
        self.objectives_agent = FakeLearningObjectivesAgent()
        self.sustainability_agent = FakeSustainabilityAgent()
        self.outcomes_agent = FakeLearningOutcomesAgent()
        self.competency_agent = FakeCompetencyMappingAgent()
        self.pedagogical_agent = FakePedagogicalEngineerAgent()

        # These are also constructed by SyllabusOrchestrator.__init__.
        self.engineer_agent = FakeAgent()
        self.recommendation_agent = FakeAgent()
        self.guide_agent = FakeAgent()

        self.patches = [
            patch.object(
                orch,
                "CourseDescriptionAgent",
                return_value=self.description_agent,
            ),
            patch.object(
                orch,
                "LearningObjectivesAgent",
                return_value=self.objectives_agent,
            ),
            patch.object(
                orch,
                "SustainabilityIntegrationAgent",
                return_value=self.sustainability_agent,
            ),
            patch.object(
                orch,
                "LearningOutcomesAgent",
                return_value=self.outcomes_agent,
            ),
            patch.object(
                orch,
                "CompetencyMappingAgent",
                return_value=self.competency_agent,
            ),
            patch.object(
                orch,
                "PedagogicalEngineerAgent",
                return_value=self.pedagogical_agent,
            ),
            patch.object(
                orch,
                "ActivityRecommendationAgent",
                return_value=self.recommendation_agent,
            ),
            patch.object(
                orch,
                "ActivityGuideGeneratorAgent",
                return_value=self.guide_agent,
            ),
        ]

        for patcher in self.patches:
            patcher.start()

        self.addCleanup(self.stop_patches)
        self.sut = orch.SyllabusOrchestrator()

    def stop_patches(self):
        for patcher in reversed(self.patches):
            patcher.stop()

    def make_metadata(self):
        return CourseMetadata(
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

    def make_user_input(
        self,
        *,
        mode=None,
        documents=None,
        provided_description=None,
        provided_objectives=None,
        objective_processing=None,
    ):
        return UserInput(
            mode=mode,
            documents=documents,
            provided_description=provided_description,
            provided_objectives=provided_objectives,
            objective_processing=objective_processing,
            metadata=self.make_metadata(),
            rag_resources=[],
        )

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    def test_detect_mode_documents_only(self):
        user_input = self.make_user_input(documents=["syllabus.pdf"])
        self.assertEqual(
            self.sut._detect_mode(user_input),
            "mode_1",
        )

    def test_detect_mode_documents_with_description(self):
        user_input = self.make_user_input(
            documents=["syllabus.pdf"],
            provided_description="Existing description",
        )
        self.assertEqual(
            self.sut._detect_mode(user_input),
            "mode_2a",
        )

    def test_detect_mode_objectives_augment(self):
        user_input = self.make_user_input(
            provided_objectives=["Objective 1"],
            objective_processing="augment",
        )
        self.assertEqual(
            self.sut._detect_mode(user_input),
            "mode_2b_augment",
        )

    def test_detect_mode_objectives_transform(self):
        user_input = self.make_user_input(
            provided_objectives=["Objective 1"],
            objective_processing="transform",
        )
        self.assertEqual(
            self.sut._detect_mode(user_input),
            "mode_2b_transform",
        )

    def test_detect_mode_metadata_only(self):
        user_input = self.make_user_input()
        self.assertEqual(
            self.sut._detect_mode(user_input),
            "mode_3",
        )

    def test_explicit_mode_has_priority(self):
        user_input = self.make_user_input(
            mode="custom_mode",
            documents=["syllabus.pdf"],
        )
        self.assertEqual(
            self.sut._detect_mode(user_input),
            "custom_mode",
        )

    # ------------------------------------------------------------------
    # Phase 0
    # ------------------------------------------------------------------

    def test_start_initializes_state_and_context(self):
        user_input = self.make_user_input(
            documents=["syllabus.pdf"],
            provided_description="Provided description",
        )

        state = self.sut.start(user_input)

        self.assertIs(state, self.sut.state)
        self.assertEqual(state.mode, "mode_2a")
        self.assertIs(state.metadata, user_input.metadata)
        self.assertEqual(state.rag_resources, [])
        self.assertEqual(
            state.provided_objectives,
            user_input.provided_objectives,
        )
        self.assertEqual(state.current_phase, orch.Phase.INPUT)
        self.assertEqual(
            self.sut._context_text,
            "Provided description",
        )

    def test_start_without_description_uses_empty_context(self):
        self.sut.start(self.make_user_input())
        self.assertEqual(self.sut._context_text, "")

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    def test_get_total_tokens_sums_all_agents(self):
        agents = [
            self.sut.course_description_agent,
            self.sut.learning_objectives_agent,
            self.sut.sustainability_agent,
            self.sut.learning_outcomes_agent,
            self.sut.competency_mapping_agent,
            self.sut.pedagogical_engineer_agent,
            self.sut.activity_recommendation_agent,
            self.sut.activity_guide_generator_agent,
        ]

        for index, agent in enumerate(agents, start=1):
            agent.total_tokens = index * 10

        self.assertEqual(
            self.sut.get_total_tokens(),
            sum(range(10, 90, 10)),
        )

    # ------------------------------------------------------------------
    # Phase 1: individual phases
    # ------------------------------------------------------------------

    async def test_generate_description(self):
        self.sut.start(self.make_user_input())

        result = await self.sut.phase_1_generate_description("test context")

        self.assertEqual(
            result.text,
            "A test course description.",
        )
        self.assertIs(self.sut.state.description, result)
        self.assertEqual(
            self.sut.state.current_phase,
            orch.Phase.DESCRIPTION,
        )
        self.description_agent.generate.assert_awaited_once()

        kwargs = self.description_agent.generate.await_args.kwargs
        self.assertEqual(kwargs["context_text"], "test context")

    async def test_generate_objectives(self):
        self.sut.start(self.make_user_input())
        self.sut.state.description = SimpleNamespace(text="Course description")

        result = await self.sut.phase_1_generate_objectives("context")

        self.assertEqual(len(result.objectives), 2)
        self.assertIs(self.sut.state.objectives, result)
        self.assertEqual(
            self.sut.state.current_phase,
            orch.Phase.OBJECTIVES,
        )

    async def test_integrate_sustainability_augment_merges_objectives(self):
        self.sut.start(
            self.make_user_input(
                provided_objectives=["Original"],
                objective_processing="augment",
            )
        )

        original = SimpleNamespace(text="Original")
        suggested = SimpleNamespace(text="Suggested")

        self.sut.state.description = SimpleNamespace(text="Description")
        self.sut.state.objectives = SimpleNamespace(objectives=[original])

        self.sustainability_agent.integrate.return_value = SimpleNamespace(
            suggested_objectives=[suggested]
        )

        result = await self.sut.phase_1_integrate_sustainability()

        self.assertIs(
            self.sut.state.sustainability_integration,
            result,
        )
        self.assertEqual(
            self.sut.state.objectives.objectives,
            [original, suggested],
        )

    async def test_integrate_sustainability_transform_does_not_merge(self):
        self.sut.start(
            self.make_user_input(
                provided_objectives=["Original"],
                objective_processing="transform",
            )
        )

        original = SimpleNamespace(text="Transformed")
        suggested = SimpleNamespace(text="Extra")

        self.sut.state.description = SimpleNamespace(text="Description")
        self.sut.state.objectives = SimpleNamespace(objectives=[original])

        self.sustainability_agent.integrate.return_value = SimpleNamespace(
            suggested_objectives=[suggested]
        )

        await self.sut.phase_1_integrate_sustainability()

        self.assertEqual(
            self.sut.state.objectives.objectives,
            [original],
        )

    async def test_generate_outcomes(self):
        self.sut.start(self.make_user_input())
        self.sut.state.objectives = SimpleNamespace(
            objectives=[SimpleNamespace(text="Objective")]
        )
        self.sut.state.sustainability_integration = SimpleNamespace()

        result = await self.sut.phase_1_generate_outcomes()

        self.assertEqual(len(result.outcomes), 1)
        self.assertIs(self.sut.state.outcomes, result)
        self.assertEqual(
            self.sut.state.current_phase,
            orch.Phase.OUTCOMES,
        )

    async def test_map_competencies(self):
        self.sut.start(self.make_user_input())
        self.sut.state.outcomes = SimpleNamespace(outcomes=[])

        result = await self.sut.phase_1_map_competencies()

        self.assertEqual(len(result.mappings), 1)
        self.assertIs(self.sut.state.competencies, result)
        self.assertEqual(
            self.sut.state.current_phase,
            orch.Phase.COMPETENCIES,
        )

        kwargs = self.competency_agent.map_competencies.await_args.kwargs
        self.assertEqual(
            set(kwargs["greencomp_framework"]),
            {
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "C8",
                "C9",
                "C10",
                "C11",
                "C12",
            },
        )

    async def test_validate_framework_success(self):
        self.sut.start(self.make_user_input())
        self.sut.state.description = SimpleNamespace(text="Description")
        self.sut.state.objectives = SimpleNamespace(objectives=[])
        self.sut.state.outcomes = SimpleNamespace(outcomes=[])
        self.sut.state.competencies = SimpleNamespace(mappings=[])
        self.sut.state.sustainability_integration = SimpleNamespace()

        result = await self.sut.phase_1_validate_framework()

        self.assertTrue(result)
        self.assertEqual(
            self.sut.state.current_phase,
            orch.Phase.VALIDATION,
        )
        self.assertEqual(
            len(self.sut.state.validation_history),
            1,
        )

    async def test_validate_framework_minor_failure_does_not_block(self):
        self.sut.start(self.make_user_input())
        self.sut.state.description = SimpleNamespace(text="Description")
        self.sut.state.objectives = SimpleNamespace(objectives=[])
        self.sut.state.outcomes = SimpleNamespace(outcomes=[])
        self.sut.state.competencies = SimpleNamespace(mappings=[])
        self.sut.state.sustainability_integration = SimpleNamespace()

        self.pedagogical_agent.validate.return_value = SimpleNamespace(
            passed=False,
            severity="minor",
            suggestions=["Fix wording"],
            issues=[],
        )

        self.assertTrue(await self.sut.phase_1_validate_framework())

    async def test_validate_framework_major_failure_does_not_block(self):
        self.sut.start(self.make_user_input())
        self.sut.state.description = SimpleNamespace(text="Description")
        self.sut.state.objectives = SimpleNamespace(objectives=[])
        self.sut.state.outcomes = SimpleNamespace(outcomes=[])
        self.sut.state.competencies = SimpleNamespace(mappings=[])
        self.sut.state.sustainability_integration = SimpleNamespace()

        self.pedagogical_agent.validate.return_value = SimpleNamespace(
            passed=False,
            severity="major",
            suggestions=[],
            issues=["Major issue"],
        )

        self.assertTrue(await self.sut.phase_1_validate_framework())

    # ------------------------------------------------------------------
    # Complete Phase 1
    # ------------------------------------------------------------------

    async def test_run_phase_1_executes_steps_in_order(self):
        self.sut.start(self.make_user_input())

        calls = []

        async def description(context):
            calls.append("description")
            self.sut.state.description = SimpleNamespace(
                text="Description",
                word_count=1,
            )

        async def objectives(context):
            calls.append("objectives")
            self.sut.state.objectives = SimpleNamespace(objectives=[])

        async def sustainability():
            calls.append("sustainability")
            self.sut.state.sustainability_integration = SimpleNamespace(
                suggested_objectives=[]
            )

        async def outcomes():
            calls.append("outcomes")
            self.sut.state.outcomes = SimpleNamespace(outcomes=[])

        async def competencies():
            calls.append("competencies")
            self.sut.state.competencies = SimpleNamespace(mappings=[])

        with patch.object(
            self.sut,
            "phase_1_generate_description",
            side_effect=description,
        ), patch.object(
            self.sut,
            "phase_1_generate_objectives",
            side_effect=objectives,
        ), patch.object(
            self.sut,
            "phase_1_integrate_sustainability",
            side_effect=sustainability,
        ), patch.object(
            self.sut,
            "phase_1_generate_outcomes",
            side_effect=outcomes,
        ), patch.object(
            self.sut,
            "phase_1_map_competencies",
            side_effect=competencies,
        ):
            result = await self.sut.run_phase_1("context")

        self.assertTrue(result)
        self.assertEqual(
            calls,
            [
                "description",
                "objectives",
                "sustainability",
                "outcomes",
                "competencies",
            ],
        )

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def test_distribute_sessions(self):
        self.sut.start(self.make_user_input())

        objectives = [SimpleNamespace(text="Objective")]
        self.sut.state.objectives = SimpleNamespace(objectives=objectives)

        expected_sessions = [
            SimpleNamespace(session_number=1),
            SimpleNamespace(session_number=2),
        ]

        with patch.object(
            orch,
            "distribute_objectives",
            return_value=expected_sessions,
        ) as distribute:
            result = self.sut.phase_2_distribute_sessions()

        self.assertIs(result, expected_sessions)
        self.assertIs(
            self.sut.state.sessions,
            expected_sessions,
        )
        self.assertEqual(
            self.sut.state.current_phase,
            orch.Phase.SESSIONS,
        )

        distribute.assert_called_once_with(
            objectives=objectives,
            metadata=self.sut.state.metadata,
        )

    # ------------------------------------------------------------------
    # Complete orchestration
    # ------------------------------------------------------------------

    async def test_run_completes_successfully(self):
        user_input = self.make_user_input(provided_description="Course context")

        with patch.object(
            self.sut,
            "run_phase_1",
            new=AsyncMock(return_value=True),
        ) as run_phase_1:
            result = await self.sut.run(user_input)

        run_phase_1.assert_awaited_once_with("Course context")
        self.assertEqual(
            self.sut.state.current_phase,
            orch.Phase.COMPLETE,
        )

        for key in (
            "description",
            "objectives",
            "outcomes",
            "competencies",
            "sustainability",
            "sessions",
        ):
            self.assertIn(key, result)

    async def test_run_raises_when_phase_1_fails(self):
        with patch.object(
            self.sut,
            "run_phase_1",
            new=AsyncMock(return_value=False),
        ):
            with self.assertRaises(Exception):
                await self.sut.run(self.make_user_input())

    def test_get_output_before_start_raises(self):
        with self.assertRaises(Exception):
            self.sut.get_output()
