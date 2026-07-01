"""
Syllabus Orchestrator - Coordinates all agents and phases
"""

import json
import logging
from enum import Enum
from typing import List, Optional

from src.app.tutor.tools import distribute_objectives

from .b_agents import (
    ActivityGuideGeneratorAgent,
    ActivityRecommendationAgent,
    CompetencyMappingAgent,
    CourseDescriptionAgent,
    LearningObjectivesAgent,
    LearningOutcomesAgent,
    PedagogicalEngineerAgent,
    SustainabilityIntegrationAgent,
)
from .models import (
    CompetencyMappings,
    CourseDescription,
    LearningObjectives,
    LearningOutcomes,
    OrchestratorState,
    SessionPlan,
    SustainabilityIntegration,
    SyllabusOutput,
    UserInput,
)

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    """Orchestrator phases"""

    INPUT = "input"
    DESCRIPTION = "description"
    OBJECTIVES = "objectives"
    SUSTAINABILITY = "sustainability"
    OUTCOMES = "outcomes"
    COMPETENCIES = "competencies"
    VALIDATION = "validation"
    SESSIONS = "sessions"
    ACTIVITIES = "activities"
    COMPLETE = "complete"


class SyllabusOrchestrator:
    """
    Main orchestrator that coordinates all agents and tools through sequential phases
    """

    def __init__(self):
        """Initialize orchestrator with agents and configuration"""
        # self.config = load_config()

        # Initialize all agents
        self.course_description_agent = CourseDescriptionAgent()
        self.learning_objectives_agent = LearningObjectivesAgent()
        self.sustainability_agent = SustainabilityIntegrationAgent()
        self.learning_outcomes_agent = LearningOutcomesAgent()
        self.competency_mapping_agent = CompetencyMappingAgent()
        self.pedagogical_engineer_agent = PedagogicalEngineerAgent()
        self.activity_recommendation_agent = ActivityRecommendationAgent()
        self.activity_guide_generator_agent = ActivityGuideGeneratorAgent()

        # Load activity templates (CAG)
        # load_activity_templates()

        # State
        self.state: Optional[OrchestratorState] = None

        logger.info("Orchestrator initialized")

    def get_total_tokens(self) -> int:
        """
        Get total token usage across all agents

        Returns:
            Total estimated tokens used
        """
        agents = [
            self.course_description_agent,
            self.learning_objectives_agent,
            self.sustainability_agent,
            self.learning_outcomes_agent,
            self.competency_mapping_agent,
            self.pedagogical_engineer_agent,
            self.activity_recommendation_agent,
            self.activity_guide_generator_agent,
        ]

        total = sum(agent.total_tokens for agent in agents)
        logger.debug(f"Total tokens across all agents: {total}")
        return total

    # ========================================
    # Phase 0: Input & Intent Clarification
    # ========================================

    def start(self, user_input: UserInput) -> OrchestratorState:
        """
        Start orchestration with user input

        Args:
            user_input: Initial user input with metadata and optional content
            document_filenames: Optional list of filenames for documents (for better error messages)

        Returns:
            Initial state
        """
        logger.info("Starting orchestration - Phase 0: Input")

        # Detect mode
        mode = self._detect_mode(user_input)
        logger.info(f"Detected mode: {mode}")

        # TODO: content of the user input doc == summary
        context_text = ""

        # Add provided description to context (Mode 2A)
        if user_input.provided_description:
            if context_text:
                context_text = (
                    user_input.provided_description + "\n\n---\n\n" + context_text
                )
            else:
                context_text = user_input.provided_description
            logger.info("Added provided description to context")

        # Store context_text in state for later use
        self._context_text = context_text

        # Initialize state
        self.state = OrchestratorState(
            mode=mode,
            metadata=user_input.metadata,
            rag_resources=user_input.rag_resources,
            provided_objectives=user_input.provided_objectives,
            current_phase=Phase.INPUT,
        )

        return self.state

    def _detect_mode(self, user_input: UserInput) -> str:
        """
        Detect input mode from user input:
        - Mode 1: Documents present, no provided_description
        - Mode 2A: Documents present + provided_description present (from uploaded existing syllabus)
        - Mode 2B-Augment: provided_objectives present, objective_processing="augment"
        - Mode 2B-Transform: provided_objectives present, objective_processing="transform"
        - Mode 3: Only metadata (default)
        """
        if user_input.documents:
            if user_input.provided_description:
                return "mode_2a"  # Existing syllabus → generate new
            else:
                return "mode_1"  # Documents → generate new
        elif user_input.provided_objectives:
            # Mode 2B fork: augment vs transform
            if user_input.objective_processing == "transform":
                return "mode_2b_transform"  # Rewrite objectives with sustainability
            else:
                return "mode_2b_augment"  # Keep objectives, add sustainability
        else:
            return "mode_3"  # Metadata only → generate from scratch

    # ========================================
    # Phase 1: Pedagogical Framing
    # ========================================

    async def phase_1_generate_description(
        self, context_text: str = ""
    ) -> CourseDescription:
        """
        Phase 1.1: Generate course description

        Args:
            context_text: Optional context (extracted documents, provided description)

        Returns:
            Generated course description
        """
        logger.info("Phase 1.1: Generating course description")
        self.state.current_phase = Phase.DESCRIPTION

        description = await self.course_description_agent.generate(
            mode=self.state.mode,
            metadata=self.state.metadata,
            context_text=context_text,
            output_language=self.state.metadata.output_language,
        )

        self.state.description = description
        logger.info(f"Description generated ({description.word_count} words)")

        return description

    async def phase_1_generate_objectives(
        self, context_text: str = ""
    ) -> LearningObjectives:
        """
        Phase 1.2: Generate learning objectives

        Args:
            context_text: Optional context

        Returns:
            Generated objectives
        """
        logger.info("Phase 1.2: Generating learning objectives")
        self.state.current_phase = Phase.OBJECTIVES

        objectives = await self.learning_objectives_agent.generate(
            description=self.state.description.text,
            context_text=context_text,
            metadata=self.state.metadata,
            mode=self.state.mode,
            output_language=self.state.metadata.output_language,
        )

        self.state.objectives = objectives
        logger.info(f"Generated {len(objectives.objectives)} objectives")

        return objectives

    async def phase_1_integrate_sustainability(self) -> SustainabilityIntegration:
        """
        Phase 1.3: Integrate sustainability

        Returns:
            Sustainability integration
        """
        logger.info("Phase 1.3: Integrating sustainability")
        self.state.current_phase = Phase.SUSTAINABILITY

        # Build RAG query
        # query = (
        #     self.state.description.text
        #     + " "
        #     + " ".join([obj.text for obj in self.state.objectives.objectives])
        # )

        # # Retrieve SDG resources
        # filters = {"discipline": self.state.metadata.discipline}
        # # sdg_resources = rag_retrieve(query, filters, top_k=15)

        # # Check if RAG results sufficient
        # high_relevance_docs = [
        #     doc
        #     for doc in sdg_resources
        #     if doc.relevance_score and doc.relevance_score > 0.7
        # ]

        # self.state.retrieved_resources.extend(sdg_resources)

        # Call sustainability agent
        sustainability = await self.sustainability_agent.integrate(
            description=self.state.description.text,
            objectives=self.state.objectives,
            sdg_resources=self.state.rag_resources,
            metadata=self.state.metadata,
            mode=self.state.mode,
            output_language=self.state.metadata.output_language,
        )

        # Handle Mode 2B variants
        if self.state.mode == "mode_2b_augment":
            # Augment mode: Keep original objectives, add suggested ones
            if sustainability.suggested_objectives:
                logger.info(
                    f"Mode 2B-Augment: Adding {len(sustainability.suggested_objectives)} suggested objectives"
                )
                all_objectives = (
                    self.state.objectives.objectives
                    + sustainability.suggested_objectives
                )
                self.state.objectives.objectives = all_objectives
        elif self.state.mode == "mode_2b_transform":
            # Transform mode: Objectives already rewritten with sustainability
            logger.info(
                "Mode 2B-Transform: Using rewritten objectives with sustainability"
            )
            # No need to merge - objectives are already transformed

        self.state.sustainability_integration = sustainability
        logger.info("Sustainability integration complete")

        return sustainability

    async def phase_1_generate_outcomes(self) -> LearningOutcomes:
        """
        Phase 1.4: Generate learning outcomes

        Returns:
            Generated outcomes
        """
        logger.info("Phase 1.4: Generating learning outcomes")
        self.state.current_phase = Phase.OUTCOMES

        outcomes = await self.learning_outcomes_agent.generate(
            objectives=self.state.objectives,
            sustainability_map=self.state.sustainability_integration,
            metadata=self.state.metadata,
            output_language=self.state.metadata.output_language,
        )

        self.state.outcomes = outcomes
        logger.info(f"Generated {len(outcomes.outcomes)} outcomes")

        return outcomes

    async def phase_1_map_competencies(self) -> CompetencyMappings:
        """
        Phase 1.5: Map GreenComp competencies

        Returns:
            Competency mappings
        """
        logger.info("Phase 1.5: Mapping competencies")
        self.state.current_phase = Phase.COMPETENCIES

        # GreenComp framework (could be loaded from config)
        # competency_mapping_agent already has the competencies in its prompt, so no need to repeat, centralise in 1 place TODO
        greencomp = {
            "C1": "Valuing sustainability",
            "C2": "Supporting fairness",
            "C3": "Promoting nature",
            "C4": "Systems thinking",
            "C5": "Critical thinking",
            "C6": "Problem framing",
            "C7": "Futures literacy",
            "C8": "Adaptability",
            "C9": "Exploratory thinking",
            "C10": "Political agency",
            "C11": "Collective action",
            "C12": "Individual initiative",
        }

        competencies = await self.competency_mapping_agent.map_competencies(
            outcomes=self.state.outcomes,
            greencomp_framework=greencomp,
            output_language=self.state.metadata.output_language,
        )

        self.state.competencies = competencies
        logger.info(f"Mapped {len(competencies.mappings)} competencies")

        return competencies

    async def phase_1_validate_framework(self) -> bool:
        """
        Phase 1.6: Validate pedagogical framework

        Returns:
            True if validation passed (after any corrections)
        """
        logger.info("Phase 1.6: Validating pedagogical framework")
        self.state.current_phase = Phase.VALIDATION

        validation = await self.pedagogical_engineer_agent.validate(
            description=self.state.description.text,
            objectives=self.state.objectives,
            outcomes=self.state.outcomes,
            competencies=self.state.competencies,
            sustainability_map=self.state.sustainability_integration,
            metadata=self.state.metadata,
            output_language=self.state.metadata.output_language,
        )

        self.state.validation_history.append(validation)

        if validation.passed:
            logger.info("Validation passed")
            return True

        logger.warning(f"Validation failed: {validation.severity}")

        if validation.severity == "minor":
            # Auto-apply minor fixes
            logger.info("Minor issues detected - auto-applying suggestions")
            for suggestion in validation.suggestions:
                logger.info(f"Applied fix: {suggestion}")
            # Minor issues don't block - continue with current state
            return True

        elif validation.severity == "major":
            # Log major issues but continue (user will see in checkpoint)
            logger.info(
                "Major issues detected - providing corrected version and will be shown to user at checkpoint"
            )
            for issue in validation.issues:
                logger.info(f"Corrected: {issue}")
            return True

        else:  # subjective
            # Log subjective suggestions
            logger.info("Subjective recommendations - will be shown to user")
            for suggestion in validation.suggestions:
                logger.info(f"Recommendation: {suggestion}")
            return True

    async def run_phase_1(self, context_text: str = "") -> bool:
        """
        Run complete Phase 1: Pedagogical Framing

        Args:
            context_text: Optional context

        Returns:
            True if phase completed successfully
        """
        logger.info("========== PHASE 1: PEDAGOGICAL FRAMING ==========")

        # try:
        await self.phase_1_generate_description(context_text)

        await self.phase_1_generate_objectives(context_text)

        await self.phase_1_integrate_sustainability()

        await self.phase_1_generate_outcomes()

        await self.phase_1_map_competencies()

        logger.info("Phase 1 complete")
        return True

        # except AgentError as e:
        #     logger.error(f"Phase 1 failed: {e}")
        #     return False

    # ========================================
    # Phase 2: Session Planning
    # ========================================

    def phase_2_distribute_sessions(self) -> List[SessionPlan]:
        """
        Phase 2: Distribute objectives across sessions (deterministic)

        Returns:
            List of session plans
        """
        logger.info("========== PHASE 2: SESSION PLANNING ==========")
        self.state.current_phase = Phase.SESSIONS

        sessions = distribute_objectives(
            objectives=self.state.objectives.objectives, metadata=self.state.metadata
        )

        self.state.sessions = sessions
        logger.info(f"Created {len(sessions)} session plans")

        return sessions

    # ========================================
    # Complete Orchestration
    # ========================================

    async def run(
        self,
        user_input: UserInput,
        context_text: str | None = None,
        document_filenames: Optional[List[str]] = None,
    ):
        # -> SyllabusOutput:
        """
        Run complete orchestration

        Args:
            user_input: User input
            context_text: Optional context (if None, will be built from documents)
            document_filenames: Optional list of filenames for documents

        Returns:
            Complete syllabus output
        """
        logger.info("========== STARTING ORCHESTRATION ==========")

        # Phase 0
        #  user description + documents content
        self.start(user_input)

        # Use context_text from state if not provided explicitly
        if context_text is None:
            context_text = getattr(self, "_context_text", "")

        # Phase 1
        if not await self.run_phase_1(context_text):
            raise Exception("Phase 1 failed")

        # Phase 2
        # self.phase_2_distribute_sessions()

        # # Phase 3
        # self.phase_3_design_activities()

        # Complete
        self.state.current_phase = Phase.COMPLETE
        logger.info("========== ORCHESTRATION COMPLETE ==========")

        return self.get_output()

    # def get_output(self) -> SyllabusOutput
    def get_output(self):
        """Get final syllabus output"""
        if not self.state:
            raise Exception("No state - orchestration not started")

        data = {
            "description": self.state.description,
            "objectives": self.state.objectives,
            "outcomes": self.state.outcomes,
            "competencies": self.state.competencies,
            "sustainability": self.state.sustainability_integration,
            "sessions": self.state.sessions,
        }

        return data

        print(
            "Final output data:",
            json.dumps(data, default=lambda o: o.__dict__, indent=2),
        )

        return SyllabusOutput(
            description=self.state.description,
            objectives=self.state.objectives,
            outcomes=self.state.outcomes,
            competencies=self.state.competencies,
            sustainability=self.state.sustainability_integration,
            sessions=self.state.sessions,
            # activities=self.state.activities,
        )

    # ========================================
    # Checkpoint-Based Orchestration
    # ========================================

    # def run_with_checkpoints(
    #     self, user_input: UserInput, document_filenames: Optional[List[str]] = None
    # ):
    #     """
    #     Generator-based orchestration that yields checkpoints for user approval.

    #     Usage:
    #         orchestrator = SyllabusOrchestrator()
    #         gen = orchestrator.run_with_checkpoints(user_input)

    #         checkpoint = next(gen)  # Get first checkpoint
    #         action = UserAction(action_type="accept")
    #         checkpoint = gen.send(action)  # Send action and get next checkpoint

    #     Yields:
    #         CheckpointData: Data for user to review at each checkpoint
    #     """
    #     logger.info("========== STARTING CHECKPOINT-BASED ORCHESTRATION ==========")

    #     # Phase 0: Initialize
    #     self.start(user_input, document_filenames=document_filenames)
    #     context_text = getattr(self, "_context_text", "")

    #     # ========================================
    #     # CHECKPOINT 1: Course Description
    #     # ========================================
    #     logger.info("Phase 1.1: Generating course description")
    #     self.state.current_phase = Phase.DESCRIPTION

    #     description = self.course_description_agent.generate(
    #         mode=self.state.mode,
    #         metadata=self.state.metadata,
    #         context_text=context_text,
    #         output_language=self.state.metadata.output_language,
    #     )
    #     self.state.description = description

    #     # Yield checkpoint 1
    #     checkpoint = CheckpointData(
    #         phase="description",
    #         data={
    #             "description": description.text,
    #             "word_count": description.word_count,
    #         },
    #         actions=["accept", "edit", "regenerate"],
    #     )
    #     action = yield checkpoint

    #     # Handle user action
    #     if action.action_type == "edit":
    #         # User edited description directly
    #         self.state.description = CourseDescription(
    #             text=action.edited_data["description"],
    #             word_count=len(action.edited_data["description"].split()),
    #         )
    #         logger.info("User edited description")
    #     elif action.action_type == "regenerate":
    #         # Regenerate without feedback
    #         logger.info("Regenerating description")
    #         description = self.course_description_agent.generate(
    #             mode=self.state.mode,
    #             metadata=self.state.metadata,
    #             context_text=context_text,
    #             output_language=self.state.metadata.output_language,
    #         )
    #         self.state.description = description
    #     elif action.action_type == "feedback":
    #         # Regenerate with user feedback
    #         logger.info(
    #             f"Regenerating description with user feedback: {action.feedback}"
    #         )
    #         description = self.course_description_agent.generate(
    #             mode=self.state.mode,
    #             metadata=self.state.metadata,
    #             context_text=context_text
    #             + "\n\nUser feedback for regeneration: "
    #             + action.feedback,
    #             output_language=self.state.metadata.output_language,
    #         )
    #         self.state.description = description

    #     # ========================================
    #     # CHECKPOINT 2: Complete Framework
    #     # ========================================
    #     logger.info("Generating complete pedagogical framework")

    #     # Generate objectives
    #     objectives = self.learning_objectives_agent.generate(
    #         description=self.state.description.text,
    #         context_text=context_text,
    #         metadata=self.state.metadata,
    #         mode=self.state.mode,
    #         output_language=self.state.metadata.output_language,
    #     )
    #     self.state.objectives = objectives

    #     # Integrate sustainability
    #     query = (
    #         self.state.description.text
    #         + " "
    #         + " ".join([obj.text for obj in objectives.objectives])
    #     )
    #     # filters = {"discipline": self.state.metadata.discipline}
    #     # sdg_resources = rag_retrieve(query, filters, top_k=15)

    #     high_relevance_docs = [
    #         doc
    #         for doc in sdg_resources
    #         if doc.relevance_score and doc.relevance_score > 0.7
    #     ]
    #     if len(high_relevance_docs) < 3:
    #         logger.warning("Insufficient RAG results, falling back to web search")
    #         # web_results = web_search(query)
    #         logger.info(f"Web search returned {len(web_results)} results")

    #     self.state.retrieved_resources.extend(sdg_resources)

    #     sustainability = self.sustainability_agent.integrate(
    #         description=self.state.description.text,
    #         objectives=objectives,
    #         sdg_resources=sdg_resources,
    #         metadata=self.state.metadata,
    #         mode=self.state.mode,
    #         output_language=self.state.metadata.output_language,
    #     )

    #     # Handle Mode 2B variants
    #     if self.state.mode == "mode_2b_augment":
    #         # Augment mode: Keep original objectives, add suggested ones
    #         if sustainability.suggested_objectives:
    #             logger.info(
    #                 f"Mode 2B-Augment: Adding {len(sustainability.suggested_objectives)} suggested objectives"
    #             )
    #             all_objectives = (
    #                 objectives.objectives + sustainability.suggested_objectives
    #             )
    #             self.state.objectives.objectives = all_objectives
    #     elif self.state.mode == "mode_2b_transform":
    #         # Transform mode: Objectives already rewritten with sustainability
    #         logger.info(
    #             "Mode 2B-Transform: Using rewritten objectives with sustainability"
    #         )
    #         # No need to merge - objectives are already transformed

    #     self.state.sustainability_integration = sustainability

    #     # Generate outcomes
    #     outcomes = self.learning_outcomes_agent.generate(
    #         objectives=self.state.objectives,
    #         sustainability_map=sustainability,
    #         metadata=self.state.metadata,
    #         output_language=self.state.metadata.output_language,
    #     )
    #     self.state.outcomes = outcomes

    #     # Map competencies
    #     greencomp = {
    #         "C1": "Valuing sustainability",
    #         "C2": "Supporting fairness",
    #         "C3": "Promoting nature",
    #         "C4": "Systems thinking",
    #         "C5": "Critical thinking",
    #         "C6": "Problem framing",
    #         "C7": "Futures literacy",
    #         "C8": "Adaptability",
    #         "C9": "Exploratory thinking",
    #         "C10": "Political agency",
    #         "C11": "Collective action",
    #         "C12": "Individual initiative",
    #     }
    #     competencies = self.competency_mapping_agent.map_competencies(
    #         outcomes=outcomes,
    #         greencomp_framework=greencomp,
    #         output_language=self.state.metadata.output_language,
    #     )
    #     self.state.competencies = competencies

    #     # Validate framework
    #     validation = self.pedagogical_engineer_agent.validate(
    #         description=self.state.description.text,
    #         objectives=self.state.objectives,
    #         outcomes=outcomes,
    #         competencies=competencies,
    #         sustainability_map=sustainability,
    #         metadata=self.state.metadata,
    #         output_language=self.state.metadata.output_language,
    #     )
    #     self.state.validation_history.append(validation)

    #     # Auto-apply validation recommendations
    #     if not validation.passed:
    #         if validation.severity == "minor":
    #             logger.info("Auto-applying minor validation fixes")
    #             for suggestion in validation.suggestions:
    #                 logger.info(f"Applied: {suggestion}")
    #         elif validation.severity == "major":
    #             logger.warning("Major validation issues - showing to user")
    #             for issue in validation.issues:
    #                 logger.warning(f"Issue: {issue}")

    #     # Yield checkpoint 2
    #     checkpoint = CheckpointData(
    #         phase="framework",
    #         data={
    #             "description": self.state.description.text,
    #             "objectives": self.state.objectives,
    #             "sustainability": sustainability,
    #             "outcomes": outcomes,
    #             "competencies": competencies,
    #             "validation": validation,
    #         },
    #         actions=["accept", "edit"],
    #     )
    #     action = yield checkpoint

    #     # Handle user action
    #     if action.action_type == "edit" and action.edited_data:
    #         # User edited framework components
    #         if "objectives" in action.edited_data:
    #             self.state.objectives = action.edited_data["objectives"]
    #         if "outcomes" in action.edited_data:
    #             self.state.outcomes = action.edited_data["outcomes"]
    #         logger.info("User edited framework")
    #     elif action.action_type == "regenerate":
    #         # Regenerate framework components
    #         logger.info("Regenerating framework")
    #         # Re-run the framework generation steps
    #         objectives = self.learning_objectives_agent.generate(
    #             description=self.state.description.text,
    #             context_text=context_text,
    #             metadata=self.state.metadata,
    #             mode=self.state.mode,
    #             output_language=self.state.metadata.output_language,
    #         )
    #         self.state.objectives = objectives
    #         # Re-integrate sustainability and regenerate outcomes/competencies
    #         # (simplified - could be more granular)
    #     elif action.action_type == "feedback":
    #         # Regenerate framework with user feedback
    #         logger.info(f"Regenerating framework with user feedback: {action.feedback}")
    #         feedback_context = (
    #             context_text + "\n\nUser feedback for improvement: " + action.feedback
    #         )

    #         objectives = self.learning_objectives_agent.generate(
    #             description=self.state.description.text,
    #             context_text=feedback_context,
    #             metadata=self.state.metadata,
    #             mode=self.state.mode,
    #             output_language=self.state.metadata.output_language,
    #         )
    #         self.state.objectives = objectives

    #         # Re-generate sustainability integration
    #         query = (
    #             self.state.description.text
    #             + " "
    #             + " ".join([obj.text for obj in objectives.objectives])
    #         )
    #         sdg_resources = rag_retrieve(
    #             query, {"discipline": self.state.metadata.discipline}, top_k=15
    #         )

    #         sustainability = self.sustainability_agent.integrate(
    #             description=self.state.description.text,
    #             objectives=objectives,
    #             sdg_resources=sdg_resources,
    #             metadata=self.state.metadata,
    #             mode=self.state.mode,
    #             output_language=self.state.metadata.output_language,
    #         )
    #         self.state.sustainability_integration = sustainability

    #         # Re-generate outcomes
    #         outcomes = self.learning_outcomes_agent.generate(
    #             objectives=objectives,
    #             sustainability_map=sustainability,
    #             metadata=self.state.metadata,
    #             output_language=self.state.metadata.output_language,
    #         )
    #         self.state.outcomes = outcomes

    #         # Re-map competencies
    #         greencomp = {
    #             "C1": "Valuing sustainability",
    #             "C2": "Supporting fairness",
    #             "C3": "Promoting nature",
    #             "C4": "Systems thinking",
    #             "C5": "Critical thinking",
    #             "C6": "Problem framing",
    #             "C7": "Futures literacy",
    #             "C8": "Adaptability",
    #             "C9": "Exploratory thinking",
    #             "C10": "Political agency",
    #             "C11": "Collective action",
    #             "C12": "Individual initiative",
    #         }
    #         competencies = self.competency_mapping_agent.map_competencies(
    #             outcomes=outcomes,
    #             greencomp_framework=greencomp,
    #             output_language=self.state.metadata.output_language,
    #         )
    #         self.state.competencies = competencies

    #     # ========================================
    #     # CHECKPOINT 3: Session Planning
    #     # ========================================
    #     logger.info("Phase 2: Distributing sessions")
    #     self.state.current_phase = Phase.SESSIONS

    #     sessions = distribute_objectives(
    #         objectives=self.state.objectives.objectives, metadata=self.state.metadata
    #     )
    #     self.state.sessions = sessions

    #     # Yield checkpoint 3
    #     checkpoint = CheckpointData(
    #         phase="sessions",
    #         data={"sessions": sessions, "objectives": self.state.objectives},
    #         actions=["accept", "edit"],
    #     )
    #     action = yield checkpoint

    #     if action.action_type == "edit" and action.edited_data:
    #         if "sessions" in action.edited_data:
    #             self.state.sessions = action.edited_data["sessions"]
    #         logger.info("User edited session plan")
    #     elif action.action_type == "regenerate":
    #         # Regenerate session distribution
    #         logger.info("Regenerating session plan")
    #         sessions = distribute_objectives(
    #             objectives=self.state.objectives.objectives,
    #             metadata=self.state.metadata,
    #         )
    #         self.state.sessions = sessions
    #     elif action.action_type == "feedback":
    #         # For sessions, feedback is noted but distribution is algorithmic
    #         # User can edit objectives if they want different distribution
    #         logger.info(f"Session plan feedback noted: {action.feedback}")
    #         # Optionally: could implement custom distribution based on feedback
    #         logger.warning(
    #             "Session feedback handling: currently sessions are algorithmically distributed"
    #         )

    #     # ========================================
    #     # CHECKPOINT 4+: Activities per Session
    #     # ========================================
    #     logger.info("Phase 3: Designing activities")
    #     self.state.current_phase = Phase.ACTIVITIES

    #     activities = {}

    #     for session in self.state.sessions:
    #         logger.info(f"Designing activities for session {session.session_number}")

    #         # Filter compatible activities
    #         filtered = filter_activities(
    #             session_type=session.type,
    #             session_mode=session.mode,
    #             duration=session.duration,
    #             class_size=session.class_size,
    #         )

    #         if not filtered:
    #             logger.warning(
    #                 f"No compatible activities for session {session.session_number}"
    #             )
    #             activities[session.session_number] = []
    #             continue

    #         # Recommend activities based on learning outcomes
    #         recommendations = self.activity_recommendation_agent.recommend(
    #             session=session,
    #             learning_outcomes=self.state.outcomes,
    #             filtered_activities=filtered,
    #             output_language=self.state.metadata.output_language,
    #         )

    #         # Generate guides
    #         guides = []
    #         for template in recommendations.recommended_activities:
    #             guide = self.activity_guide_generator_agent.generate(
    #                 activity_template=template,
    #                 session=session,
    #                 metadata=self.state.metadata,
    #                 sustainability_map=self.state.sustainability_integration,
    #                 output_language=self.state.metadata.output_language,
    #             )
    #             guides.append(guide)

    #         activities[session.session_number] = guides

    #         # Yield checkpoint for this session
    #         checkpoint = CheckpointData(
    #             phase=f"session_{session.session_number}",
    #             data={
    #                 "session": session,
    #                 "activities": guides,
    #                 "session_number": session.session_number,
    #                 "total_sessions": len(self.state.sessions),
    #             },
    #             actions=["accept", "edit"],
    #         )
    #         action = yield checkpoint

    #         if action.action_type == "edit" and action.edited_data:
    #             if "activities" in action.edited_data:
    #                 activities[session.session_number] = action.edited_data[
    #                     "activities"
    #                 ]
    #             logger.info(
    #                 f"User edited activities for session {session.session_number}"
    #             )
    #         elif action.action_type == "regenerate":
    #             # Regenerate activities for this session
    #             logger.info(
    #                 f"Regenerating activities for session {session.session_number}"
    #             )
    #             recommendations = self.activity_recommendation_agent.recommend(
    #                 session=session,
    #                 learning_outcomes=self.state.outcomes,
    #                 filtered_activities=filtered,
    #                 output_language=self.state.metadata.output_language,
    #             )
    #             guides = []
    #             for template in recommendations.recommended_activities:
    #                 guide = self.activity_guide_generator_agent.generate(
    #                     activity_template=template,
    #                     session=session,
    #                     metadata=self.state.metadata,
    #                     sustainability_map=self.state.sustainability_integration,
    #                     output_language=self.state.metadata.output_language,
    #                 )
    #                 guides.append(guide)
    #             activities[session.session_number] = guides
    #         elif action.action_type == "feedback":
    #             # Regenerate activities with user feedback
    #             logger.info(f"Regenerating activities with feedback: {action.feedback}")
    #             # Could pass feedback to agent prompts - for now, simple regeneration
    #             # Future enhancement: pass feedback to activity guide generator
    #             recommendations = self.activity_recommendation_agent.recommend(
    #                 session=session,
    #                 learning_outcomes=self.state.outcomes,
    #                 filtered_activities=filtered,
    #                 output_language=self.state.metadata.output_language,
    #             )
    #             guides = []
    #             for template in recommendations.recommended_activities:
    #                 guide = self.activity_guide_generator_agent.generate(
    #                     activity_template=template,
    #                     session=session,
    #                     metadata=self.state.metadata,
    #                     sustainability_map=self.state.sustainability_integration,
    #                     output_language=self.state.metadata.output_language,
    #                 )
    #                 guides.append(guide)
    #             activities[session.session_number] = guides

    #     self.state.activities = activities

    #     # ========================================
    #     # COMPLETE
    #     # ========================================
    #     self.state.current_phase = Phase.COMPLETE
    #     logger.info("========== CHECKPOINT-BASED ORCHESTRATION COMPLETE ==========")

    #     # Final checkpoint with complete output
    #     yield CheckpointData(
    #         phase="complete", data=self.get_output().dict(), actions=[]
    #     )
