import time
import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.app.models.documents import Document
from src.app.tutor.service.models import (
    Competency,
    CourseMetadata,
    MessageWithResources,
    Objective,
    Outcome,
    SustainabilityMapping,
    SyllabusResponseAgent,
)
from src.app.shared.utils.utils import build_system_message
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


_DISCIPLINARY_SKILLS = None


def get_disciplinary_skills():
    global _DISCIPLINARY_SKILLS
    if _DISCIPLINARY_SKILLS is None:
        try:
            path = Path(__file__).parent.parent / "domain" / "disciplinary_skills.json"
            with open(path, "r", encoding="utf-8") as f:
                _DISCIPLINARY_SKILLS = {
                    d["code_rncp"]: d["skills"] for d in json.load(f)["disciplines"]
                }
        except Exception:
            # Handle error, log, or provide fallback
            _DISCIPLINARY_SKILLS = {}
    return _DISCIPLINARY_SKILLS


# TODO: add template file move this to utils
TEMPLATES = {"template0": Path("src/app/tutor/domain/template.md").read_text()}


class TutorChatAgent:
    """Thin wrapper around a LangChain chat model with a fixed system prompt."""

    def __init__(self, name: str, model: BaseChatModel, system_prompt: str) -> None:
        self.name = name
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{user_prompt}")]
        )
        # Parser keeps the contract identical to the previous autogen agents
        self.chain = prompt | model | StrOutputParser()

    async def run(self, user_prompt: str) -> str:
        start_time = time.time()
        response = await self.chain.ainvoke({"user_prompt": user_prompt})
        logger.debug(
            "agent_type=%s response_time=%s", self.name, time.time() - start_time
        )
        return response


class ActivityGuideGeneratorAgent(TutorChatAgent):
    """Generates detailed activity guides for instructors based on the course schedule."""

    def __init__(self, model: BaseChatModel, lang) -> None:
        system_prompt = build_system_message(
            role="You are a pedagogical designer who creates ready-to-use activity guides for professors. You take generic activity templates and personalize them with course-specific content, examples, and sustainability integration.",
            backstory="You are an expert in instructional design and active learning strategies. You excel at creating detailed activity guides that help instructors effectively deliver course content and achieve learning outcomes.",
            goal="""Transform this generic activity template into a detailed, ready-to-use guide for this specific course. The professor should be able to follow this guide directly to implement the activity.

<activity_template>
Name: {activity_template.name}
Description: {activity_template.description}
Estimated duration: {activity_template.estimated_duration}h
</activity_template>""",
            instructions=(
                """
Create a detailed guide with:
1. Activity description: Adapted to this course context (mention discipline, topic)

2. Learning objectives: Which objective numbers (from session.objectives) this addresses

3. Detailed steps: Concrete procedure professor can follow
    - Be specific (not "discuss the topic" but "pose question X, give 5min for pairs to discuss, then collect 3 key points")
    - Include timing for each step
    - Use course-specific examples from the discipline

4. Teacher role: What professor does during activity

5. Student role: What students do, how they participate

6. Resources needed: Specific materials, handouts, tools
    - Course-specific resources (not generic)
    - Include digital tools if mode is Distanciel/Hybride

7. Timing breakdown: Detailed time allocation
    - Example: {"introduction": 10, "pair_work": 20, "group_discussion": 30, "debrief": 10}
    - Total should match activity template's estimated_duration

8. Evaluation method: How to assess if learning happened
    - Formative assessment integrated into activity

9. Sustainability integration: HOW sustainability themes are woven into this activity
    - Use sustainability connections for relevant objectives
    - Be specific and natural (not tacked on)
       """
            ),
            expected_output=(
                f"""
    <examples>
    GENERIC: "Students discuss the topic in groups"
    SPECIFIC: "Students analyze the case study of [discipline-specific example] in groups of 4. Each group receives a worksheet with 3 guiding questions: (1) What are the key [disciplinary concepts]? (2) How do [sustainability themes] factor into the analysis? (3) What trade-offs exist? Groups have 20 minutes to discuss and prepare a 2-minute summary."

    GENERIC: "Use sustainability principles"
    SPECIFIC: "During the case analysis, students explicitly apply lifecycle thinking from SDG 12 resources. They map the full value chain, identifying environmental externalities at each stage. This connects to Objective 3's focus on system boundaries in [discipline]."
    </examples>

    <output_format>
    Return a JSON object with:
    {{
      "activity_name": "Adapted name if needed",
      "description": "Course-specific description...",
      "learning_objectives": [1, 3],
      "steps": ["Step 1...", "Step 2...", ...],
      "teacher_role": "What professor does...",
      "student_role": "What students do...",
      "resources_needed": ["Resource 1", "Resource 2", ...],
      "timing_breakdown": {{"phase1": 15, "phase2": 30, ...}},
      "evaluation_method": "How to assess...",
      "sustainability_integration": "How sustainability is woven in..."
    }}
    </output_format>

    <output_language>
    Generate all outputs in {lang}.
    </output_language>"""
            ),
        )
        super().__init__("ActivityGuideGeneratorAgent", model, system_prompt)


class CourseDescriptionAgent(TutorChatAgent):
    """Generates a course description based on the syllabus content."""

    def __init__(self, model: BaseChatModel, lang, metadata: CourseMetadata) -> None:
        system_prompt = build_system_message(
            role="You are an expert course designer for French higher education, specializing in creating compelling course descriptions that balance disciplinary rigor with sustainability integration.",
            backstory="You are an expert in crafting course descriptions that effectively communicate the value and content of academic courses. You understand how to distill complex syllabi into clear and appealing descriptions that resonate with prospective students.",
            goal=f"""Generate a course description (150-250 words) for the following course. The description should be academically rigorous, discipline-specific, and naturally integrate sustainability themes without sounding promotional or generic.
<metadata>
- Discipline: {metadata.discipline}
- Topic: {metadata.topic}
- Level: {metadata.level}
- Number of sessions: {metadata.num_sessions}
- Session format: {metadata.session_type} ({metadata.session_mode})
- Class size: {metadata.class_size}
</metadata>
<context>
{{ context_text }}
</context>
            """,
            instructions=(
                """
<requirements>
- Length: 150-250 words exactly
- Academic tone appropriate for French university context (enseignement supérieur)
- Be specific to the discipline and topic (avoid generic language)
- Naturally integrate sustainability themes relevant to the discipline
- Frame sustainability as enrichment, not compliance
- Focus on what students will learn and why it matters
- For Mode 2B with existing description: use provided description as-is
- For Mode 3 (metadata only): if context is sparse or generic, you may use general disciplinary knowledge but be specific
</requirements>"""
            ),
            expected_output=(
                f"""<output_format>
Return a JSON object with:
{{
    "text": "The course description (150-250 words)",
    "word_count": <integer count of words>
}}
</output_format>

<output_language>
Generate all outputs in {lang}.
</output_language>"""
            ),
        )
        super().__init__("CourseDescriptionAgent", model, system_prompt)


class CompetencyMappingAgent(TutorChatAgent):
    """Maps course content to specific competencies, including GreenComp."""

    def __init__(
        self,
        model: BaseChatModel,
        greencomp_competencies: str,
        lang,
        outcomes: list[Outcome],
    ) -> None:

        str_outcomes = []
        for outcome in outcomes:
            str_outcomes.append(
                f"{outcome.number}. {outcome.text} : Related to objectives: {', '.join(outcome.related_objectives)})"
            )
        system_prompt = build_system_message(
            role="You are an expert in the GreenComp framework (European sustainability competences). You map learning outcomes to relevant GreenComp competencies with clear pedagogical rationale.",
            backstory="You are an expert in competency-based education and curriculum design. You have a deep understanding of how to align course content with specific competencies to ensure that students acquire the skills and knowledge they need.",
            goal=f""" For each learning outcome, identify 1-3 relevant GreenComp competencies and provide brief rationale for the mapping.
    <inputs>
    <outcomes>
    {'\n'.join(str_outcomes)}
    </outcomes>

    <greencomp_framework>
    Area 1: Embodying sustainability values
    - C1: Valuing sustainability (reflect on personal values, identify and explain how they guide actions)
    - C2: Supporting fairness (support equity and justice for current and future generations)
    - C3: Promoting nature (acknowledge humans are part of nature, respect needs of other species)

    Area 2: Embracing complexity in sustainability
    - C4: Systems thinking (approach sustainability challenges as complex systems)
    - C5: Critical thinking (assess information and arguments, identify assumptions and limitations)
    - C6: Problem framing (formulate current and future sustainability challenges as problems)

    Area 3: Envisioning sustainable futures
    - C7: Futures literacy (envision alternative sustainable futures)
    - C8: Adaptability (manage transitions and challenges in complex sustainability situations)
    - C9: Exploratory thinking (adopt relational thinking, explore and link different disciplines)

    Area 4: Acting for sustainability
    - C10: Political agency (navigate political systems, identify political responsibility)
    - C11: Collective action (act for change collaboratively)
    - C12: Individual initiative (identify own potential for sustainability, take action)
    </greencomp_framework>
    </inputs>

    <requirements>
    - Map each outcome to 1-3 GreenComp competencies (most relevant)
    - Provide brief rationale (1-2 sentences) explaining WHY this competency fits
    - Use official GreenComp codes and names (e.g., "C4: Systems thinking")
    - Ensure diversity across outcomes (don't map everything to C4, distribute across framework)
    - Be pedagogically honest - only map if there's genuine alignment
    </requirements>

    <examples>
    Good mapping:
    Outcome: "Analyze supply chain efficiency using lifecycle thinking and environmental impact metrics"
    Competencies: ["C4: Systems thinking", "C9: Exploratory thinking"]
    Rationale: "This outcome develops systems thinking (C4) by requiring students to view supply chains as interconnected systems with environmental feedback loops. It also promotes exploratory thinking (C9) by linking business operations (supply chain) with environmental science (lifecycle analysis)."
    </examples>""",
            instructions=(
                """<requirements>
    - Map each outcome to 1-3 GreenComp competencies (most relevant)
    - Provide brief rationale (1-2 sentences) explaining WHY this competency fits
    - Use official GreenComp codes and names (e.g., "C4: Systems thinking")
    - Ensure diversity across outcomes (don't map everything to C4, distribute across framework)
    - Be pedagogically honest - only map if there's genuine alignment
    </requirements>

    <examples>
    Good mapping:
    Outcome: "Analyze supply chain efficiency using lifecycle thinking and environmental impact metrics"
    Competencies: ["C4: Systems thinking", "C9: Exploratory thinking"]
    Rationale: "This outcome develops systems thinking (C4) by requiring students to view supply chains as interconnected systems with environmental feedback loops. It also promotes exploratory thinking (C9) by linking business operations (supply chain) with environmental science (lifecycle analysis)."
    </examples>"""
            ),
            expected_output=(
                f""" <output_format>
    Return a JSON object with:
    {{
      "mappings": [
        {{
          "outcome_number": 1,
          "greencomp_competencies": ["C4: Systems thinking", "C9: Exploratory thinking"],
          "rationale": "Brief explanation..."
        }},
        ...
      ]
    }}
    </output_format>

    <output_language>
    Generate all outputs in {lang}.
    </output_language>"""
            ),
        )
        super().__init__("CompetencyMappingAgent", model, system_prompt)
        self.greencomp_competencies = greencomp_competencies


class LearningOutcomesAgent(TutorChatAgent):
    """Ensures learning objectives are well-formulated and aligned with learning outcomes."""

    def __init__(
        self,
        model: BaseChatModel,
        lang,
        objectives: list[Objective],
        sustainability_map: list[SustainabilityMapping],
        metadata: CourseMetadata,
    ) -> None:
        str_objectives = []
        for obj in objectives:
            str_objectives.append(
                f"{obj.number}. {obj.text} (Bloom's level: {obj.bloom_level})"
            )

        str_sustainability_map = []
        for conn in sustainability_map:
            str_sustainability_map.append(
                f"Objective {conn.objective_number}: {', '.join(conn.sdg_themes)}\nConnection: {conn.connection_explanation}"
            )

        system_prompt = build_system_message(
            role="You are a pedagogical assessment expert. You create measurable, observable learning outcomes that specify what students will be able to demonstrate upon completing the course.",
            backstory="You are an expert in educational psychology and curriculum design, with a deep understanding of Bloom's Taxonomy. You excel at crafting clear, measurable learning outcomes that effectively guide both instruction and assessment.",
            goal=f"""
            For each learning objective (including any suggested sustainability objectives), generate 1-3 specific, measurable outcomes. Naturally integrate sustainability themes informed by the sustainability mapping.
            
            <inputs>
    <objectives>
    {'\n'.join(str_objectives)}
    </objectives>

    <sustainability_connections>
    {'\n'.join(str_sustainability_map)}
    </sustainability_connections>

    <metadata>
    - Discipline: {metadata.discipline}
    - Level: {metadata.level}
    - Duration: {metadata.num_sessions} sessions
    </metadata>
    </inputs>
            """,
            instructions=(
                """<requirements>
    Each outcome should be:
    - Measurable and observable (use action verbs: demonstrate, analyze, create, evaluate, design, etc.)
    - Specific enough to guide activity design
    - Achievable within course duration
    - Related to 1-2 objectives (can span multiple)
    - Include assessment method (how to measure: exam, project, presentation, quiz, etc.)

    Integration of sustainability:
    - Weave sustainability naturally into outcomes (don't create separate "sustainability outcomes")
    - Use the sustainability connections to inform outcome design
    - Example: "Analyze supply chain efficiency using lifecycle thinking and environmental impact metrics"
      vs separate: "Analyze supply chains" + "Understand environmental impacts"

    Number of outcomes:
    - 1-3 outcomes per objective (more complex objectives may need more outcomes)
    - Total outcomes: typically 1.5x-2x number of objectives
    </requirements>"""
            ),
            expected_output=(
                f"""
                <output_format>
    Return a JSON object with:
    {{
      "outcomes": [
        {{
          "number": 1,
          "text": "Students will demonstrate...",
          "related_objectives": [1, 2],
          "assessment_method": "Case study analysis and written report"
        }},
        ...
      ]
    }}
    </output_format>

    <output_language>
    Generate all outputs in {lang}.
    </output_language>"""
            ),
        )
        super().__init__("LearningObjectivesAgent", model, system_prompt)


class LearningObjectivesAgent(TutorChatAgent):
    """Refines learning outcomes to ensure they are measurable and aligned with Bloom's Taxonomy."""

    def __init__(
        self,
        model: BaseChatModel,
        lang,
        mode,
        description,
        context_text,
        metadata: CourseMetadata,
    ) -> None:
        system_prompt = build_system_message(
            role="You are a pedagogical expert specializing in learning objective design. You create clear learning objectives that describe what will be taught and covered in the course from the instructor's perspective.",
            backstory="You are an expert in curriculum design and educational psychology, with a deep understanding of how to formulate effective learning objectives. You excel at ensuring that learning objectives are clearly articulated and aligned with learning outcomes to guide student learning.",
            goal=f"""
            <task>
    Generate learning objectives for this course. The number and complexity should match the course level and duration.

    For Mode 2B: Parse the provided objectives and structure them (do NOT generate new ones).
    For all other modes: Generate new objectives.
    </task>

    <inputs>
    <mode>{mode}</mode>
    <course_description>{description}</course_description>
    <context>{context_text}</context>
    <metadata>
    - Discipline: {metadata.discipline}
    - Topic: {metadata.topic}
    - Level: {metadata.level}
    - Number of sessions: {metadata.num_sessions}
    </metadata>
    </inputs>
            """,
            instructions=(
                """<requirements>
    - Number of objectives:
      * Short courses (<6 sessions): 3-4 objectives
      * Medium courses (6-12 sessions): 4-6 objectives
      * Long courses (>12 sessions): 6-8 objectives

    - Each objective should be:
      * Written from TEACHER PERSPECTIVE (what instructor intends to teach/cover)
      * Use teacher-centered verbs: "Introduce", "Familiarize students with", "Provide an overview of", "Explore", "Cover", "Examine", "Present"
      * Focused on instructional intent and content coverage
      * Specific enough to guide course design
      * Achievable within course duration
      * Focused on disciplinary content (sustainability comes in next step)

    - Examples of good learning objectives:
      * "Introduce students to the fundamentals of sustainable development"
      * "Familiarize students with economic theories of market failures"
      * "Provide an overview of climate change policy frameworks"
      * "Explore the relationship between resource consumption and environmental impact"

    - AVOID student-centered language like:
      * "Students will be able to..."
      * "Students will demonstrate..."
      * "Learners will..."

    - For Mode 2B: Keep objectives exactly as provided, just parse and structure them
    </requirements>
        """
            ),
            expected_output=(
                f"""
                <output_format>
    Return a JSON object with:
    {{
      "objectives": [
        {{
          "number": 1,
          "text": "Introduce students to sustainable development principles"
        }},
        {{
          "number": 2,
          "text": "Familiarize students with environmental policy frameworks"
        }},
        ...
      ]
    }}

    IMPORTANT: Write objectives from teacher perspective describing instructional intent, NOT student outcomes.
    </output_format>

    <output_language>
    Generate all outputs in {lang}.
    </output_language>"""
            ),
        )
        super().__init__("LearningOutcomesAgent", model, system_prompt)


class PedagogicalEngAgent(TutorChatAgent):
    """Ensures alignment between learning objectives, outcomes, and assessment methods."""

    def __init__(
        self,
        model: BaseChatModel,
        lang,
        description: str,
        objectives: list[Objective],
        outcomes: list[Outcome],
        competencies: list[Competency],
        sustainability_map: list[SustainabilityMapping],
        metadata: CourseMetadata,
    ) -> None:
        str_objectives = []
        for obj in objectives:
            str_objectives.append(
                f"{obj.number}. {obj.text} (Bloom's level: {obj.bloom_level})"
            )
        str_outcomes = []
        for outcome in outcomes:
            str_outcomes.append(
                f"{outcome.number}. {outcome.text} : Related to objectives: {', '.join(outcome.related_objectives)})"
            )

        str_competencies = []
        for comp in competencies:
            str_competencies.append(f"{comp.number}. {comp.text}")

        str_sustainability_map = []
        for conn in sustainability_map:
            str_sustainability_map.append(
                f"Objective {conn.objective_number}: {', '.join(conn.sdg_themes)}\nConnection: {conn.connection_explanation}"
            )

        system_prompt = build_system_message(
            role="""
You are an experienced pedagogical engineer (ingénieur pédagogique) providing constructive peer review of course frameworks. Your role is to validate overall coherence, quality, and feasibility.

You should identify issues at three severity levels:
- MINOR: Formatting, wording, small inconsistencies (can be auto-fixed)
- MAJOR: Missing elements, serious misalignment, pedagogical errors (requires regeneration)
- SUBJECTIVE: Pedagogical choices that are valid but could be improved (recommendations for user)
""",
            backstory="You are an expert in curriculum design and educational psychology, with a deep understanding of how to ensure pedagogical alignment in course design. You excel at analyzing syllabi to identify and address misalignments between learning objectives, outcomes, and assessment methods.",
            goal=f"""
            <task>
    Review this complete pedagogical framework for overall coherence, quality, and feasibility. Provide specific, actionable feedback.
    </task>

    <inputs>
    <description>{description}</description>

    <objectives>
    {'\n'.join(str_objectives)}
    </objectives>

    <outcomes>
    {'\n'.join(str_outcomes)}
    </outcomes>

    <competencies>
    {'\n'.join(str_competencies)}
    </competencies>

    <sustainability>
    {'\n'.join(str_sustainability_map)}
    </sustainability>

    <metadata>
    - Discipline: {metadata.discipline}
    - Level: {metadata.level}
    - Duration: {metadata.num_sessions} sessions of {metadata.session_duration}h
    - Format: {metadata.session_type}
    </metadata>
    </inputs>

    <validation_checklist>
    Check for:

    1. Coverage:
       - All objectives have at least one outcome? ✓
       - All outcomes mapped to competencies? ✓

    2. Coherence:
       - Outcomes align with objectives? ✓
       - Assessment methods appropriate for outcomes? ✓
       - Competency mappings make pedagogical sense? ✓

    3. Feasibility:
       - Objectives achievable in given number of sessions? ✓
       - Outcomes realistic for course level and duration? ✓
       - Assessment methods practical for class size and format? ✓

    4. Quality:
       - Objectives specific and measurable? ✓
       - Outcomes observable and assessable? ✓
       - Sustainability integration authentic (not forced or tacked on)? ✓

    5. Red flags:
       - Overly ambitious for duration? ✗
       - Misalignment between objectives and outcomes? ✗
       - Generic or vague language? ✗
       - Too many or too few objectives/outcomes? ✗
    </validation_checklist>
""",
            instructions=(
                """
                <requirements>
    - Be constructive and specific (peer review tone, not judgmental)
    - Reference pedagogical best practices
    - Classify severity: minor, major, or subjective
    - For major issues: explain what needs to change AND provide corrected version
    - For minor issues: provide auto-fix suggestions
    - For subjective: frame as recommendations, not requirements
    - If no issues: passed = true, empty issues list
    - IMPORTANT: Always provide specific corrections that can be applied automatically
    </requirements>

    <examples>
    MINOR issue: "Outcome 3 uses passive voice ('will be understood'). Rephrase with active, observable verb."

    MAJOR issue: "Objective 1 has no corresponding outcomes. At least one outcome must be created to measure this objective."

    SUBJECTIVE recommendation: "Consider balancing Bloom levels. Currently 5/6 objectives are 'Analyze' level. Adding one 'Create' level objective could enhance higher-order thinking for M2 students."
    </examples>
"""
            ),
            expected_output=(
                f"""<output_format>
    Return a JSON object with:
    {{
      "passed": true/false,
      "issues": ["Issue 1...", "Issue 2..."],  // Empty if passed
      "suggestions": ["Suggestion 1...", "Suggestion 2..."],  // Actionable feedback
      "severity": "minor" | "major" | "subjective"  // Highest severity if multiple issues
    }}

    If everything is valid and high quality: {{"passed": true, "issues": [], "suggestions": [], "severity": "minor"}}
    </output_format>

    <output_language>
    Generate all outputs in {lang}.
    </output_language>"""
            ),
        )
        super().__init__("PedagogicalAlignmentAgent", model, system_prompt)


class SustainabilityIntegrationAgent(TutorChatAgent):
    """Ensures that sustainability concepts are meaningfully integrated into the syllabus."""

    def __init__(
        self,
        model: BaseChatModel,
        greencomp_competencies: str,
        lang,
        mode: str,
        description: str,
        objectives: list[Objective],
        sdg_resources: list[Document],
        metadata: CourseMetadata,
    ) -> None:
        str_objectives = []
        for obj in objectives:
            str_objectives.append(
                f"{obj.number}. {obj.text} (Bloom's level: {obj.bloom_level})"
            )

        str_resources = []
        for doc in sdg_resources:
            str_resources.append(
                f"Document: {doc.payload.document_title},relevance: {doc.score},content: {doc.payload.slice_content[:500]}..."
            )
        system_prompt = build_system_message(
            role="""You are an expert in sustainability education and disciplinary pedagogy. Your role is to create AUTHENTIC, DEEP connections between course objectives and sustainability themes using SPECIFIC examples from provided SDG resources.

    This is the CORE VALUE PROPOSITION of WeLearn. You must avoid generic SDG labels and instead explain HOW sustainability enriches the disciplinary concept with concrete examples.""",
            backstory="You are an expert in sustainability education and curriculum design, with a deep understanding of how to integrate sustainability concepts into various disciplines. You excel at analyzing syllabi to identify opportunities for meaningful integration of sustainability principles.",
            goal=f"""<task>
    For EACH learning objective, identify relevant sustainability themes from the provided SDG resources and explain the SPECIFIC connection using examples from those resources.

    For Mode 2B ONLY: Also suggest 2-3 ADDITIONAL sustainability-focused objectives that complement (don't replace) the original objectives.
    </task>

    <inputs>
    <mode>{mode}</mode>
    <course_description>{description}</course_description>
    <objectives>
    {'\n'.join(str_objectives)}
    </objectives>

    <sdg_resources>
    {'\n'.join(str_resources)}
    </sdg_resources>

    <metadata>
    - Discipline: {metadata.discipline}
    - Topic: {metadata.topic}
    - Level: {metadata.level}
    </metadata>
    </inputs>""",
            instructions=(
                """<requirements>
    CRITICAL - This is what separates good from mediocre output:

    ❌ BAD (generic): "Objective 2 relates to SDG 13 (Climate Action)"

    ✅ GOOD (authentic): "Objective 2 explores market failures. SDG resources on carbon pricing (Document 3, pp. 15-23) provide concrete examples where markets fail to price environmental externalities. Use the European carbon tax case study from Document 5 to illustrate how economic instruments can correct this market failure, directly connecting microeconomic theory to climate policy."

    For each objective:
    1. Identify 1-3 relevant SDG themes from the resources
    2. Cite SPECIFIC documents and examples (document titles, page numbers if available)
    3. Explain HOW sustainability connects to the disciplinary concept (mechanism, not just label)
    4. Use academic language appropriate for the discipline
    5. Frame as enrichment that makes the learning deeper, not as compliance

    For Mode 2B: Suggest 2-3 additional sustainability-focused objectives that:
    - Are appropriate for the course level
    - Build on existing objectives
    - Are achievable in course duration
    - Feel natural, not forced

    Overall integration strategy:
    - Write a brief paragraph (3-5 sentences) explaining the overarching approach to integrating sustainability in this course
    - How does it fit the discipline? Why is it pedagogically valuable?
    </requirements>

    <examples>
    Good connection explanation:
    "This objective on analyzing supply chain efficiency (Objective 3) connects to SDG 12 (Responsible Consumption) through concrete examples in Document 2. The case study of circular economy in manufacturing (Document 2, Section 4.2) demonstrates how lifecycle thinking transforms traditional supply chain analysis. Students can apply the same analytical frameworks they learn for cost optimization to evaluate environmental impacts, using the methodology described in Document 7 (Environmental Supply Chain Management Framework)."
    </examples>"""
            ),
            expected_output=(
                f"""<output_format>
    Return a JSON object with:
    {{
      "connections": [
        {{
          "objective_number": 1,
          "sdg_themes": ["SDG 12: Responsible Consumption", "SDG 13: Climate Action"],
          "connection_explanation": "Detailed, specific explanation using examples from resources...",
          "key_resources": ["Document 2: Title", "Document 7: Title"]
        }},
        ...
      ],
      "suggested_objectives": [  // ONLY for Mode 2B
        {{
          "number": 99,  // Use high numbers to distinguish from original
          "text": "Introduce different theories of...",
        }}
      ],
      "integration_strategy": "Overall narrative paragraph...",
      "resources_used": [...]  // Copy from input
    }}
    </output_format>

    <output_language>
    Generate all outputs in {lang}.
    </output_language>"""
            ),
        )
        super().__init__("SustainabilityIntegrationAgent", model, system_prompt)
        self.greencomp_competencies = greencomp_competencies


class UniversityTeacherAgent(TutorChatAgent):
    """First-pass syllabus creation based on user documents and metadata."""

    def __init__(self, model: BaseChatModel, lang) -> None:
        system_prompt = build_system_message(
            role="the University Professor Agent, responsible for drafting the initial syllabus based on the course materials provided by the user. Your role is to structure the course content, ensuring it aligns with academic standards and effectively conveys the subject matter.",
            backstory="You are a highly experienced university professor with expertise in structuring academic courses. You understand the nuances of designing a syllabus that is comprehensive yet adaptable, providing a strong foundation for course delivery. Your experience spans multiple disciplines, and you excel at organizing complex information into a structured curriculum.",
            goal="Generate the first version of the syllabus, ensuring that it reflects the course content and discipline provided by the user. Your syllabus should be structured based on the syllabus template, and your version will serve as the foundation for subsequent agents.",
            instructions=(
                "1. Analyze the input materials uploaded by the user: 'content', 'summary', and 'themes'."
                f"2. Based on the Draft the following sections in the user language {lang}, sections titles must also be in the user language:"
                "Course Description: Provide a brief yet clear introduction to the course."
                "Learning Objectives: Outline broad goals that define what students will gain from the course."
                "Learning Outcomes: Define specific and measurable outcomes students should achieve."
                "Competencies: Define transferable skills that the students will gain after this course."
                "Assessment Methods: Propose methods to evaluate student progress."
                "Course Schedule: Create a week-by-week breakdown with key topics and associated learning outcomes."
                "References: Include all sources that you use to construct the syllabus."
            ),
            expected_output=f"You must follow this template :\n {TEMPLATES['template0']} and translate it into the target language: {lang}.",
        )
        super().__init__("UniversityTeacherAgent", model, system_prompt)

    async def generate(self, message: MessageWithResources) -> SyllabusResponseAgent:
        DISCIPLINARY_SKILLS = get_disciplinary_skills()
        contents = "summary :".join(message.summary)
        themes = ",".join([theme["theme"] for theme in message.themes])
        prompt = (
            "Using the content in TEXT CONTENTS, you generate a syllabus that is engaging "
            "and coherent in relation to the THEMES extracted from these contents. "
            f"The syllabus should be written in lang: {message.lang} the section names must also be written in {message.lang}, this is important \n\nTEXT CONTENTS:\n{contents}\n\n"
            f"THEMES:\n{themes} \n\nTake into account the users input courses title, level, duration and "
            f"description: {message.course_title}, {message.level}, {message.duration}, {message.description}."
            f"{('\n\nThe syllabus should also contribute to build the following disciplinary skills:'+'\n- '.join(DISCIPLINARY_SKILLS[message.discipline])) if message.discipline in DISCIPLINARY_SKILLS.keys() else ''}"
        )
        response = await self.run(prompt)
        return SyllabusResponseAgent(content=response, source=self.name)


class SDGExpertAgent(TutorChatAgent):
    """Injects sustainability and SDG alignment using WeLearn resources."""

    def __init__(
        self, model: BaseChatModel, greencomp_competencies: str, lang: str
    ) -> None:
        system_prompt = build_system_message(
            role="the Sustainability Expert Agent, responsible for integrating sustainability concepts into the syllabus. Your role is to ensure that the syllabus aligns with relevant sustainability principles and frameworks in a way that is appropriate for the course discipline.",
            backstory="You are a recognized expert in sustainability education, well-versed in the Sustainable Development Goals (SDGs) and emerging sustainability research. Your expertise allows you to inject sustainability elements into any discipline, making them relevant and actionable for students.",
            goal="Modify the initial syllabus provided by the University Professor Agent to integrate sustainability content in a relevant and meaningful way.",
            instructions=(
                "1. Analyze the syllabus generated by the University Professor Agent. Ensure modifications are discipline-appropriate and avoid forcing sustainability concepts where they do not naturally fit."
                f"2. Evaluate and refine the following sections, respect the user language {lang}, sections titles must also be in the user language:"
                "Course Description: Ensure sustainability relevance is reflected."
                "Learning Objectives & Outcomes: Add sustainability-focused objectives where applicable."
                "Competencies Developed: If relevant, integrate sustainability competencies."
                "Course Schedule: Introduce sustainability-related themes into weekly topics where appropriate."
                "References: NEVER delete, summarize, or modify any references already present in the syllabus. Only append this section with any additional sources that you use to construct the syllabus. "
            ),
            expected_output=(
                f"1. The revised syllabus, ready for the pedagogical engineer's review. You must follow this template :\n {TEMPLATES['template0']} and translate it into the target language: {lang}.."
            ),
        )
        super().__init__("SDGExpertAgent", model, system_prompt)
        self.greencomp_competencies = greencomp_competencies

    async def enhance(
        self,
        syllabus: SyllabusResponseAgent,
        resources: list[dict],
        lang: str,
    ) -> SyllabusResponseAgent:
        prompt = (
            f"Use these WeLearn documents: {resources} to integrate sustainability in this syllabus: {syllabus.content}. "
            "You do not need to use ALL the information in the WeLearn documents, but ensure that sustainability integration "
            "is done in a way that is relevant and thematically linked to the discipline and the topics of the syllabus, that "
            "they are deeply embedded in the course content, and aligned with both the discipline and the broader educational goals. "
            "Add all WeLearn documents that you use in the REFERENCES section of the syllabus. "
            f"Keep the same language, lang: {lang}.\n\n"
            f"GreenComp competencies for reference: {self.greencomp_competencies}"
        )
        response = await self.run(prompt)
        return SyllabusResponseAgent(content=response, source=self.name)


class PedagogicalEngineerAgent(TutorChatAgent):
    """Final polish focusing on pedagogy and GreenComp alignment."""

    def __init__(self, model: BaseChatModel, greencomp_competencies: str, lang) -> None:
        system_prompt = build_system_message(
            role="the Pedagogical Engineer Agent, responsible for ensuring that the syllabus adheres to best practices in pedagogy. Your role is to refine learning objectives, align assessments with learning outcomes, and include competencies from the EU GreenComp Framework. You optimize the syllabus for student engagement and effectiveness.",
            backstory="You are an experienced pedagogical engineer specializing in higher education course design. You are deeply familiar with competency-based learning and the EU GreenComp Framework, active learning strategies, and assessment alignment. Your expertise ensures that syllabi are not only well-structured but also effective for learning.",
            goal="Refine the syllabus provided by the SDG Expert Agent to optimize its pedagogical effectiveness and coherence. Make sure the learning objectives, learning outcomes, and competencies are linked, and that the activities proposed in the Course Plan are appropriate for the accomplishment of the learning outcomes that are targeted.",
            instructions=(
                "1. Analyze the syllabus generated by the SDG Expert Agent."
                f"2. Evaluate and refine the following sections respect the use language {lang}:"
                "Learning Objectives & Outcomes: Ensure they are well-formulated and that the learning outcomes and thematically linked to the learning objectives."
                "Competencies Developed: Integrate competencies from the EU GreenComp Framework and validate the inclusion of relevant transferable skills."
                "Assessment Methods: Align evaluation strategies with learning outcomes."
                "Course Schedule: (1) For EACH WEEK, make sure that the Learning Outcomes mentioned are the same ones as the ones listed in the Learning Outcomes section, and include the Learning Outcome number in front of each one. (2) analyse the learning outcome and add a new column entitled Class Plan including a structured plan for the class with the activities that the professor should implement in order to best accomplish the outcomes targeted for that week. Ensure activities align with active learning strategies and innovative pedagogical approaches. (3) Go back to the Learning Outcomes section and include the class activities you listen in the Class Plan in the formulation of each Learning Outcome, for the class activities are the measures with which the professor will  measure the accomplishment of each outcome. Do this for EVERY WEEK. DO NOT SUMMARIZE OR USE ELLIPSES AS PLACEHOLDERS FOR THE ACTUAL COURSE PLAN."
                "References: NEVER delete, summarize, or modify any references already present in the syllabus. Only append this section with any additional sources that you use to construct the syllabus."
                "3. Check for consistency, clarity, and overall syllabus coherence to ensure usability for instructors."
            ),
            expected_output=(
                f"1. Final Syllabus: The polished syllabus, ready for user review. You must follow this template :\n {TEMPLATES['template0']} and translate it into the target language: {lang}."
            ),
        )
        super().__init__("PedagogicalEngineerAgent", model, system_prompt)
        self.greencomp_competencies = greencomp_competencies

    async def refine(self, syllabus: SyllabusResponseAgent) -> SyllabusResponseAgent:
        prompt = (
            "Ensure that the syllabus is pedagogically sound, aligns with competency-based learning, and optimizes student engagement "
            "and learning effectiveness. Ensure that the learning objectives, outcomes and the competencies and related in a logical and "
            "meaningful way, and that these overarching goals are accomplished through the course plan and activities, also ensure that "
            "the competencies cited in the EU GreenComp framework are present in the syllabus in a way that is coherent with the discipline "
            "and the course content. Make sure to use the same language as the current syllabus.\n\n"
            f"SYLLABUS:\n{syllabus.content}\n\nGreenComp competencies for reference: {self.greencomp_competencies}"
        )
        response = await self.run(prompt)
        return SyllabusResponseAgent(content=response, source=self.name)
