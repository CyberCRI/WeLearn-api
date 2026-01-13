import time
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.app.services.tutor.models import MessageWithResources, SyllabusResponseAgent
from src.app.services.tutor.utils import build_system_message
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

# TODO: add template file move this to utils
TEMPLATES = {"template0": Path("src/app/services/tutor/template.md").read_text()}


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
        contents = "summary :".join(message.summary)
        themes = ",".join([theme["theme"] for theme in message.themes])
        prompt = (
            "Using the content in TEXT CONTENTS, you generate a syllabus that is engaging "
            "and coherent in relation to the THEMES extracted from these contents. "
            f"The syllabus should be written in lang: {message.lang} the section names must also be written in {message.lang}, this is important \n\nTEXT CONTENTS:\n{contents}\n\n"
            f"THEMES:\n{themes} \n\nTake into account the users input courses title, level, duration and "
            f"description: {message.course_title}, {message.level}, {message.duration}, {message.description}."
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
