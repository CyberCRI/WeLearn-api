import time

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import (
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_core.memory import ListMemory
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from src.app.services.tutor.models import MessageWithResources, SyllabusResponseAgent
from src.app.services.tutor.utils import build_system_message
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

# TODO: add template file mouve this to utils
TEMPLATES = {"template0": open("src/app/services/tutor/template.md").read()}


university_teacher_topic_type = "UniversityTeacherAgent"
sdg_expert_topic_type = "SDGExpertAgent"
pedagogical_engineer_topic_type = "PedagogicalEngineerAgent"
user_input_topic_type = "UserInputAgent"
user_topic_type = "User"

CLOSURE_AGENT_TYPE = "collect_result_agent"
TASK_RESULTS_TOPIC_TYPE = "task-results"
task_results_topic_id = TopicId(type=TASK_RESULTS_TOPIC_TYPE, source="default")


@type_subscription(topic_type=university_teacher_topic_type)
class UniversityTeacherAgent(RoutedAgent):
    """
    A university professor agent that designs courses in a given discipline.
    Attributes:
        _system_message (SystemMessage): The system message containing the agent's role, backstory, and goals.
        _model_client (ChatCompletionClient): The client used to interact with the language model.
    Methods:
        __init__(model_client: ChatCompletionClient) -> None:
            Initializes the UniversityTeacherAgent with a system message and model client.
        handle_documents_and_themes(message: MessageWithSources, ctx: MessageContext) -> None:
            Handles incoming messages, generates a syllabus based on the provided text contents and themes, and publishes the generated syllabus.
    """

    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A university teacher agent.")
        self._system_message = SystemMessage(
            content=(
                build_system_message(
                    role="the University Professor Agent, responsible for drafting the initial syllabus based on the course materials provided by the user. Your role is to structure the course content, ensuring it aligns with academic standards and effectively conveys the subject matter.",
                    backstory="You are a highly experienced university professor with expertise in structuring academic courses. You understand the nuances of designing a syllabus that is comprehensive yet adaptable, providing a strong foundation for course delivery. Your experience spans multiple disciplines, and you excel at organizing complex information into a structured curriculum.",
                    goal="Generate the first version of the syllabus, ensuring that it reflects the course content and discipline provided by the user. Your syllabus should be structured based on the syllabus template, and your version will serve as the foundation for subsequent agents.",
                    instructions=(
                        "1. Analyze the input materials uploaded by the user: 'content', 'summary', and 'themes'."
                        "2. Based on the Draft the following sections:"
                        "Course Description: Provide a brief yet clear introduction to the course."
                        "Learning Objectives: Outline broad goals that define what students will gain from the course."
                        "Learning Outcomes: Define specific and measurable outcomes students should achieve."
                        "Competencies: Define transferable skills that the students will gain after this course."
                        "Assessment Methods: Propose methods to evaluate student progress."
                        "Course Schedule: Create a week-by-week breakdown with key topics and associated learning outcomes."
                        "References: Include all sources that you use to construct the syllabus."
                    ),
                    expected_output=f"You must follow this template :\n {TEMPLATES['template0']}",
                )
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_documents_and_themes(
        self, message: MessageWithResources, ctx: MessageContext
    ) -> None:
        contents = "summary :".join(message.summary)
        themes = ",".join(message.themes)

        prompt = f"Using the content in TEXT CONTENTS, you generate a syllabus that is engaging and coherent in relation to the THEMES extracted from these contents. \n\nTEXT CONTENTS:\n{contents}\n\nTHEMES:\n{themes}"

        start_time = time.time()
        llm_result = await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=prompt, source=self.id.key),
            ],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        end_time = time.time()
        logger.debug(
            "agent_type=%s response_time=%s", self.id.type, end_time - start_time
        )
        assert isinstance(response, str)

        await self.publish_message(
            MessageWithResources(
                content=response,
                resources=message.resources,
                source=self.id.type,
                summary=message.summary,
                themes=message.themes,
            ),
            topic_id=TopicId(sdg_expert_topic_type, source=self.id.key),
        )

        await self.publish_message(
            SyllabusResponseAgent(content=response, source=self.id.type),
            task_results_topic_id,
        )


@type_subscription(topic_type=sdg_expert_topic_type)
class SDGExpertAgent(RoutedAgent):
    """
    An SDG expert that assists university professors in integrating sustainability topics in their discipline.
    Attributes:
        _delegate (AssistantAgent): An assistant agent that handles the core functionality.
        _model_client (ChatCompletionClient): The model client used for generating responses.
    Methods:
        handle_syllabus(message: Message, ctx: MessageContext) -> None:
            Handles messages from university professors and integrates sustainability concepts in a thematically linked and relevant manner.
    """

    def __init__(self, model_client: ChatCompletionClient, memory: ListMemory) -> None:
        super().__init__("An SDG expert agent.")
        self._delegate = AssistantAgent(
            "SDGExpert",
            model_client=model_client,
            system_message=build_system_message(
                role="the Sustainability Expert Agent, responsible for integrating sustainability concepts into the syllabus. Your role is to ensure that the syllabus aligns with relevant sustainability principles and frameworks in a way that is appropriate for the course discipline.",
                backstory="You are a recognized expert in sustainability education, well-versed in the Sustainable Development Goals (SDGs) and emerging sustainability research. Your expertise allows you to inject sustainability elements into any discipline, making them relevant and actionable for students.",
                goal="Modify the initial syllabus provided by the University Professor Agent to integrate sustainability content in a relevant and meaningful way.",
                instructions=(
                    "1. Analyze the syllabus generated by the University Professor Agent. Ensure modifications are discipline-appropriate and avoid forcing sustainability concepts where they do not naturally fit."
                    "2. Evaluate and refine the following sections:"
                    "Course Description: Ensure sustainability relevance is reflected."
                    "Learning Objectives & Outcomes: Add sustainability-focused objectives where applicable."
                    "Competencies Developed: If relevant, integrate sustainability competencies."
                    "Course Schedule: Introduce sustainability-related themes into weekly topics where appropriate."
                    "References: NEVER delete, summarize, or modify any references already present in the syllabus. Only append this section with any additional sources that you use to construct the syllabus. "
                ),
                expected_output=(
                    f"1. The revised syllabus, ready for the pedagogical engineer's review. You must follow this template :\n {TEMPLATES['template0']}."
                ),
            ),
            memory=[memory],
        )
        self._model_client = model_client

    @message_handler(match=lambda msg, ctx: msg.source.startswith("UniversityTeacher"))  # type: ignore
    async def handle_syllabus(
        self, message: MessageWithResources, ctx: MessageContext
    ) -> None:
        """
        Handles the syllabus by integrating SDG content from the documents in the WeLearn resources.
        Args:
            message (MessageWithResources): The message containing the syllabus content as well as the documents from the WeLearn database.
            ctx (MessageContext): The context of the message.
        Returns:
            None
        """

        prompt = f"Use these WeLearn documents: {message.resources} to integrate sustainability in this syllabus: {message.content}. You do not need to use ALL the information in the WeLearn documents, but ensure that sustainability integration is done in a way that is relevant and thematically linked to the discipline and the topics of the syllabus, that they are deeply embedded in the course content, and aligned with both the discipline and the broader educational goals. Add all WeLearn documents that you use in the REFERENCES section of the syllabus."
        try:
            start_time = time.time()
            llm_result = await self._delegate.on_messages(
                [TextMessage(content=prompt, source=self.id.key)],
                ctx.cancellation_token,
            )
        except Exception as e:
            print("Error in SDGExpertAgent:", e)
            raise e
        end_time = time.time()
        response = llm_result.chat_message.content

        logger.debug(
            "agent_type=%s response_time=%s", self.id.type, end_time - start_time
        )
        assert isinstance(response, str)
        await self.publish_message(
            SyllabusResponseAgent(content=response, source=self.id.type),
            topic_id=TopicId(pedagogical_engineer_topic_type, source=self.id.key),
        )

        await self.publish_message(
            SyllabusResponseAgent(content=response, source=self.id.type),
            task_results_topic_id,
        )


@type_subscription(topic_type=pedagogical_engineer_topic_type)
class PedagogicalEngineerAgent(RoutedAgent):
    """
    A pedagogical engineer agent that assists university professors in constructing their syllabus. This agent ensures that the syllabus contains appropriate learning objectives, learning outcomes, and a corresponding course plan. It focuses specifically on teaching methodologies and pedagogical approaches. It integrates the European sustainability competence framework GreenComp into existing courses.
    Attributes:
        _delegate (AssistantAgent): An assistant agent that handles the core functionality.
        _model_client (ChatCompletionClient): The model client used for generating responses.
    Methods:
        handle_syllabus(message: Message, ctx: MessageContext) -> None:
            Handles messages from university teachers to ensure the syllabus integrates GreenComp competencies.
        handle_user_feedback(message: MessageWithFeedback, ctx: MessageContext) -> None:
            Handles user feedback to improve the syllabus based on the provided feedback.
    """

    def __init__(self, model_client: ChatCompletionClient, memory: ListMemory) -> None:
        super().__init__("A pedagogical engineer agent.")
        self._delegate = AssistantAgent(
            "PedagogicalEngineer",
            model_client=model_client,
            system_message=build_system_message(
                role="the Pedagogical Engineer Agent, responsible for ensuring that the syllabus adheres to best practices in pedagogy. Your role is to refine learning objectives, align assessments with learning outcomes, and include competencies from the EU GreenComp Framework. You optimize the syllabus for student engagement and effectiveness.",
                backstory="You are an experienced pedagogical engineer specializing in higher education course design. You are deeply familiar with competency-based learning and the EU GreenComp Framework, active learning strategies, and assessment alignment. Your expertise ensures that syllabi are not only well-structured but also effective for learning.",
                goal="Refine the syllabus provided by the SDG Expert Agent to optimize its pedagogical effectiveness and coherence. Make sure the learning objectives, learning outcomes, and competencies are linked, and that the activities proposed in the Course Plan are appropriate for the accomplishment of the learning outcomes that are targeted.",
                instructions=(
                    "1. Analyze the syllabus generated by the SDG Expert Agent."
                    "2. Evaluate and refine the following sections:"
                    "Learning Objectives & Outcomes: Ensure they are well-formulated and that the learning outcomes and thematically linked to the learning objectives."
                    "Competencies Developed: Integrate competencies from the EU GreenComp Framework and validate the inclusion of relevant transferable skills."
                    "Assessment Methods: Align evaluation strategies with learning outcomes."
                    "Course Schedule: (1) For EACH WEEK, make sure that the Learning Outcomes mentioned are the same ones as the ones listed in the Learning Outcomes section, and include the Learning Outcome number in front of each one. (2) analyse the learning outcome and add a new column entitled Class Plan including a structured plan for the class with the activities that the professor should implement in order to best accomplish the outcomes targeted for that week. Ensure activities align with active learning strategies and innovative pedagogical approaches. (3) Go back to the Learning Outcomes section and include the class activities you listen in the Class Plan in the formulation of each Learning Outcome, for the class activities are the measures with which the professor will  measure the accomplishment of each outcome. Do this for EVERY WEEK. DO NOT SUMMARIZE OR USE ELLIPSES AS PLACEHOLDERS FOR THE ACTUAL COURSE PLAN."
                    "References: NEVER delete, summarize, or modify any references already present in the syllabus. Only append this section with any additional sources that you use to construct the syllabus."
                    "3. Check for consistency, clarity, and overall syllabus coherence to ensure usability for instructors."
                ),
                expected_output=(
                    f"1. Final Syllabus: The polished syllabus, ready for user review. You must follow this template :\n {TEMPLATES['template0']}."
                ),
            ),
            memory=[memory],
        )
        self._model_client = model_client

    @message_handler(match=lambda msg, ctx: msg.source.startswith("SDGExpert"))  # type: ignore
    async def handle_syllabus(
        self, message: SyllabusResponseAgent, ctx: MessageContext
    ) -> None:
        """
        Handles the syllabus by improving on the pedagogical aspects and ensures that the competencies cited in the EU GreenComp framework are present in a coherent manner with the discipline and course content.
        Args:
            message (SyllabusResponseAgent): The message containing the syllabus content.
            ctx (MessageContext): The context of the message.
        Returns:
            None
        """

        prompt = f"Ensure that the syllabus is pedagogically sound, aligns with competency-based learning, and optimizes student engagement and learning effectiveness. Ensure that the learning objectives, outcomes and the competencies and related in a logical and meaningful way, and that these overarching goals are accomplished through the course plan and activities, also ensure that the competencies cited in the EU GreenComp framework are present in the syllabus in a way that is coherent with the discipline and the course content.\n\nSYLLABUS:\n{message.content}"
        start_time = time.time()
        llm_result = await self._delegate.on_messages(
            [TextMessage(content=prompt, source=self.id.key)],
            ctx.cancellation_token,
        )
        response = llm_result.chat_message.content
        end_time = time.time()
        assert isinstance(response, str)

        logger.debug(
            "agent_type=%s response_time=%s", self.id.type, end_time - start_time
        )
        await self.publish_message(
            SyllabusResponseAgent(content=response, source=self.id.type),
            task_results_topic_id,
        )
