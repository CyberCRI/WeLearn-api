import asyncio

from autogen_core import (
    AgentId,
    ClosureAgent,
    ClosureContext,
    MessageContext,
    SingleThreadedAgentRuntime,
    TypeSubscription,
)
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.app.api.dependencies import get_settings
from src.app.services.tutor.agents import (
    CLOSURE_AGENT_TYPE,
    TASK_RESULTS_TOPIC_TYPE,
    PedagogicalEngineerAgent,
    SDGExpertAgent,
    UniversityTeacherAgent,
    pedagogical_engineer_topic_type,
    sdg_expert_topic_type,
    university_teacher_topic_type,
)
from src.app.services.tutor.models import (
    MessageWithResources,
    SyllabusFeedback,
    SyllabusResponseAgent,
    TutorSearchResponse,
)
from src.app.services.tutor.utils import extract_doc_info

settings = get_settings()


llm_mistral = OpenAIChatCompletionClient(
    model="mistral-large-latest",
    api_key=settings.MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1/",
    model_info=ModelInfo(
        vision=False,
        function_calling=True,
        json_output=True,
        family=ModelFamily.MISTRAL,
        structured_output=True,
        multiple_system_messages=True,
    ),
)

async def select_manager(content: TutorSearchResponse | SyllabusFeedback):
    if isinstance(content, TutorSearchResponse):
        return await tutor_manager(content, lang="en")
    if isinstance(content, SyllabusFeedback):
        # TODO: Implement feedback manager
        pass

    raise ValueError("Invalid content type provided to select_manager")


async def tutor_manager(
    content: TutorSearchResponse, lang: str
) -> list[SyllabusResponseAgent]:
    queue = asyncio.Queue[SyllabusResponseAgent]()

    formatted_content = MessageWithResources(
        lang=lang,
        content=content.extracts,
        resources=extract_doc_info(content.documents),
        themes=[theme for extract in content.extracts for theme in extract.themes],
        summary=[extract.summary for extract in content.extracts],
    )

    async def collect_result(
        _agent: ClosureContext, message: SyllabusResponseAgent, ctx: MessageContext
    ) -> None:
        await queue.put(message)

    greencomp_memory = ListMemory()
    await greencomp_memory.add(
        MemoryContent(
            content="Here are the GreenComp competencies: "
            "url: https://joint-research-centre.ec.europa.eu/greencomp-european-sustainability-competence-framework_en"
            "1.1 Valuing sustainability: To reflect on personal values; identify and explain how values vary among people and over time, while critically evaluating how they align with sustainability values. "
            "1.2 Supporting fairness: To support equity and justice for current and future generations and learn from previous generations for sustainability. "
            "1.3 Promoting nature: To acknowledge that humans are part of nature; and to respect the needs and rights of other species and of nature itself in order to restore and regenerate healthy and resilient ecosystems. "
            "2.1 Systems thinking: To approach a sustainability problem from all sides; to consider time, space and context in order to understand how elements interact within and between systems. "
            "2.2 Critical thinking: To assess information and arguments, identify assumptions, challenge the status quo, and reflect on how personal, social and cultural backgrounds influence thinking and conclusions. "
            "2.3 Problem framing: To formulate current or potential challenges as a sustainability problem in terms of difficulty, people involved, time and geographical scope, in order to identify suitable approaches  to anticipating and preventing problems, and to mitigating and adapting to already existing problems. "
            "3.1 Futures literacy: To envision alternative sustainable futures by imagining and developing alternative scenarios and identifying the steps needed to achieve a preferred sustainable future. "
            "3.2 Adaptability: To manage transitions and challenges in complex sustainability situations and make decisions related to the future in the face of uncertainty, ambiguity and risk. "
            "3.3 Exploratory thinking: To adopt a relational way of thinking by exploring and linking different disciplines, using creativity and experimentation with novel ideas or methods. "
            "4.1 Political agency: To navigate the political system, identify political responsibility and accountability for unsustainable behaviour, and demand effective policies for sustainability. "
            "4.2 Collective action: To act for change in collaboration with others. "
            "4.3 Individual initiative: To identify own potential for sustainability and to actively contribute to improving prospects for the community and the planet.The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    runtime = SingleThreadedAgentRuntime()

    await UniversityTeacherAgent.register(
        runtime,
        type=university_teacher_topic_type,
        factory=lambda: UniversityTeacherAgent(model_client=llm_mistral),
    )

    await SDGExpertAgent.register(
        runtime,
        type=sdg_expert_topic_type,
        factory=lambda: SDGExpertAgent(
            model_client=llm_mistral, memory=greencomp_memory
        ),
    )

    await PedagogicalEngineerAgent.register(
        runtime,
        type=pedagogical_engineer_topic_type,
        factory=lambda: PedagogicalEngineerAgent(
            model_client=llm_mistral, memory=greencomp_memory
        ),
    )

    runtime.start()

    await ClosureAgent.register_closure(
        runtime,
        CLOSURE_AGENT_TYPE,
        collect_result,
        subscriptions=lambda: [
            TypeSubscription(
                topic_type=TASK_RESULTS_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE
            )
        ],
    )

    await runtime.send_message(
        formatted_content,
        recipient=AgentId(university_teacher_topic_type, "default"),
    )

    await runtime.stop_when_idle()
    responses = []
    while not queue.empty():
        response = await queue.get()
        responses.append(response)

    await runtime.close()
    return [
        SyllabusResponseAgent(content=response.content, source=response.source)
        for response in responses
    ]
