from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel  # type: ignore

from src.app.api.dependencies import get_settings
from src.app.services.tutor.agents import (
    PedagogicalEngineerAgent,
    SDGExpertAgent,
    UniversityTeacherAgent,
)
from src.app.services.tutor.models import (
    MessageWithResources,
    SyllabusResponseAgent,
    TutorSyllabusRequest,
)
from src.app.services.tutor.utils import extract_doc_info

settings = get_settings()


GREENCOMP_COMPETENCIES = (
    "Here are the GreenComp competencies: "
    "url: https://joint-research-centre.ec.europa.eu/greencomp-european-sustainability-competence-framework_en "
    "1.1 Valuing sustainability: To reflect on personal values; identify and explain how values vary among people "
    "and over time, while critically evaluating how they align with sustainability values. "
    "1.2 Supporting fairness: To support equity and justice for current and future generations and learn from previous "
    "generations for sustainability. "
    "1.3 Promoting nature: To acknowledge that humans are part of nature; and to respect the needs and rights of other "
    "species and of nature itself in order to restore and regenerate healthy and resilient ecosystems. "
    "2.1 Systems thinking: To approach a sustainability problem from all sides; to consider time, space and context in "
    "order to understand how elements interact within and between systems. "
    "2.2 Critical thinking: To assess information and arguments, identify assumptions, challenge the status quo, and "
    "reflect on how personal, social and cultural backgrounds influence thinking and conclusions. "
    "2.3 Problem framing: To formulate current or potential challenges as a sustainability problem in terms of "
    "difficulty, people involved, time and geographical scope, in order to identify suitable approaches to anticipating "
    "and preventing problems, and to mitigating and adapting to already existing problems. "
    "3.1 Futures literacy: To envision alternative sustainable futures by imagining and developing alternative scenarios "
    "and identifying the steps needed to achieve a preferred sustainable future. "
    "3.2 Adaptability: To manage transitions and challenges in complex sustainability situations and make decisions "
    "related to the future in the face of uncertainty, ambiguity and risk. "
    "3.3 Exploratory thinking: To adopt a relational way of thinking by exploring and linking different disciplines, "
    "using creativity and experimentation with novel ideas or methods. "
    "4.1 Political agency: To navigate the political system, identify political responsibility and accountability for "
    "unsustainable behaviour, and demand effective policies for sustainability. "
    "4.2 Collective action: To act for change in collaboration with others. "
    "4.3 Individual initiative: To identify own potential for sustainability and to actively contribute to improving "
    "prospects for the community and the planet.The weather should be in metric units"
)


def _build_chat_model() -> AzureAIChatCompletionsModel:
    return AzureAIChatCompletionsModel(
        endpoint=settings.AZURE_APIM_API_BASE,
        credential=settings.AZURE_APIM_API_KEY,
        model=settings.LLM_MODEL_NAME,
        temperature=0.4,
    )


async def tutor_manager(
    content: TutorSyllabusRequest, lang: str
) -> list[SyllabusResponseAgent]:
    formatted_content = MessageWithResources(
        lang=lang,
        content=content.extracts,
        resources=extract_doc_info(content.documents),
        themes=[theme for extract in content.extracts for theme in extract.themes],
        summary=[extract.summary for extract in content.extracts],
        course_title=content.course_title,
        level=content.level,
        duration=content.duration,
        description=content.description,
    )

    chat_model = _build_chat_model()
    teacher_agent = UniversityTeacherAgent(chat_model)
    sdg_agent = SDGExpertAgent(chat_model, GREENCOMP_COMPETENCIES)
    pedagogical_agent = PedagogicalEngineerAgent(chat_model, GREENCOMP_COMPETENCIES)

    teacher_response = await teacher_agent.generate(formatted_content)
    sdg_response = await sdg_agent.enhance(
        teacher_response, formatted_content.resources, lang
    )
    pedagogical_response = await pedagogical_agent.refine(sdg_response)

    return [teacher_response, sdg_response, pedagogical_response]
