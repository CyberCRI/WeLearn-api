from typing import Annotated

import backoff
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Request,
    Response,
    UploadFile,
)

from src.app.baml_client.async_client import b, types
from src.app.core.config import Settings
from src.app.search.helpers.search_helpers import search_multi_inputs
from src.app.search.models.search import EnhancedSearchQuery
from src.app.search.services.search import SearchService, get_search_service
from src.app.services.data_collection import get_data_collection_service
from src.app.shared.domain.exceptions import NoResultsError
from src.app.shared.infra.abst_chat import get_chat_service
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.requests import extract_session_cookie
from src.app.shared.utils.utils import get_files_content

# from src.app.tutor.service.agents import TEMPLATES
from src.app.tutor.service.agents import TEMPLATES
from src.app.tutor.service.models import (
    CompetencyMappingRequest,
    CourseDescriptionRequest,
    ExtractorOutputList,
    IntegrateSustainabilityRequest,
    LearningObjectivesRequest,
    LearningOutcomesRequest,
    SummariesList,
    SyllabusFeedback,
    SyllabusGenerationRequest,
    SyllabusResponse,
    SyllabusResponseAgent,
    SyllabusUserUpdate,
    TutorSearchResponse,
    TutorSyllabusRequest,
    UserInput,
)
from src.app.tutor.service.orchestrator import SyllabusOrchestrator
from src.app.tutor.service.prompts import (
    extractor_system_prompt,
    extractor_user_prompt,
    summaries_schema,
)
from src.app.tutor.service.tutor import tutor_manager
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()


def backoff_hdlr(details):
    logger.info(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs "
        "{kwargs}".format(**details)
    )


def with_backoff():
    return backoff.on_exception(
        wait_gen=backoff.expo,
        exception=Exception,
        logger=logger,
        max_tries=3,
        max_time=180,
        jitter=backoff.random_jitter,
        on_backoff=backoff_hdlr,
        factor=2,
    )


@with_backoff()
@router.post("/files/content")
async def extract_files_content(
    files: Annotated[list[UploadFile], File()],
    response: Response,
    chatfactory=Depends(get_chat_service),
    lang: str = "en",
) -> ExtractorOutputList | None:
    files_content = await get_files_content(files)
    files_content_str = ("__DOCUMENT_SEPARATOR__").join(files_content)

    messages = [
        {
            "role": "system",
            "content": extractor_system_prompt.format(
                json_schema=summaries_schema, lang=lang
            ),
        },
        {
            "role": "user",
            "content": extractor_user_prompt.format(documents=files_content_str),
        },
    ]

    try:
        summaries_output = await chatfactory.run_llm_with_json_parsing(
            messages,
            ExtractorOutputList,
            fallback_formatter="{extracts: [ 'summary': 'Summary', 'themes': [{'theme': 'Theme 1', 'reason': 'Reason for Theme 1'}, {'theme': 'Theme 2', 'reason': 'Reason for Theme 2'}, ...]}, { 'summary': 'Sumamry', 'themes': [{'theme': 'Theme 1', 'reason': 'Reason for Theme 1'}, {'theme': 'Theme 2', 'reason': 'Reason for Theme 2'}, ...]}] }",
        )

        return summaries_output

    except Exception as e:
        logger.error(f"Error in extractor schema: {e}")
        response.status_code = 204
        raise e


@router.post("/search_extracts")
@with_backoff()
async def tutor_search_extract(
    summaries: SummariesList,
    background_tasks: BackgroundTasks,
    response: Response,
    sp: SearchService = Depends(get_search_service),
    nb_results: int = 15,
):

    try:
        qp = EnhancedSearchQuery(
            query=summaries.summaries,
            nb_results=nb_results,
            sdg_filter=None,
            corpora=None,
        )

        search_results = await search_multi_inputs(
            qp=qp,
            background_tasks=background_tasks,
            callback_function=sp.search_handler,
        )
    except NoResultsError as e:
        response.status_code = 404
        logger.error(f"No results found: {e}")
        return TutorSearchResponse(
            extracts=[],
            nb_results=0,
            documents=[],
        )

    if not search_results:
        return TutorSearchResponse(
            extracts=[],
            nb_results=0,
            documents=[],
        )

    resp = TutorSearchResponse(
        extracts=[],
        nb_results=len(search_results),
        documents=search_results,
    )

    return resp


@with_backoff()
@router.post("/syllabus")
async def create_syllabus(
    request: Request,
    body: TutorSyllabusRequest,
    lang: str = "en",
    data_collection=Depends(get_data_collection_service),
    settings: Settings = Depends(get_settings),
) -> SyllabusResponse:
    session_id = extract_session_cookie(request)
    results = await tutor_manager(body, lang, settings)

    # TODO: handle errors

    message_id = await data_collection.register_syllabus_data(
        session_id=session_id,
        input_data=body,
        agent_answer=results[0].content if results else "",
        feature="syllabus_creation",
    )

    return SyllabusResponse(
        syllabus=results,
        documents=body.documents,
        extracts=body.extracts,
        syllabus_message_id=message_id,
    )


feedback_prompt = """
You are a pedagogical engineer and are  given a syllabus and a feedback. by the teacher that will teach the course.
Your responsibility is to analyze the syllabus and return an improved version of it in a markdown format. Do not add the backticks and the markdown mention.
It is important to take into account the feedback given by the teacher and to keep the syllabus structure.
The syllabus structure is:
    {syllabus_structure}

To be able to do that, the assistant gives you:
    - the syllabus of the course
    - the feedback given by the teacher
    - a list of documents related to the course
    - a list of extracts from a document of interest
    - the themes that the course is related to

You will respond with the syllabus. Do not provide explanations or notes
"""

feedback_assistant_prompt = """
IMPORTANT: you must follow the syllabus structure given by the system message.
and the respect the format of the original syllabus.
Keep the same language as the original syllabus.

here is the original syllabus:
    {syllabus}

take into account the user feedback:
    {feedback}

keep the references section with the format <a href="document.url">document.title</a>, references are based on these documents:
    {documents}

for more context, here are the extracts of the original document the user sent to build the syllabus from. Extracts:
    {extracts}

and the themes extracted from those documents:
    {themes}
"""


@with_backoff()
@router.post("/syllabus/feedback")
async def handle_syllabus_feedback(
    request: Request,
    body: SyllabusFeedback,
    chatfactory=Depends(get_chat_service),
    data_collection=Depends(get_data_collection_service),
):
    session_id = extract_session_cookie(request)

    messages = [
        {
            "role": "system",
            "content": feedback_prompt.format(syllabus_structure=TEMPLATES),
        },
        {
            "role": "user",
            "content": feedback_assistant_prompt.format(
                syllabus=body.syllabus[0],
                feedback=body.feedback,
                documents=body.documents,
                extracts="\n".join([extract.summary for extract in body.extracts]),
                themes=(", ").join(
                    [
                        (", ").join([theme["theme"] for theme in extract.themes])
                        for extract in body.extracts
                    ]
                ),
            ),
        },
    ]

    try:
        syllabus = await chatfactory.chat_client.completion(messages=messages)

        if not isinstance(syllabus, str):
            raise ValueError("Syllabus feedback response is not a string")

        await data_collection.register_syllabus_data(
            session_id=session_id,
            input_data=body,
            agent_answer=syllabus,
            feature="syllabus_feedback",
        )

        return SyllabusResponse(
            syllabus=[SyllabusResponseAgent(content=syllabus)],
            documents=body.documents,
            extracts=body.extracts,
        )

    except Exception as e:
        logger.error(f"Error in chat schema: {e}")
        raise e


@router.post("/syllabus/user_update")
async def register_syllabus_user_update(
    request: Request,
    body: SyllabusUserUpdate,
    data_collection=Depends(get_data_collection_service),
):
    session_id = extract_session_cookie(request)

    await data_collection.register_syllabus_update(
        session_id=session_id,
        syllabus_content=body.syllabus,
    )

    return {"message": "user update registered"}


@router.post("/syllabus/description")
async def baml_test(body: CourseDescriptionRequest) -> types.CourseDescription:
    b_response = await b.GenerateCourseDescription(
        mode=body.mode,
        metadata=types.CourseMetadata(
            discipline=body.course_metadata.discipline,
            topic=body.course_metadata.topic,
            level=body.course_metadata.level,
            num_sessions=body.course_metadata.num_sessions,
            session_duration=body.course_metadata.session_duration,
            session_type=body.course_metadata.session_type,
            session_mode=body.course_metadata.session_mode,  # type: ignore
            class_size=body.course_metadata.class_size,
            output_language=body.course_metadata.output_language,
            user_description=body.course_metadata.user_description,
        ),
        context_text=body.context_text,
        output_language=body.course_metadata.output_language,
    )
    return b_response


@with_backoff()
@router.post("/syllabus/learning_objectives")
async def baml_learning_objectives(
    body: LearningObjectivesRequest,
) -> types.LearningObjectives:
    b_response = await b.GenerateLearningObjectives(
        description=body.description,
        mode=body.mode,
        metadata=types.CourseMetadata(
            discipline=body.course_metadata.discipline,
            topic=body.course_metadata.topic,
            level=body.course_metadata.level,
            num_sessions=body.course_metadata.num_sessions,
            session_duration=body.course_metadata.session_duration,
            session_type=body.course_metadata.session_type,
            session_mode=body.course_metadata.session_mode,  # type: ignore
            class_size=body.course_metadata.class_size,
            output_language=body.course_metadata.output_language,
        ),
        context_text=body.context_text,
        output_language=body.course_metadata.output_language,
    )

    return b_response


@with_backoff()
@router.post("/syllabus/sustainability_integration")
async def integrate_sustainability(
    body: IntegrateSustainabilityRequest,
) -> types.SustainabilityIntegration:
    print(body.course_metadata)
    # session_id = request.headers.get("X-Session-ID")

    metadata = types.CourseMetadata(
        discipline=body.course_metadata.discipline,
        topic=body.course_metadata.topic,
        level=body.course_metadata.level,
        num_sessions=body.course_metadata.num_sessions,
        session_duration=body.course_metadata.session_duration,
        session_type=body.course_metadata.session_type,
        session_mode=body.course_metadata.session_mode,  # type: ignore
        class_size=body.course_metadata.class_size,
        output_language=body.course_metadata.output_language,
    )

    b_response = await b.IntegrateSustainability(
        description=body.description,
        objectives=types.LearningObjectives(
            objectives=[
                types.LearningObjective(
                    number=obj.number,
                    text=obj.text,
                    bloom_level=obj.bloom_level,
                )
                for obj in body.objectives.objectives
            ]
        ),
        metadata=metadata,
        sdg_resources=[
            types.Document(
                text=res.text,
                metadata=res.metadata,
                relevance_score=res.relevance_score if res.relevance_score else None,
            )
            for res in body.sdg_resources
        ],
        mode=body.mode,
        output_language=body.course_metadata.output_language,
    )

    return b_response


@with_backoff()
@router.post("/syllabus/learning_outcomes")
async def baml_learning_outcomes(
    body: LearningOutcomesRequest,
) -> types.LearningOutcomes:
    b_response = await b.GenerateLearningOutcomes(
        metadata=types.CourseMetadata(
            discipline=body.course_metadata.discipline,
            topic=body.course_metadata.topic,
            level=body.course_metadata.level,
            num_sessions=body.course_metadata.num_sessions,
            session_duration=body.course_metadata.session_duration,
            session_type=body.course_metadata.session_type,
            session_mode=body.course_metadata.session_mode,  # type: ignore
            class_size=body.course_metadata.class_size,
            output_language=body.course_metadata.output_language,
        ),
        output_language=body.course_metadata.output_language,
        sustainability_map=body.sustainability_map,
        objectives=types.LearningObjectives(
            objectives=[
                types.LearningObjective(
                    number=obj.number,
                    text=obj.text,
                    bloom_level=obj.bloom_level,
                )
                for obj in body.objectives.objectives
            ]
        ),
    )

    return b_response


@with_backoff()
@router.post("/syllabus/competency_map")
async def generate_competency_map(
    body: CompetencyMappingRequest,
) -> types.CompetencyMappings:
    # session_id = request.headers.get("X-Session-ID")

    competencies = await b.MapCompetencies(
        outcomes=types.LearningOutcomes(
            outcomes=[
                types.LearningOutcome(
                    number=outcome.number,
                    text=outcome.text,
                    related_objectives=[],  # This field is not used in the BAML function, so we can leave it empty
                    assessment_method="",  # This field is not used in the BAML function, so we can leave it empty
                )
                for outcome in body.outcomes
            ]
        ),
        output_language=body.output_language,
        greencomp_framework=body.framework,
    )

    return competencies


@router.post("/api/generate")
async def generate_syllabus(body: SyllabusGenerationRequest):

    try:
        metadata = types.CourseMetadata(
            discipline=body.course_metadata.discipline,
            topic=body.course_metadata.topic,
            level=body.course_metadata.level,
            num_sessions=body.course_metadata.num_sessions,
            session_duration=body.course_metadata.session_duration,
            session_type=body.course_metadata.session_type,
            session_mode=body.course_metadata.session_mode,
            class_size=body.course_metadata.class_size,
            output_language=body.course_metadata.output_language,
        )

        user_input = UserInput(
            metadata=metadata,
            mode=body.mode,
            rag_resources=body.rag_resources,
            provided_description=(
                body.course_metadata.user_description
                if body.course_metadata.user_description
                else None
            ),
        )
        orchestrator = SyllabusOrchestrator()
        output = await orchestrator.run(user_input)

        return {
            "status": "success",
            "validation": {
                "passed": True,
                "major_issues": [],
                "minor_issues": [],
                "suggestions": [],
            },  # Placeholder for validation results
            **output,
        }
    except Exception as e:
        logger.error(f"Error generating syllabus: {e}")
        return {
            "status": "error",
            "message": str(e),
            "validation": {
                "passed": False,
                "major_issues": [str(e)],
                "minor_issues": [],
                "suggestions": [],
            },
        }
