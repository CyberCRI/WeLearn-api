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

from src.app.shared.utils.dependencies import get_settings
from src.app.core.config import Settings
from src.app.models.search import EnhancedSearchQuery
from src.app.services.abst_chat import get_chat_service
from src.app.services.data_collection import get_data_collection_service
from src.app.services.exceptions import NoResultsError
from src.app.services.search import SearchService, get_search_service
from src.app.services.search_helpers import search_multi_inputs
from src.app.services.tutor.agents import TEMPLATES
from src.app.services.tutor.models import (
    ExtractorOutputList,
    SummariesList,
    SyllabusFeedback,
    SyllabusResponse,
    SyllabusResponseAgent,
    SyllabusUserUpdate,
    TutorSearchResponse,
    TutorSyllabusRequest,
)
from src.app.services.tutor.prompts import (
    extractor_system_prompt,
    extractor_user_prompt,
    summaries_schema,
)
from src.app.services.tutor.tutor import tutor_manager
from src.app.services.tutor.utils import get_files_content
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
    session_id = request.headers.get("X-Session-ID")
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
    session_id = request.headers.get("X-Session-ID")

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
    session_id = request.headers.get("X-Session-ID")

    await data_collection.register_syllabus_update(
        session_id=session_id,
        syllabus_content=body.syllabus,
    )

    return {"message": "user update registered"}
