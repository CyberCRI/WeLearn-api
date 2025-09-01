from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from src.app.api.dependencies import get_settings
from src.app.models.search import EnhancedSearchQuery
from src.app.services.abst_chat import AbstractChat
from src.app.services.exceptions import NoResultsError
from src.app.services.search import SearchService
from src.app.services.search_helpers import search_multi_inputs
from src.app.services.tutor.agents import TEMPLATES
from src.app.services.tutor.models import (
    ExtractorOuputList,
    SyllabusFeedback,
    SyllabusResponse,
    SyllabusResponseAgent,
    TutorSearchResponse,
    TutorSyllabusRequest,
)
from src.app.services.helpers import extract_json_from_response
from src.app.services.tutor.tutor import tutor_manager
from src.app.services.tutor.utils import get_file_content
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()


chatfactory = AbstractChat(
    model="Mistral-Large-2411",
    API_KEY=settings.AZURE_MISTRAL_API_KEY,
    API_BASE=settings.AZURE_MISTRAL_API_BASE,
    API_VERSION="2024-05-01-preview",
    is_azure_model=True,
)

sp = SearchService()


extractor_prompt = """
role="An assistant to summarize a text and extract the main themes from it",
backstory="You are specialised in analysing documents, summarizing them and extracting the main themes. You value precision and clarity.",
goal="Analyse each document, summarize it and extract the main themes, explaining why each theme was identified.",
expected_output="You must follow the following JSON schema: {extracts: [{'original_document': 'Document', 'summary': 'Summary', 'themes': ['Theme 1', 'Theme 2', ...]}, {'original_document': 'Document', 'summary': 'Sumamry', 'themes': ['Theme 1', 'Theme 2', ...]}, ...]} an entry by document",
"""


@router.post("/search")
async def tutor_search(
    files: Annotated[list[UploadFile], File()],
    response: Response,
):
    files_content: list[str] = []

    for file in files:
        file_content = await get_file_content(file)

        if not file_content:
            raise HTTPException(status_code=400, detail="added files are empty")

        files_content.append(file_content)

    doc_list_to_string = "Document {doc_nb}: {content}"

    file_content_str = [
        doc_list_to_string.format(
            doc_nb=index + 1,
            content=content,
        )
        for index, content in enumerate(files_content)
    ]
    file_content_str = "\n\n".join(file_content_str)

    messages = [
        {"role": "system", "content": extractor_prompt},
        {"role": "user", "content": file_content_str},
    ]

    try:
        themes_extracted = await chatfactory.chat_client.completion(
            messages=messages, response_format=ExtractorOuputList
        )

        jsn = {}
        if isinstance(themes_extracted, str):
            jsn = extract_json_from_response(themes_extracted)
        elif isinstance(themes_extracted, dict):
            jsn = themes_extracted
        else:
            raise ValueError("Unexpected response format")

        themes_extracted = ExtractorOuputList(**jsn)

    except Exception as e:
        logger.error(f"Error in chat schema: {e}")
        # todo: handle error
        return TutorSearchResponse(
            extracts=[],
            nb_results=0,
            documents=[],
        )

    if not themes_extracted or not themes_extracted.extracts:
        return TutorSearchResponse(
            extracts=[],
            nb_results=0,
            documents=[],
        )

    inputs = [doc.summary for doc in themes_extracted.extracts]  # type: ignore

    try:
        qp = EnhancedSearchQuery(
            query=inputs,
            nb_results=10,
            sdg_filter=None,
            corpora=None,
        )

        search_results = await search_multi_inputs(
            qp=qp,
            callback_function=sp.search_handler,
        )
    except NoResultsError as e:
        response.status_code = 404
        logger.error(f"No results found: {e}")
        return TutorSearchResponse(
            extracts=themes_extracted.extracts,
            nb_results=0,
            documents=[],
        )

    if not search_results:
        return TutorSearchResponse(
            extracts=themes_extracted.extracts,
            nb_results=0,
            documents=[],
        )

    resp = TutorSearchResponse(
        extracts=themes_extracted.extracts,
        nb_results=len(search_results),
        documents=search_results,
    )

    return resp


@router.post("/syllabus")
async def create_syllabus(
    body: TutorSyllabusRequest, lang: str = "en"
) -> SyllabusResponse:
    results = await tutor_manager(body, lang)

    # TODO: handle errors

    return SyllabusResponse(
        syllabus=results, documents=body.documents, extracts=body.extracts
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

keep the references section with the formar <a href="document.url">document.title</a>, references are based on these documents:
    {documents}

for more context, here are the extracts of the original document the user sent to build the syllaus from. Extracts:
    {extracts}

and the themes extracted from those documents:
    {themes}
"""


@router.post("/syllabus/feedback")
async def handle_syllabus_feedback(body: SyllabusFeedback):

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
                extracts=("/n").join([extract.summary for extract in body.extracts]),
                themes=(", ").join(
                    [(", ").join(extract.themes) for extract in body.extracts]
                ),
            ),
        },
    ]

    try:
        syllabus = await chatfactory.chat_client.completion(messages=messages)

        assert isinstance(syllabus, str)

        return SyllabusResponse(
            syllabus=[SyllabusResponseAgent(content=syllabus)],
            documents=body.documents,
            extracts=body.extracts,
        )

    except Exception as e:
        logger.error(f"Error in chat schema: {e}")
