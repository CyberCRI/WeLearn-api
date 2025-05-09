from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from src.app.api.dependencies import get_settings
from src.app.models.search import EnhancedSearchQuery
from src.app.services.abst_chat import AbstractChat
from src.app.services.exceptions import NoResultsError
from src.app.services.search import SearchService
from src.app.services.search_helpers import search_multi_inputs
from src.app.services.tutor.models import (
    ExtractorOuputList,
    SyllabusResponse,
    TutorSearchResponse,
)
from src.app.services.tutor.tutor import tutor_manager
from src.app.services.tutor.utils import get_file_content
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()


chatfactory = AbstractChat(
    model="azure/gpt-4o",
    API_KEY=settings.AZURE_GPT_4O_API_KEY,
    API_BASE=settings.AZURE_GPT_4O_API_BASE,
    API_VERSION=settings.AZURE_GPT_4O_API_VERSION,
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
    files_content: list[bytes] = []

    for file in files:
        file_content = await get_file_content(file)

        if not file_content:
            raise HTTPException(status_code=400, detail="added files are empty")

        files_content.append(file_content)

    doc_list_to_string = "Document {doc_nb}: {content}"

    file_content_str = [
        doc_list_to_string.format(
            doc_nb=index + 1,
            content=content.decode("utf-8", errors="ignore"),
        )
        for index, content in enumerate(files_content)
    ]
    file_content_str = "\n\n".join(file_content_str)

    messages = [
        {"role": "system", "content": extractor_prompt},
        {"role": "assistant", "content": file_content_str},
    ]

    try:
        themes_extracted = await chatfactory.chat_client.completion(
            messages=messages, response_format=ExtractorOuputList
        )

        assert isinstance(themes_extracted, dict)

        themes_extracted = ExtractorOuputList(**themes_extracted)

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
            nb_results=5,
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
    body: TutorSearchResponse, lang: str = "en"
) -> SyllabusResponse:
    results = await tutor_manager(body, lang)

    # TODO: handle errors

    return SyllabusResponse(syllabus=results, documents=body.documents)
