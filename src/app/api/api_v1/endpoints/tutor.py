from typing import Annotated

from fastapi import APIRouter, File, Response, UploadFile

from src.app.api.dependencies import get_settings
from src.app.services.abst_chat import AbstractChat, ChatFactory
from src.app.services.search import SearchService
from src.app.services.search_helpers import search_multi_inputs
from src.app.services.tutor.models import (
    ExtractorOuputList,
    SyllabusResponse,
    TutorSearchResponse,
)
from src.app.services.tutor.tutor import tutor_manager
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()

chatfactory: AbstractChat = ChatFactory().create_chat("openai")
chatfactory.init_client()

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
    file_content: list[bytes] = [await file.read() for file in files]
    doc_list_to_string = "Document {doc_nb}: {content}"

    file_content_str = [
        doc_list_to_string.format(
            doc_nb=index + 1,
            content=content.decode("utf-8", errors="ignore"),
        )
        for index, content in enumerate(file_content)
    ]
    file_content_str = "\n\n".join(file_content_str)

    messages = [
        {"role": "system", "content": extractor_prompt},
        {"role": "assistant", "content": file_content_str},
    ]

    themes_extracted = await chatfactory.chat_schema(
        model="gpt-4o-mini", messages=messages, response_format=ExtractorOuputList  # type: ignore
    )

    if not themes_extracted or not themes_extracted.extracts:
        # handle error
        return TutorSearchResponse(
            extracts=[],
            nb_results=0,
            documents=[],
        )

    inputs = [doc.summary for doc in themes_extracted.extracts]  # type: ignore

    search_results = await search_multi_inputs(
        response=response,
        inputs=inputs,
        nb_results=5,
        sdg_filter=None,
        collections=None,
        callback_function=sp.search,
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

    # TODO: handle duplicates

    return resp


@router.post("/syllabus")
async def create_syllabus(body: TutorSearchResponse) -> SyllabusResponse:
    result = await tutor_manager(body)

    return SyllabusResponse(syllabus=result.content, documents=body.documents)
