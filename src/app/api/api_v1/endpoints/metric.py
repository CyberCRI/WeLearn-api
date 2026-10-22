from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from src.app.metric.infra.queries import get_document_qty_table_info_sync
from src.app.models.metric import (
    DocumentClickUpdateResponse,
    RegisterFormAnswerBody,
    RowCorpusQtyDocInfo,
)
from src.app.services.data_collection import get_data_collection_service
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as utils_logger
from src.app.shared.utils.requests import extract_session_cookie

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()


@router.get("/nb_docs_info_per_corpus")
async def get_nb_docs_info_per_corpus(
    response: Response,
) -> list[RowCorpusQtyDocInfo | None]:
    result = await run_in_threadpool(get_document_qty_table_info_sync)
    if result is None or len(result) == 0:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return []
    ret = []
    for r_corpus, r_qty_doc_in_qdrant, r_qty_doc_total in result:
        try:
            current = RowCorpusQtyDocInfo(
                corpus=r_corpus.source_name,
                url=r_corpus.main_url,
                qty_total=r_qty_doc_total,
                qty_in_qdrant=r_qty_doc_in_qdrant,
            )
            ret.append(current)
        except ValidationError as e:
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
            logger.error(f"Validation error for corpus {r_corpus.source_name}: {e}")
    if len(ret) == 0:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    ret.sort(key=lambda x: x.qty_in_qdrant, reverse=True)
    return ret


@router.post("/clicked_document")
async def update_clicked_doc_from_chat_message(
    body: DocumentClickUpdateResponse,
    request: Request,
    data_collection=Depends(get_data_collection_service),
) -> str:
    await data_collection.register_document_click(body.doc_id, body.message_id)
    return "updated"


@router.post("/syllabus_downloaded")
async def register_syllabus_download(
    request: Request, data_collection=Depends(get_data_collection_service)
) -> str:
    session_id = extract_session_cookie(request)
    await data_collection.register_syllabus_download(session_id)
    return "registered"


@router.post("/form_data")
async def register_form_data(
    request: Request,
    response: Response,
    body: RegisterFormAnswerBody,
    data_collection=Depends(get_data_collection_service),
):
    try:
        session_id = extract_session_cookie(request)

        if not session_id:
            response.status_code = 400
            return "missing session id"

        await data_collection.register_form_answer(
            form_name=body.form_name,
            question=body.question,
            answer=body.answer,
            answer_type=body.answer_type,
            session_id=session_id,
        )
        response.status_code = 200

        return "form data registered"
    except Exception as e:
        logger.error("error while registering form data", e)
        response.status_code = 500
        return "an error occured when registering the form data"
