from fastapi import APIRouter, Response, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from src.app.api.dependencies import get_settings
from src.app.models.metric import RowCorpusQtyDocInfo
from src.app.services.sql_db.queries import get_document_qty_table_info_sync
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()


@router.get("/nb_docs_info_per_corpus")
async def get_nb_docs_info_per_corpus(
    response: Response,
) -> list[RowCorpusQtyDocInfo | None]:
    result = await run_in_threadpool(get_document_qty_table_info_sync)
    if not result:
        return []

    ret = []
    for r_corpus, r_qty_doc_in_qdrant, r_qty_doc_total in result:
        try:
            current = RowCorpusQtyDocInfo(
                corpus=r_corpus.source_name,
                url=r_corpus.main_url,
                qty_total=r_qty_doc_total.count,
                qty_in_qdrant=r_qty_doc_in_qdrant.count,
            )
            ret.append(current)
        except ValidationError as e:
            response.status_code = status.HTTP_206_PARTIAL_CONTENT
            logger.error(f"Validation error for corpus {r_corpus.source_name}: {e}")
    if len(ret) == 0:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return ret
