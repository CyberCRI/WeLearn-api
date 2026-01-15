# src/app/api/api_v1/endpoints/search.py

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from qdrant_client.models import ScoredPoint

from src.app.models.documents import Document
from src.app.models.search import (
    EnhancedSearchQuery,
    SDGFilter,
    SearchMethods,
    SearchQuery,
)
from src.app.services.exceptions import (
    CollectionNotFoundError,
    EmptyQueryError,
    ModelNotFoundError,
    bad_request,
)
from src.app.services.search import (
    SearchService,
    get_search_service,
    MIX_NOT_ALLOWED_CORPUS,
)
from src.app.services.search_helpers import search_multi_inputs
from src.app.services.sql_db.queries import (
    get_collections_sync,
    get_documents_payload_by_ids_sync,
    get_nb_docs_sync,
)
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)


def get_params(
    body: SearchQuery,
    nb_results: int = 30,
    subject: str | None = None,
    influence_factor: float = 2,
    relevance_factor: float = 1,
    concatenate: bool = True,
) -> EnhancedSearchQuery:
    resp = EnhancedSearchQuery(
        query=body.query or "",
        nb_results=nb_results,
        corpora=tuple(body.corpora) if body.corpora else None,
        subject=subject,
        influence_factor=influence_factor,
        relevance_factor=relevance_factor,
        concatenate=concatenate,
        sdg_filter=body.sdg_filter,
    )

    if not resp.query:
        e = EmptyQueryError()
        bad_request(message=e.message, msg_code=e.msg_code)

    return resp


@router.get("/collections")
async def get_corpus():
    collections = await run_in_threadpool(get_collections_sync)

    return [
        {
            "name": name,
            "lang": lang,
            "model": model,
            "corpus": f"{name}_{lang}_{model}",
            "is_allowed": name not in MIX_NOT_ALLOWED_CORPUS,
        }
        for name, lang, model in collections
    ]


@router.get("/nb_docs")
async def get_nb_docs() -> dict[str, int]:
    result = await run_in_threadpool(get_nb_docs_sync)

    if not result:
        return {"nb_docs": 0}
    return {"nb_docs": result.document_in_qdrant}


@router.post(
    "/collections/{collection}",
    summary="search documents in a specific collection",
    description="Search documents in a specific collection",
    response_model=list[ScoredPoint] | str | None,
)
async def search_doc_by_collection(
    background_tasks: BackgroundTasks,
    response: Response,
    query: str,
    collection: str = "conversation",
    nb_results: int = 10,
    sdg_filter: SDGFilter | None = None,
    sp: SearchService = Depends(get_search_service),
):
    if not query:
        e = EmptyQueryError()
        return bad_request(message=e.message, msg_code=e.msg_code)

    qp = EnhancedSearchQuery(
        query=query,
        nb_results=nb_results,
        corpora=(collection,),
        sdg_filter=sdg_filter.sdg_filter if sdg_filter else None,
    )

    try:
        res = await sp.search_handler(
            qp=qp, method=SearchMethods.BY_DOCUMENT, background_tasks=background_tasks
        )

        if not res:
            response.status_code = 206
            return []

        return res
    except (CollectionNotFoundError, ModelNotFoundError) as e:
        raise HTTPException(
            status_code=404,
            detail={"message": e.message, "code": e.msg_code},
        )


@router.post(
    "/by_slices",
    summary="search all slices",
    description="Search slices in all collections or in collections specified",
    response_model=list[ScoredPoint] | None | str,
)
async def search_all_slices_by_lang(
    background_tasks: BackgroundTasks,
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
    sp: SearchService = Depends(get_search_service),
):
    try:

        res = await sp.search_handler(
            qp=qp, method=SearchMethods.BY_SLICES, background_tasks=background_tasks
        )

        if not res:
            logger.debug("No results found")
            response.status_code = 204
            return []

        return res
    except CollectionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"message": e.message, "code": e.msg_code},
        )


@router.post(
    "/multiple_by_slices",
    summary="search all slices",
    description="Search slices in all collections or in collections specified",
    response_model=list[ScoredPoint] | None,
)
async def multi_search_all_slices_by_lang(
    background_tasks: BackgroundTasks,
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
    sp: SearchService = Depends(get_search_service),
):
    if isinstance(qp.query, str):
        qp.query = [qp.query]

    results = await search_multi_inputs(
        qp=qp,
        background_tasks=background_tasks,
        callback_function=sp.search_handler,
    )
    if not results:
        logger.error("No results found")
        response.status_code = 204
        return []

    return results


@router.post(
    "/by_document",
    summary="search all documents",
    description="Search by documents, returns only one result by document id",
    response_model=list[ScoredPoint] | None | str,
)
async def search_all(
    background_tasks: BackgroundTasks,
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
    sp: SearchService = Depends(get_search_service),
):
    try:
        res = await sp.search_handler(
            qp=qp, method=SearchMethods.BY_DOCUMENT, background_tasks=background_tasks
        )

        if not res:
            logger.error("No results found")
            response.status_code = 204
            return []
    except CollectionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"message": e.message, "code": e.msg_code},
        )

    response.status_code = 200

    return res


@router.post(
    "/documents/by_ids",
    summary="Get documents payload by ids",
    description="Get documents payload by list of document ids",
)
async def get_documents_payload_by_ids(documents_ids: list[str]) -> list[Document]:
    docs = await run_in_threadpool(get_documents_payload_by_ids_sync, documents_ids)
    return docs
