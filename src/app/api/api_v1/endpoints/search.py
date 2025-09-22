import uuid
from fastapi import APIRouter, Depends, Response, Request, Header
from qdrant_client.models import ScoredPoint
from sqlalchemy.sql import select
from typing import Annotated

from src.app.models.db_models import CorpusEmbedding, QtyDocumentInQdrant
from src.app.models.documents import Collection_schema
from src.app.models.search import (
    EnhancedSearchQuery,
    SDGFilter,
    SearchMethods,
    SearchQuery,
)
from src.app.services.exceptions import (
    CollectionNotFoundError,
    EmptyQueryError,
    bad_request,
)
from src.app.services.search import SearchService
from src.app.services.search_helpers import search_multi_inputs
from src.app.services.sql_db import session_maker, register_endpoint
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)

sp = SearchService()


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
        return bad_request(message=e.message, msg_code=e.msg_code)

    return resp


@router.get(
    "/collections",
    summary="get all collections",
    description="Get all collections available in the database",
    response_model=list[Collection_schema],
)
async def get_corpus():
    statement = select(
        CorpusEmbedding.source_name, CorpusEmbedding.lang, CorpusEmbedding.title
    )
    with session_maker() as s:
        collections = s.execute(statement).all()

    return [
        {
            "name": name,
            "lang": lang,
            "model": model,
            "corpus": f"{name}_{lang}_{model}",
        }
        for name, lang, model in collections
    ]


@router.get(
    "/nb_docs",
    summary="Get total number of documents",
    description="Returns the total number of documents stored in Qdrant",
)
async def get_nb_docs() -> dict[str, int]:
    statement = select(QtyDocumentInQdrant.document_in_qdrant)
    with session_maker() as s:
        result = s.execute(statement).first()
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
    response: Response,
    request: Request,
    query: str,
    collection: str = "conversation",
    nb_results: int = 10,
    sdg_filter: SDGFilter | None = None,
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
        res = await sp.search_handler(qp=qp, method=SearchMethods.BY_DOCUMENT)

        if not res:
            response.status_code = 206
            return []


        return res
    except CollectionNotFoundError as e:
        response.status_code = 404
        return e.message


@router.post(
    "/by_slices",
    summary="search all slices",
    description="Search slices in all collections or in collections specified",
    response_model=list[ScoredPoint] | None | str,
)
async def search_all_slices_by_lang(
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
):
    try:

        res = await sp.search_handler(qp=qp, method=SearchMethods.BY_SLICES)

        if not res:
            logger.debug("No results found")
            response.status_code = 404
            return []

        return res
    except CollectionNotFoundError as e:
        response.status_code = 404
        return e.message


@router.post(
    "/multiple_by_slices",
    summary="search all slices",
    description="Search slices in all collections or in collections specified",
    response_model=list[ScoredPoint] | None,
)
async def multi_search_all_slices_by_lang(
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
):
    if isinstance(qp.query, str):
        qp.query = [qp.query]

    results = await search_multi_inputs(
        qp=qp,
        callback_function=sp.search_handler,
    )
    if not results:
        logger.error("No results found")
        # todo switch to 204 no content
        response.status_code = 404
        return []

    return results


@router.post(
    "/by_document",
    summary="search all documents",
    description="Search by documents, returns only one result by document id",
    response_model=list[ScoredPoint] | None | str,
)
async def search_all(
    response: Response,
    request: Request,
    X_Session_id: Annotated[uuid.UUID, Header()],
    qp: EnhancedSearchQuery = Depends(get_params),
):
    try:
        res = await sp.search_handler(qp=qp, method=SearchMethods.BY_DOCUMENT)

        if not res:
            logger.error("No results found")
            response.status_code = 206
            return []
    except CollectionNotFoundError as e:
        response.status_code = 404
        return e.message

    response.status_code = 200

    register_endpoint(endpoint=request.url.path, session_id=X_Session_id, http_code=response.status_code)

    return res
