from typing import List, Optional, Union

from fastapi import APIRouter, Depends, Response
from sqlalchemy.sql import select

from src.app.models.db_models import CorpusEmbedding
from src.app.models.documents import Collection_schema, Document
from src.app.models.search import EnhancedSearchQuery, SearchFilter, SearchQuery
from src.app.services.exceptions import EmptyQueryError, bad_request
from src.app.services.search import SearchService
from src.app.services.search_helpers import (
    search_all_base,
    search_items_base,
    search_multi_inputs,
)
from src.app.services.sql_db import session_maker
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)

sp = SearchService()


def get_params(
    body: SearchQuery,
    nb_results: int = 30,
    subject: Optional[str] = None,
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
    response_model=List[Collection_schema],
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


@router.post(
    "/collections/{collection_query}",
    summary="search items in a specific collection",
    description="Search items in a specific collection",
    response_model=Union[List[Document], None],
)
async def search_items(
    query: Optional[str] = None,
    collection_query: str = "conversation",
    nb_results: int = 10,
    sdg_filter: Optional[SearchFilter] = None,
):
    if not query:
        e = EmptyQueryError()
        return bad_request(message=e.message, msg_code=e.msg_code)

    return await search_items_base(
        query=query,
        collection_query=collection_query,
        nb_results=nb_results,
        sdg_filter=sdg_filter,
        search_func=sp.search_group_by_document,
    )


@router.post(
    "/by_slices",
    summary="search all slices",
    description="Search slices in all collections or in collections specified",
    response_model=Union[List[Document], None],
)
async def search_all_slices_by_lang(
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
):
    return await search_all_base(
        response=response,
        qp=qp,
        search_func=sp.search,
    )


@router.post(
    "/multiple_by_slices",
    summary="search all slices",
    description="Search slices in all collections or in collections specified",
    response_model=Union[List[Document], None],
)
async def multi_search_all_slices_by_lang(
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
):
    if isinstance(qp.query, str):
        qp.query = [qp.query]

    return await search_multi_inputs(
        response=response,
        nb_results=qp.nb_results,
        inputs=qp.query,
        callback_function=sp.search,
    )


@router.post(
    "/by_document",
    summary="search all documents",
    description="Search documents in all collections or in collections specified",
    response_model=Union[List[Document], None],
)
async def search_all(
    response: Response,
    qp: EnhancedSearchQuery = Depends(get_params),
):
    return await search_all_base(
        response=response,
        qp=qp,
        search_func=sp.search_group_by_document,
    )
