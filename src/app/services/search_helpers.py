import asyncio
from typing import Awaitable, Callable, List, Optional

from fastapi import Response
from qdrant_client.http.models import ScoredPoint

from src.app.models.documents import Document
from src.app.models.search import EnhancedSearchQuery, SDGFilter, SearchFilters
from src.app.services.exceptions import (
    NoResultsError,
    handle_error,
)
from src.app.services.helpers import detect_language_from_entry
from src.app.services.search import (
    get_subject_vector,
    SearchService,
    concatenate_same_doc_id_slices,
    sort_slices_using_mmr,
)
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)
sp = SearchService()



async def search_all_base(
    qp: EnhancedSearchQuery,
    search_func: Callable[..., Awaitable[List[ScoredPoint]]],
) -> Optional[List[ScoredPoint]]:
    assert isinstance(qp.query, str)

    lang = detect_language_from_entry(qp.query)
    subject_vector = get_subject_vector(qp.subject)

    collection = await sp.get_collection_by_language(lang)

    embedding = sp.get_query_embed(
        model=collection.model,
        subject_vector=subject_vector,
        query=qp.query,
        subject_influence_factor=qp.influence_factor,
    )

    data = await search_func(
        collection_info=collection.name,
        embedding=embedding,
        nb_results=qp.nb_results,
        filters=SearchFilters(slice_sdg=qp.sdg_filter, document_corpus=qp.corpora),
    )

    sorted_data = sort_slices_using_mmr(data, theta=qp.relevance_factor)

    if qp.concatenate:
        sorted_data = concatenate_same_doc_id_slices(sorted_data)

    return sorted_data


async def search_multi_inputs(
    response: Response,
    inputs: List[str],
    nb_results: int,
    sdg_filter: list[int] | None,
    callback_function: Callable[..., Awaitable[List[ScoredPoint]]],
    collections: tuple[str, ...] | None,
):
    try:
        qps: list[EnhancedSearchQuery] = [
            EnhancedSearchQuery(
                nb_results=nb_results,
                sdg_filter=sdg_filter,
                corpora=collections,
                query=input,
            )
            for input in inputs
        ]
        tasks = [
            search_all_base(
                search_func=callback_function,
                qp=qp,
            )
            for qp in qps
        ]

        all_data: list[ScoredPoint] = []
        for coroutine in asyncio.as_completed(tasks):
            data = await coroutine
            if data:
                all_data.extend(data)

        return all_data
    except Exception as e:
        handle_error(response=response, exc=e)
    return None


# todo: clean code
# andle tests
