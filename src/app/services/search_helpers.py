import asyncio
from typing import Awaitable, Callable, List, Optional

from fastapi import Response
from qdrant_client.http.models import ScoredPoint

from src.app.models.documents import Document
from src.app.models.search import EnhancedSearchQuery, SearchFilter
from src.app.services.exceptions import (
    CollectionNotFoundError,
    NoResultsError,
    handle_error,
)
from src.app.services.helpers import detect_language_from_entry
from src.app.services.search import (
    SearchService,
    concatenate_same_doc_id_slices,
    get_subject_vector,
    parallel_search,
    sort_slices_using_mmr,
)
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)
sp = SearchService()


async def search_items_base(
    query: str,
    collection_query: str,
    nb_results: int,
    sdg_filter: Optional[SearchFilter],
    search_func: Callable[..., Awaitable[List[Document]]],
) -> Optional[List[Document]]:
    logger.info("search_query=%s searched_collection=%s", query, collection_query)

    try:
        lang = detect_language_from_entry(query)
        collection_alias = await sp.get_collection_alias(
            collection_name=collection_query, lang=lang
        )
        col = sp._get_info_from_collection_alias(collection_alias=collection_alias)
        model_embedding = sp.embed_query(search_input=query, curr_model=col.model)

        data = await search_func(
            collection_info=col.alias,
            embedding=model_embedding,
            filters=sdg_filter.sdg_filter if sdg_filter else None,
            nb_results=nb_results,
        )

        if not data:
            raise NoResultsError()

        return data
    except Exception as e:
        return handle_error(response=None, exc=e)


async def search_all_base(
    response: Response,
    qp: EnhancedSearchQuery,
    search_func: Callable[..., Awaitable[List[ScoredPoint]]],
) -> Optional[List[ScoredPoint]]:
    try:
        lang = detect_language_from_entry(qp.query)
        subject_vector = get_subject_vector(qp.subject)

        collections = await sp.get_collections_aliases_by_language(
            lang=lang, collections=qp.corpora
        )
        collections_to_search = [
            sp.get_collection_dict_with_embed(
                collection_alias=col,
                query=qp.query,
                subject_vector=subject_vector,
                subject_influence_factor=qp.influence_factor,
            )
            for col in collections
        ]

        if not collections_to_search:
            raise CollectionNotFoundError()

        logger.info(
            "Found %s collections to search: %s",
            len(collections_to_search),
            collections,
        )

        data = await parallel_search(
            callback_function=search_func,
            nb_results=qp.nb_results,
            collections=collections_to_search,
            sdg_filter=qp.sdg_filter,
        )

        if not data:
            raise NoResultsError()

        sorted_data = sorted(data, key=lambda x: x.score, reverse=True)
        sorted_data = sort_slices_using_mmr(sorted_data, theta=qp.relevance_factor)

        if qp.concatenate:
            sorted_data = concatenate_same_doc_id_slices(sorted_data)

        return sorted_data
    except Exception as e:
        handle_error(response=response, exc=e)
    return None


async def search_multi_inputs(
    response: Response,
    inputs: List[str],
    nb_results: int,
    callback_function: Callable[..., Awaitable[List[ScoredPoint]]],
):
    try:
        qps: list[EnhancedSearchQuery] = [
            EnhancedSearchQuery(
                nb_results=nb_results,
                sdg_filter=None,
                query=input,
            )
            for input in inputs
        ]
        tasks = [
            search_all_base(
                response=response,
                search_func=callback_function,
                qp=qp,
            )
            for qp in qps
        ]

        data = await asyncio.gather(*tasks)
        all_data = [*data[0], *data[1]]

        doc: list[Document] = [
            Document(
                score=d.score,
                payload=d.payload,
            )
            for d in all_data
        ]

        sorted_data = sorted(doc, key=lambda x: x.score, reverse=True)

        return sorted_data
    except Exception as e:
        handle_error(response=response, exc=e)
    return None
