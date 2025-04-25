import asyncio
import json
import time
from functools import cache
from typing import Annotated, Any, Callable, Dict, Literal, List, Optional, Tuple, cast

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http import models as http_models
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.app.api.dependencies import get_settings
from src.app.models.collections import Collection
from src.app.models.search import EnhancedSearchQuery, SearchFilters
from src.app.services.exceptions import (
    CollectionNotFoundError,
    ModelNotFoundError,
)
from src.app.services.helpers import detect_language_from_entry
from src.app.utils.decorators import log_time_and_error
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class DBClientSingleton(AsyncQdrantClient):
    instance = None

    def __new__(cls):
        if cls.instance is None:
            logger.debug("DBClientSingleton=init_DBClient")
            settings = get_settings()

            cls.instance = super().__new__(cls)
            cls.instance = AsyncQdrantClient(
                url=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=100
            )
        return cls.instance


class SearchService:
    def __init__(self):
        logger.debug("SearchService=init_searchService")
        self.client = DBClientSingleton()
        self.collections = None

        self.payload_keys = [
            "document_title",
            "document_id",
            "document_url",
            "document_lang",
            "document_corpus",
            "document_desc",
            "document_sdg",
            "slice_content",
            "slice_sdg",
            "document_scrape_date",
            "document_details.authors",
            "document_details.author",
            "document_details.publisher",
            "document_details.journal",
            "document_details.duration",
            "document_details.readability",
            "document_details.source",
        ]
        self.col_prefix = "collection_welearn_"

    @log_time_and_error
    async def get_collections(self) -> Tuple[str, ...]:
        collections = await self.client.get_collections()
        self.collections = tuple(
            collection.name for collection in collections.collections
        )
        logger.info("collections=%s", self.collections)
        return self.collections

    @log_time_and_error
    async def get_collection_by_language(self, lang: str) -> Collection:
        collections = self.collections or await self.get_collections()

        collection = next(
            (
                collection
                for collection in collections
                if collection.startswith(f"{self.col_prefix}{lang}")
            ),
            None,
        )

        if not collection:
            raise CollectionNotFoundError(
                message=f"No collection found for this language {lang}"
            )

        return self._get_info_from_collection_name(collection)

    def _get_info_from_collection_name(self, collection_name: str) -> Collection:
        lang, model = collection_name.replace(self.col_prefix, "").split("_")
        return Collection(lang=lang, model=model, name=collection_name)

    def get_query_embed(
        self,
        model: str,
        query: str,
        subject_vector: Optional[List[float]] = None,
        subject_influence_factor: float = 1.0,
    ) -> np.ndarray:
        embedding = self._embed_query(query, model)

        if subject_vector:
            embedding = embedding + [
                subject_influence_factor * vec for vec in subject_vector
            ]

            logger.debug(
                "Adding subject vector influence_factor=%s",
                subject_influence_factor,
            )

        return embedding

    @cache
    def _get_model(self, curr_model: str) -> SentenceTransformer:
        try:
            time_start = time.time()
            model = SentenceTransformer(f"../models/embedding/{curr_model}/")
            time_end = time.time()
            logger.info(
                "method=get_model latency=%s model=%s",
                round(time_end - time_start, 2),
                curr_model,
            )
        except ValueError:
            logger.error("api_error=MODEL_NOT_FOUND model=%s", curr_model)
            raise ModelNotFoundError()
        return model

    @cache
    def _embed_query(self, search_input: str, curr_model: str) -> np.ndarray:
        logger.debug("Creating embeddings model=%s", curr_model)
        time_start = time.time()
        model = self._get_model(curr_model)
        try:
            embeddings = model.encode(sentences=search_input)
        except Exception as ex:
            logger.error("api_error=EMBED_ERROR model=%s", curr_model)
            raise RuntimeError("Not able to create embed", "EMBED_ERROR") from ex
        time_end = time.time()
        logger.debug(
            "Creating embeddings time_elapsed=%s query_length=%s model=%s",
            round(time_end - time_start, 2),
            len(search_input),
            curr_model,
        )
        return cast(np.ndarray, embeddings)

    def build_filters(
        self, filters: Optional[SearchFilters] = None
    ) -> Optional[qdrant_models.Filter]:
        if filters is None:
            return None

        qdrant_filter = []
        for key, values in filters.model_dump().items():
            if not values:
                continue

            qdrant_filter.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchAny(any=values),
                )
            )
        print(qdrant_filter)

        return qdrant_models.Filter(must=qdrant_filter)

    async def search_handler(self, qp: EnhancedSearchQuery, method: Literal['by_slices', 'by_document']) -> List[http_models.ScoredPoint]:
        assert isinstance(qp.query, str)

        lang = detect_language_from_entry(qp.query)
        subject_vector = get_subject_vector(qp.subject)
        collection = await self.get_collection_by_language(lang)
        embedding = self.get_query_embed(
                model=collection.model,
                subject_vector=subject_vector,
                query=qp.query,
                subject_influence_factor=qp.influence_factor,
                )

        filters = SearchFilters(slice_sdg=qp.sdg_filter, document_corpus=qp.corpora)
        if method == 'by_slices':
            data = await self.search(
            collection_info=collection.name,
            embedding=embedding,
            filters=filters,
            nb_results=qp.nb_results,
            )
        else:
            data = await self.search_group_by_document(
                collection_info=collection.name,
                embedding=embedding,
                filters=filters,
                nb_results=qp.nb_results,
            )

        sorted_data = sort_slices_using_mmr(data, theta=qp.relevance_factor)

        if qp.concatenate:
            sorted_data = concatenate_same_doc_id_slices(sorted_data)

        return sorted_data



    @log_time_and_error
    async def search_group_by_document(
        self,
        collection_info: str,
        embedding: np.ndarray,
        filters: Optional[SearchFilters] = None,
        nb_results: int = 100,
    ) -> List[http_models.ScoredPoint]:
        logger.debug("search_group_by_document collection=%s", collection_info)
        try:
            resp = await self.client.search_groups(
                query_filter=self.build_filters(filters),
                collection_name=collection_info,
                query_vector=embedding,
                limit=nb_results,
                with_vectors=True,
                group_size=1,
                with_payload=self.payload_keys,
                group_by="document_id",
            )
            return [r.hits[0] for r in resp.groups]
        except (
            qdrant_exceptions.ApiException,
            qdrant_exceptions.ResponseHandlingException,
        ):
            return []

    @log_time_and_error
    async def search(
        self,
        collection_info: str,
        embedding: np.ndarray,
        filters: Optional[SearchFilters] = None,
        nb_results: int = 100,
    ) -> List[http_models.ScoredPoint]:
        try:
            resp = await self.client.search(
                query_filter=self.build_filters(filters),
                collection_name=collection_info,
                query_vector=embedding,
                limit=nb_results,
                with_vectors=True,
                with_payload=self.payload_keys,
                score_threshold=0.5,
            )
            logger.debug(
                "method=search collection=%s nb_results=%s", collection_info, len(resp)
            )
        except qdrant_exceptions.ResponseHandlingException:
            return []
        return resp


# def search_items_method(
#     callback_function: Callable,
#     nb_results: int,
#     embedding: np.ndarray,
#     collection: str,
#     filters: Optional[List[int]] = None,
# ) -> Optional[List[http_models.ScoredPoint]]:
#     return callback_function(
#         collection_info=collection,
#         embedding=embedding,
#         filters=filters,
#         nb_results=nb_results,
#     )


# @log_time_and_error
# async def parallel_search(
#     callback_function: Callable,
#     nb_results: int,
#     collections: List[Dict[str, Any]],
#     sdg_filter: Optional[List[int]] = None,
# ) -> List[http_models.ScoredPoint]:
#     tasks = [
#         callback_function(
#             collection_info=col["alias"],
#             embedding=col["embed"],
#             nb_results=nb_results,
#             filters=sdg_filter,
#         )
#         for col in collections
#     ]
#     data = await asyncio.gather(*tasks)
#     if len(data) < len(collections):
#         raise PartialResponseResultError()
#     return [doc for source in data for doc in source]


def sort_slices_using_mmr(
    qdrant_results: List[http_models.ScoredPoint],
    theta: float = 1.0,
) -> List[http_models.ScoredPoint]:
    if not qdrant_results:
        return []

    logger.debug("sort_slices_using_mmr=start")
    reward = [r.score for r in qdrant_results]
    sim = cosine_similarity(np.array([r.vector for r in qdrant_results]))

    id_s = [0]
    id_r = list(range(1, len(qdrant_results)))

    while id_r:
        marginal_relevance_scores = [
            theta * reward[i] - (1 - theta) * max(sim[i, id_s]) for i in id_r
        ]
        j = np.argmax(marginal_relevance_scores)
        id_s.append(id_r.pop(j))

    logger.debug("sort_slices_using_mmr=end")
    return [qdrant_results[i] for i in id_s]


def concatenate_same_doc_id_slices(
    qdrant_results: List[http_models.ScoredPoint],
) -> List[http_models.ScoredPoint]:
    """
    Concatenate slices on the same document ID and remove duplicates.

    Args:
        qdrant_results (List[http_models.ScoredPoint]): Qdrant results.

    Returns:
        List[http_models.ScoredPoint]: Qdrant results without duplicates and concatenated slices.
    """
    logger.debug("concatenate_same_doc_id_slices=start")
    doc_id_to_slices = {}

    for qresult in qdrant_results:
        if not qresult.payload:
            continue

        curr_doc_id = qresult.payload.get("document_id", "")
        if not curr_doc_id:
            continue

        if curr_doc_id not in doc_id_to_slices:
            doc_id_to_slices[curr_doc_id] = qresult
        else:
            existing_result = doc_id_to_slices[curr_doc_id]
            existing_result.payload[
                "slice_content"
            ] += f"\n\n{qresult.payload.get('slice_content', '')}"

    new_results = list(doc_id_to_slices.values())

    logger.debug(
        "concatenate_same_doc_id_slices=end nb_results_initial=%s nb_docs_final=%s",
        len(qdrant_results),
        len(new_results),
    )

    return new_results


def get_subject_vector(subject: str | None) -> List[float] | None:
    if not subject:
        return None
    with open("src/app/services/subject_vectors.json") as f:
        logger.info("Loading subject vectors: subject=%s", subject)
        vectors = json.load(f)
        vector = vectors.get(subject.lower(), None)

        if not vector:
            logger.error("Subject vector not found, subject=%s", subject)
            return None

        return vector
