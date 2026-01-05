# src/app/services/search.py

import time
from typing import Tuple, cast

import numpy as np
from fastapi import Depends, Request
from fastapi.concurrency import run_in_threadpool
from numpy import ndarray
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http import models as http_models
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.app.models.collections import Collection
from src.app.models.search import (
    EnhancedSearchQuery,
    FilterDefinition,
    SearchFilters,
    SearchMethods,
)
from src.app.services.exceptions import CollectionNotFoundError, ModelNotFoundError
from src.app.services.helpers import convert_embedding_bytes
from src.app.services.sql_service import get_subject
from src.app.utils.decorators import log_time_and_error, log_time_and_error_sync
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


async def get_qdrant(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant


class SearchService:
    import threading

    model = {}

    def __init__(self, client):
        logger.debug("SearchService=init_searchService")
        self.client = client
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

    @staticmethod
    def flavored_with_subject(
        sdg_emb: ndarray, subject_emb: ndarray, discipline_factor: int | float = 2
    ):
        embedding = sdg_emb + (discipline_factor * subject_emb)

        return embedding

    @log_time_and_error
    async def get_collections(self) -> Tuple[str, ...]:
        collections = await self.client.get_collections()
        self.collections = tuple(
            collection.name for collection in collections.collections
        )
        logger.info("collections=%s", self.collections)
        return self.collections

    @log_time_and_error
    async def get_collection_by_language(self, lang: str = "mul") -> Collection:
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

    @log_time_and_error_sync
    def _get_info_from_collection_name(self, collection_name: str) -> Collection:
        lang, model = collection_name.replace(self.col_prefix, "").split("_")
        return Collection(lang=lang, model=model, name=collection_name)

    @log_time_and_error_sync
    async def get_query_embed(
        self,
        model: str,
        query: str,
        subject_vector: list[float] | None = None,
        subject_influence_factor: float = 1.0,
    ) -> np.ndarray:
        embedding = await self._embed_query(query, model)

        if subject_vector:
            embedding = self.flavored_with_subject(
                sdg_emb=embedding,
                subject_emb=subject_vector,
                discipline_factor=subject_influence_factor,
            )

            logger.debug(
                "Adding subject vector influence_factor=%s",
                subject_influence_factor,
            )

        return embedding

    @log_time_and_error_sync
    def _get_model(self, curr_model: str) -> dict:
        # Thread-safe model loading and caching
        if curr_model in self.model:
            return self.model[curr_model]
        try:
            print('>>>>>>>>>>>>>>>>>>>>')
            time_start = time.time()
            # TODO: path should be an env variable
            model = SentenceTransformer(f"../models/embedding/{curr_model}/")
            self.model[curr_model] = {
                "max_seq_length": model.get_max_seq_length(),
                "instance": model,
            }
            time_end = time.time()

            logger.info(
                "method=get_model latency=%s model=%s",
                round(time_end - time_start, 2),
                curr_model,
            )
        except ValueError:
            logger.error("api_error=MODEL_NOT_FOUND model=%s", curr_model)
            raise ModelNotFoundError()
        return self.model[curr_model]

    @log_time_and_error_sync
    def _split_input_seq_len(self, seq_len: int, input: str) -> list[str]:
        if not seq_len:
            raise ValueError("Sequence length value is not valid")

        if len(input) <= seq_len:
            return [input]

        input_words = input.split()
        inputs = []
        curr_seq: str = input_words[0]
        for word in input_words[1:]:
            if len(curr_seq) + len(word) <= seq_len:
                curr_seq = curr_seq + f" {word}"
            else:
                inputs.append(curr_seq.strip())
                curr_seq = word

        if curr_seq:
            inputs.append(curr_seq.strip())

        return inputs

    @log_time_and_error_sync
    async def _embed_query(self, search_input: str, curr_model: str) -> np.ndarray:
        logger.debug("Creating embeddings model=%s", curr_model)
        time_start = time.time()
        if curr_model not in self.model:
            self._get_model(curr_model)

        seq_len = self.model[curr_model]["max_seq_length"]
        model = self.model[curr_model]["instance"]
        inputs = self._split_input_seq_len(seq_len, search_input)

        try:
            embeddings = await run_in_threadpool(model.encode, inputs)
            # embeddings = model.encode(sentences=inputs)
            embeddings = np.mean(embeddings, axis=0)
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

    async def simple_search_handler(
            self,
            qp: EnhancedSearchQuery
            ):
        model = await run_in_threadpool(self._get_model, curr_model="granite-embedding-107m-multilingual")
        model_instance = model['instance']
        embedding = await run_in_threadpool(model_instance.encode, qp.query)
        result = await self.search(
                collection_info="collection_welearn_mul_granite-embedding-107m-multilingual",
                embedding=embedding,
                nb_results=30
                )

        return result

    @log_time_and_error
    async def search_handler(
        self, qp: EnhancedSearchQuery, method: SearchMethods = SearchMethods.BY_SLICES
    ) -> list[http_models.ScoredPoint]:
        assert isinstance(qp.query, str)

        collection = await self.get_collection_by_language(lang="mul")
        subject_vector = await run_in_threadpool(get_subject_vector, qp.subject)
        embedding = await self.get_query_embed (
            model=collection.model,
            query=qp.query,
            subject_vector=subject_vector,
            subject_influence_factor=qp.influence_factor,
        )

        # embedding = [0.049987275,0.04785869,-0.021510484,0.015238845,0.018591229,-0.012600919,0.025832081,0.0005433896,-0.03597837,0.051383518,-0.005686089,0.022538887,0.05297212,0.03222598,0.030791527,0.04426355,-0.0694498,-0.00565751,0.014864093,0.034637913,0.044148076,0.04201736,0.064112954,-0.011100708,-0.19178922,0.00187254,0.1037741,-0.00645192,0.020949572,0.03605938,0.03643103,0.00043291005,0.05828419,-0.08315432,-0.102733605,0.026146093,-0.0110145,0.0055463063,0.01576909,0.07627406,0.023534346,0.005309002,0.012557643,0.08540956,0.01604243,-0.039152242,0.032488924,-0.0020820773,0.017954636,-0.026919981,-0.025180824,0.04390012,-0.0043573556,0.04504469,-0.012268467,0.038814478,0.0040594796,0.0029402429,-0.02380883,0.028509747,0.004087014,0.041373964,0.045721132,0.05641647,0.07393443,-0.0012816414,-0.02319111,-0.00089557073,0.027971193,-0.022518348,0.07223412,0.054478507,0.030545434,0.036976576,0.06611776,0.18475257,-0.015086186,-0.031988166,-0.044567697,0.029626375,0.09986318,0.009391292,0.030026685,0.020191217,0.09890805,0.18790029,-0.01828645,0.012527724,0.02154056,0.012938439,0.016866632,0.014903305,0.026707504,0.007886832,0.054003544,0.050609842,-0.25583458,0.010745114,0.049883965,-0.007095737,0.055308,0.014106844,-0.004310428,0.016197747,0.023646072,-0.011860886,0.014185364,0.048141476,0.055713203,0.0596933,0.121606395,-0.021451375,0.02475858,0.024296043,0.014458568,0.006148835,0.023800103,0.01749048,0.022842212,0.01705037,0.025711475,0.0058475495,0.059756134,0.0050629154,-0.017637372,0.047793955,0.02691839,-0.025728816,0.03182989,-0.085264415,0.034255628,-0.0018601939,0.037861057,0.04273244,0.017540967,-0.02800376,0.027991591,0.009038762,-0.011161276,0.08670358,-0.021121288,-0.093277454,0.055243775,0.042672835,-0.065887146,0.008352424,0.012101927,0.059602745,0.002964636,0.0029458138,0.040898602,0.027603174,0.09611371,0.025623087,0.059096392,0.052753776,0.0517581,0.05863239,0.021987524,0.041949194,-0.02365657,0.019705513,-0.055574693,0.03750193,0.08980106,0.06181546,0.028064243,0.08038597,0.0031036828,0.039561104,-0.027965264,-0.040692486,0.018571734,0.006028422,0.098076336,-0.035969194,-0.014065342,0.015492974,0.0055635655,0.10601647,0.04247313,-0.02212567,0.023426482,0.01786058,-0.016981965,0.013728997,0.09295916,-0.04476623,0.01755914,0.06952539,0.064954296,0.08885461,-0.03427526,-0.0033800644,-0.01743231,0.0099793365,0.028777288,-0.03194725,0.017474106,0.02243706,0.037019197,-0.011065656,-0.077229746,0.0062980526,0.025028022,-0.0076323277,0.06266369,0.06835804,0.035101276,-0.018555624,-0.05480254,-0.005808755,0.023345495,0.00033683557,0.014842423,0.015582394,-0.009580413,0.0047217025,-0.02095926,0.04197348,0.07151979,0.04723259,0.0029915997,0.014750157,0.028415939,0.026752807,0.008502906,0.0015074041,0.0029820295,-0.112886906,0.045829225,0.07617795,0.05909385,0.05823271,0.0034231003,-0.05250317,-0.00016068456,0.07143429,0.031993337,0.008188158,0.024158072,0.0008511741,0.024923284,0.00510406,0.011779183,0.05562784,0.09705153,-0.0149990395,0.059656583,-0.0066453526,0.022248833,0.03471138,-0.046187088,-0.004898068,0.026626432,0.16767602,0.037592273,0.014521678,-0.009666635,-0.004218361,0.019604528,0.04296006,0.027959447,0.07724517,0.0017243444,0.019838225,0.09142305,0.0152593125,0.045357615,0.023832586,0.010326789,0.111930855,-0.12603767,0.0047025555,0.028510377,0.01229013,0.025225984,0.019829933,0.050275527,0.065341055,0.019456618,-0.12311401,-0.035176184,0.04264648,0.047447067,0.018034518,0.01034674,-0.010025917,0.018647775,-0.09339026,0.00020907584,0.007795478,0.0035876548,0.055496518,0.036946736,0.04650201,0.027638914,-0.0021364363,0.011118179,0.015180203,0.078340724,-0.013788043,0.03286299,0.08039025,-0.048537094,0.006743794,-0.029251566,0.041721594,0.07259037,0.044788018,-0.05053859,-0.0036784743,0.021406945,0.054073785,0.04264001,-0.0055695293,-0.035805985,0.023218896,0.020362763,0.014852337,0.038528286,-0.009602926,0.07408133,0.0129254805,0.005253085,0.08015224,0.053607646,-0.08427196,0.094638854,0.024174618,0.100035764,-0.007481447,0.08885887,0.034382984,-0.014909978,0.03151468,-0.038760148,0.10007381,0.03524178,0.010494562,0.010239562,0.015023033,0.033422746,0.061052494,-0.06101102,0.02706595,-0.09865235,0.027603492,0.029072909,0.06061424,0.031207219,-0.0059469156,0.03003269,-0.13649338,0.03568019,-0.0222212,0.042833015,-0.034120306,0.098128274,0.043379553,-0.09582961,0.0014761128,-0.025659285,0.05281996,0.017461082,0.03361553,0.061774824,-0.032325648,0.048860274,0.03009949,0.10000992,-0.13419971,0.020790055,0.05419631,0.06463346,0.030819586,0.00033004582,0.0018264992,0.02057477,0.0453175,0.046780422,-0.103836544,-0.117962375,0.0063544377]

        filter_content = [
            FilterDefinition(key="document_corpus", value=qp.corpora),
            FilterDefinition(key="document_details.readability", value=qp.readability),
            FilterDefinition(
                key=(
                    "slice_sdg" if method == SearchMethods.BY_SLICES else "document_sdg"
                ),
                value=qp.sdg_filter,
            ),
        ]

        filters = SearchFilters(filters=filter_content).build_filters()

        data = []
        if method == "by_slices":
            data = await self.search(
                collection_info=collection.name,
                embedding=embedding,
                filters=filters,
                nb_results=qp.nb_results,
            )
        elif method == "by_document":
            data = await self.search_group_by_document(
                collection_info=collection.name,
                embedding=embedding,
                filters=filters,
                nb_results=qp.nb_results,
            )
        else:
            raise ValueError(f"Unknown search method: {method}")

        # sorted_data = sort_slices_using_mmr(data, theta=qp.relevance_factor)

        # if qp.concatenate:
        #     sorted_data = concatenate_same_doc_id_slices(sorted_data)

        return data

    @log_time_and_error
    async def search_group_by_document(
        self,
        collection_info: str,
        embedding: np.ndarray,
        filters: qdrant_models.Filter | None = None,
        nb_results: int = 100,
    ) -> list[http_models.ScoredPoint]:
        logger.debug("search_group_by_document collection=%s", collection_info)
        try:
            resp = await self.client.search_groups(
                query_filter=filters,
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
        embedding,
        filters: qdrant_models.Filter | None = None,
        nb_results: int = 100,
        with_vectors: bool = True,
    ) -> list[http_models.ScoredPoint]:
        try:
            resp = await self.client.search(
                query_filter=filters,
                collection_name=collection_info,
                query_vector=embedding,
                limit=nb_results,
                with_vectors=with_vectors,
                with_payload=self.payload_keys,
                score_threshold=0.5,
                search_params=qdrant_models.SearchParams(indexed_only=True),
            )
            logger.debug(
                "method=search collection=%s nb_results=%s", collection_info, len(resp)
            )
        except qdrant_exceptions.ResponseHandlingException:
            return []
        return resp


@log_time_and_error_sync
def sort_slices_using_mmr(
    qdrant_results: list[http_models.ScoredPoint],
    theta: float = 1.0,
) -> list[http_models.ScoredPoint]:
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


async def get_search_service(
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
) -> SearchService:
    return SearchService(qdrant)


@log_time_and_error_sync
def concatenate_same_doc_id_slices(
    qdrant_results: list[http_models.ScoredPoint],
) -> list[http_models.ScoredPoint]:
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


@log_time_and_error_sync
def get_subject_vector(subject: str | None) -> list[float] | None:
    """
    Get the subject vector from the database.
    Args:
        subject: The subject to get. If None, return None.

    Returns: The subject vector as a list of floats, or None if not found.

    """
    if not subject:
        return None

    subject_from_db = get_subject(subject=subject)
    if not subject_from_db:
        return None

    embedding = convert_embedding_bytes(subject_from_db.embedding)
    return embedding.tolist()
