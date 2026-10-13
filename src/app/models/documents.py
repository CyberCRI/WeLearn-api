import uuid
from enum import StrEnum, auto
from itertools import groupby
from statistics import mean
from typing import Counter

from pydantic import BaseModel
from qdrant_client.http.models import ScoredPoint
from welearn_database.data.models import Corpus, DocumentSlice, Sdg, WeLearnDocument

from src.app.models.collections import CorpusRelation
from src.app.services.sql_db.queries import get_corpus_and_sub_corpus_repartition


class Collection_schema(BaseModel):
    corpus: str
    name: str
    lang: str
    model: str


class DocumentPayloadModel(BaseModel):
    document_corpus: str
    document_sub_corpus: str | None
    document_desc: str
    document_details: dict[
        str, list[dict] | list[str] | str | int | float | dict | None
    ]
    document_id: uuid.UUID
    document_lang: str
    document_sdg: list[int]
    document_title: str
    document_url: str
    slice_content: str | None
    slice_sdg: int | None


class Document(BaseModel):
    score: float
    payload: DocumentPayloadModel
    vector: list[float] | None = None


class JourneySectionType(StrEnum):
    INTRODUCTION = auto()
    TARGET = auto()


class JourneySection(BaseModel):
    type: JourneySectionType
    content: Document


class ContainedSlice:
    def __init__(
        self,
        id_: uuid.UUID,
        content: str,
        order: int,
        vector: list[float],
        score: float,
        sdgs: list[int],
    ):
        self.id = id_
        self.content = content
        self.order = order
        self.vector = vector
        self.score = score
        self.sdgs = sdgs


class FullContainer:
    def __init__(
        self,
        wl_document: WeLearnDocument,
        wl_slices: list[DocumentSlice],
        scored_points: list[ScoredPoint],
        sdg_map: dict[uuid.UUID, list[Sdg]],
    ):
        self._mmr_rank = None
        self.id = wl_document.id
        self.title = wl_document.title
        self.description = wl_document.description
        self.url = wl_document.url
        self.details = wl_document.details

        self._corpus = wl_document.corpus
        self._corpus_relation: CorpusRelation | None = None

        self.contained_slices: list[ContainedSlice] = []

        slice_id_score_points = {s.id: s for s in scored_points}
        for wl_slice in wl_slices:
            related_score_point = slice_id_score_points.get(wl_slice.id)
            related_sdgs = [s.sdg_number for s in sdg_map.get(wl_slice.id, [])]
            self.contained_slices.append(
                ContainedSlice(
                    id_=wl_slice.id,
                    content=wl_slice.body,
                    order=wl_slice.order_sequence,
                    vector=related_score_point.vector,
                    score=related_score_point.score,
                    sdgs=related_sdgs,
                )
            )

    @property
    def corpus(self) -> Corpus:
        if not self._corpus_relation:
            self._corpus_relation = get_corpus_and_sub_corpus_repartition(self._corpus)
        return self._corpus_relation.corpus

    @property
    def sub_corpus(self) -> Corpus | None:
        if not self._corpus_relation:
            self._corpus_relation = get_corpus_and_sub_corpus_repartition(self._corpus)
        return self._corpus_relation.sub_corpus

    @property
    def mean_score(self) -> float:
        scores = [x.score for x in self.contained_slices]
        return mean(scores)

    @property
    def mean_vector(self) -> list[float]:
        vectors = [s.vector for s in self.contained_slices]
        len_vector = len(vectors[0])
        ret = []
        for i in range(len_vector):
            ret.append(mean([v[i] for v in vectors]))
        return ret

    @property
    def mmr_rank(self):
        return self._mmr_rank

    @mmr_rank.setter
    def mmr_rank(self, value: int) -> None:
        self._mmr_rank = value

    def n_most_common_sdg(self, n: int) -> list[int]:
        if n <= 0:
            return []

        sdgs = []
        for slice in self.contained_slices:
            sdgs.extend(slice.sdgs)
        ret = Counter(sdgs).most_common(n)
        return [i[0] for i in ret]
