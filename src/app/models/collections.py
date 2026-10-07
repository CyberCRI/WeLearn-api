from typing import NamedTuple

from pydantic import BaseModel
from welearn_database.data.models import Corpus


class Collection_schema(BaseModel):
    corpus: str
    name: str
    lang: str
    model: str


class Collection(NamedTuple):
    lang: str
    model: str
    name: str


class CorpusRelation:
    def __init__(self, corpus: Corpus, sub_corpus: Corpus | None = None):
        self.corpus = corpus
        self.sub_corpus = sub_corpus
