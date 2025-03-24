from typing import NamedTuple

from pydantic import BaseModel


class Collection_schema(BaseModel):
    corpus: str
    name: str
    lang: str
    model: str


class Collection(NamedTuple):
    name: str
    lang: str
    model: str
    alias: str
