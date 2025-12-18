from pydantic import BaseModel


class RowCorpusQtyDocInfo(BaseModel):
    corpus: str
    url: str
    qty_total: int
    qty_in_qdrant: int
