from uuid import UUID
from pydantic import BaseModel


class RowCorpusQtyDocInfo(BaseModel):
    corpus: str
    url: str
    qty_total: int
    qty_in_qdrant: int


class DocumentClickUpdateResponse(BaseModel):
    message_id: UUID
    doc_id: UUID
