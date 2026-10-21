from uuid import UUID

from pydantic import BaseModel
from welearn_database.data.models import QuestionType


class RowCorpusQtyDocInfo(BaseModel):
    corpus: str
    url: str
    qty_total: int
    qty_in_qdrant: int


class DocumentClickUpdateResponse(BaseModel):
    message_id: UUID
    doc_id: UUID


class RegisterFormAnswerBody(BaseModel):
    form_name: str
    question: str
    answer: str
    answer_type: QuestionType
