from uuid import UUID

from pydantic import BaseModel, Field

from src.app.shared.domain.documents import MAX_DOCUMENTS_IDS


class DocumentIDs(BaseModel):
    documents_ids: list[UUID] = Field(max_length=MAX_DOCUMENTS_IDS)
