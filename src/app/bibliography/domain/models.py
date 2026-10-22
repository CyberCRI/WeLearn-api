from uuid import UUID

from pydantic import BaseModel


class DocumentIDs(BaseModel):
    documents_ids: list[UUID]
