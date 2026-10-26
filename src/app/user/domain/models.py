import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class InstitutionData(BaseModel):
    institution: str
    role: str


class InstitutionDataResponse(InstitutionData):
    message: str


class UserAndSessionResponse(BaseModel):
    message: str
    session_id: uuid.UUID


class UserResponse(BaseModel):
    user_id: uuid.UUID


class SessionResponse(BaseModel):
    session_id: uuid.UUID


class BookmarkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    inferred_user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class BookmarksResponse(BaseModel):
    bookmarks: list[BookmarkRead]


class BookmarksDeletedResponse(BaseModel):
    deleted: int


class BookmarkDeletedResponse(BaseModel):
    deleted: uuid.UUID


class BookmarkAddedResponse(BaseModel):
    added: uuid.UUID
