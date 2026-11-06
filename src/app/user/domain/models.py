import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

# basic shape check; Keycloak validates the address itself
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# the locales of the Keycloak realm (supportedLocales)
SIGNUP_LOCALES = ("en", "fr")


class SignUpRequest(BaseModel):
    email: str
    first_name: str | None = None
    last_name: str | None = None
    # language of the sign-up page, Keycloak sends its emails in it
    locale: str | None = None

    @field_validator("locale")
    @classmethod
    def normalize_locale(cls, value: str | None) -> str | None:
        # "fr-FR" -> "fr". an unsupported locale is dropped rather than refused:
        # Keycloak then uses the realm default, the sign-up itself still works
        if not value:
            return None
        language = re.split(r"[-_]", value.strip().lower())[0]
        return language if language in SIGNUP_LOCALES else None

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not EMAIL_PATTERN.match(value):
            raise ValueError("Invalid email address")
        return value


class SignUpResponse(BaseModel):
    """
    Same answer whether the email is new or already has an account, so sign-up
    does not reveal which emails are registered.
    """

    message: str


class InstitutionData(BaseModel):
    institution: str
    role: str


class InstitutionDataResponse(InstitutionData):
    message: str


class UserAndSessionResponse(BaseModel):
    message: str
    session_id: uuid.UUID
    user_id: uuid.UUID


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
