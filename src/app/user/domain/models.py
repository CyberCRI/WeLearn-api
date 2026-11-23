import re
import uuid
from datetime import date, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# basic shape check; Keycloak validates the address itself
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# the locales of the Keycloak realm (supportedLocales)
SIGNUP_LOCALES = ("en", "fr")


def normalize_locale(value: str | None) -> str | None:
    # "fr-FR" -> "fr". an unsupported locale is dropped rather than refused:
    # Keycloak then uses the realm default, the sign-up itself still works
    if not value:
        return None
    language = re.split(r"[-_]", value.strip().lower())[0]
    return language if language in SIGNUP_LOCALES else None


def normalize_email(value: str) -> str:
    value = value.strip().lower()
    if not EMAIL_PATTERN.match(value):
        raise ValueError("Invalid email address")
    return value


def normalize_optional_email(value: str | None) -> str | None:
    return normalize_email(value) if value and value.strip() else None


def required_single_line(value: str) -> str:
    # one line, so the text can go in an email subject: no CR/LF, spaces collapsed
    value = " ".join(value.split())
    if not value:
        raise ValueError("Must not be empty")
    return value


Email = Annotated[str, AfterValidator(normalize_email)]
Locale = Annotated[str | None, AfterValidator(normalize_locale)]


class SignUpRequest(BaseModel):
    email: Email
    first_name: str | None = None
    last_name: str | None = None
    # language of the sign-up page, Keycloak sends its emails in it
    locale: Locale = None


class SignUpResponse(BaseModel):
    """
    Same answer whether the email is new or already has an account, so sign-up
    does not reveal which emails are registered.
    """

    message: str


class UniversityRequest(BaseModel):
    """A university missing from the login page, to add as an identity provider."""

    university: Annotated[
        str, Field(max_length=200), AfterValidator(required_single_line)
    ]
    city: Annotated[str, Field(max_length=100), AfterValidator(required_single_line)]
    # optional, to tell the person once the university is added
    email: Annotated[str | None, AfterValidator(normalize_optional_email)] = None
    locale: Locale = None


class UniversityRequestResponse(BaseModel):
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


class LogoutResponse(BaseModel):
    message: str


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


class GuestFeatureQuota(BaseModel):
    limit: int
    remaining: int
    # seconds until the oldest counted use leaves the 24h window (0 if none)
    resets_in: int


class AccountAccessResponse(BaseModel):
    """
    Whether the account's access is time-limited (a Google or email/password
    login, not an institution one) and until when. `access_until` is the last
    day of access, `days_left` counts today; both are null when the account is
    not limited, or when the end date could not be stored yet.
    """

    limited: bool
    access_until: date | None = None
    days_left: int | None = None


class GuestQuotaResponse(BaseModel):
    """
    The caller's daily guest quotas. `limited` is false for an account, or when
    guest restrictions are off: the features are then null.
    """

    limited: bool
    chat: GuestFeatureQuota | None = None
    syllabus: GuestFeatureQuota | None = None
    # changes asked on a generated syllabus
    feedback: GuestFeatureQuota | None = None
    # LLM summaries of uploaded files (tutor step 1)
    files: GuestFeatureQuota | None = None
    # changes proposed for a sustainability connection (syllabus v2)
    adaptation: GuestFeatureQuota | None = None
