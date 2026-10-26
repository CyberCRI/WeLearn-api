from typing import Any

from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class ErrorDetail(BaseModel):
    message: str
    code: str


class ErrorResponse(BaseModel):
    """
    Body of every error the API returns. `detail` is an `ErrorDetail` for
    domain errors, a plain string for most `HTTPException`s, and a list of
    validation errors for 422s.
    """

    detail: ErrorDetail | str | list[dict[str, Any]]


# declared on every API route so generated clients type the error body
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    "default": {"model": ErrorResponse, "description": "Error"},
}
