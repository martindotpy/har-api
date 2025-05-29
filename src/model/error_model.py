from pydantic import BaseModel


class ValidationErrorDetail(BaseModel):
    """Represents a validation error detail."""

    loc: list[str | int]
    msg: str
    type: str


class HTTPValidationErrorResponse(BaseModel):
    """Represents an HTTP validation error."""

    detail: list[ValidationErrorDetail]


class HTTPErrorResponse(BaseModel):
    """Represents an HTTP error."""

    detail: str
