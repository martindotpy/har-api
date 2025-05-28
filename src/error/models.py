from pydantic import BaseModel


class ValidationErrorDetail(BaseModel):
    """Represents a validation error detail."""

    loc: list[str | int]
    msg: str
    type: str


class HTTPValidationError(BaseModel):
    """Represents an HTTP validation error."""

    detail: list[ValidationErrorDetail]


class HTTPError(BaseModel):
    """Represents an HTTP error."""

    detail: str
