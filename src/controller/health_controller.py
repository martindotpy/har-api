from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health", include_in_schema=False)
def health_check() -> str:
    """Health check endpoint."""
    return "Ok"
