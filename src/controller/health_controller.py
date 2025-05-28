from fastapi import APIRouter

health_router = APIRouter(
    tags=["health"],
)


@health_router.get("/health")
def health_check() -> str:
    """Health check endpoint.

    Returns:
        str: A simple ok.

    """
    return "Ok"
