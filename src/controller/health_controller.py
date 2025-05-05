from fastapi import APIRouter

health_router = APIRouter(
    tags=["health"],
)


@health_router.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict[str, str]: A dictionary indicating the health status.

    """
    return {"status": "UP"}
