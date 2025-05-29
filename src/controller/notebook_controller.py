import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from error.models import HTTPError

logger = logging.getLogger(__name__)

notebook_router = APIRouter(tags=["notebook"])

notebook_static_folder = Path.cwd() / "static"


@notebook_router.get(
    "/notebook/{file_path:path}",
    responses={
        404: {
            "description": "File not found",
            "content": {
                "application/json": {
                    "example": {"detail": "File not found"},
                }
            },
            "model": HTTPError,
        },
    },
)
def get_notebook_file(file_path: str, request: Request) -> FileResponse:
    """Retrieve a notebook file."""
    client_host = request.client.host if request.client else "unknown"
    logger.info("Request from IP: %s", client_host)
    logger.info("Request headers: %s", dict(request.headers))
    logger.info("Retrieving notebook file: %s", file_path)

    # Ensure the file path is safe and does not escape the static folder
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")

    file = notebook_static_folder / file_path

    if not file.exists() or not file.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file)
