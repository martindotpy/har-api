from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

notebook_router = APIRouter(tags=["notebook"])
notebook_folder = Path(__file__).parent.parent / "assets" / "notebook"


@notebook_router.get("/notebook/{file_path:path}")
def get_notebook_file(file_path: str) -> FileResponse:
    """Health check endpoint.

    Returns:
        dict[str, str]: A dictionary indicating the health status.

    """
    file = notebook_folder / file_path

    if not file.exists() or not file.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file)
