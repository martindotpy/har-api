from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

notebook_router = APIRouter(tags=["notebook"])
notebook_folder = Path(__file__).parent.parent / "assets" / "notebook"


@notebook_router.get("/notebook/{file_path:path}")
def get_notebook_file(file_path: str) -> FileResponse:
    """Retrieve a notebook file.

    Args:
        file_path (str): The path to the notebook file.

    Returns:
        FileResponse: The notebook file response.

    """
    file = notebook_folder / file_path

    if not file.exists() or not file.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file)
