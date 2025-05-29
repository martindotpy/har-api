from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from models.errors import HTTPErrorResponse
from service.notebook_service import get_static_file_of_notebook

notebook_router = APIRouter(tags=["notebook"])


@notebook_router.get(
    "/notebook/{file_path:path}",
    responses={
        400: {
            "description": "Invalid file path",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid file path: ../example.ipynb"
                    },
                }
            },
            "model": HTTPErrorResponse,
        },
        404: {
            "description": "File not found",
            "content": {
                "application/json": {
                    "example": {"detail": "File not found: example.ipynb"},
                }
            },
            "model": HTTPErrorResponse,
        },
    },
)
def get_notebook_file(file_path: str) -> FileResponse:
    """Retrieve a notebook file."""
    try:
        file = get_static_file_of_notebook(file_path)
        return FileResponse(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
