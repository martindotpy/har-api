import logging
from pathlib import Path

logger = logging.getLogger(__name__)

notebook_static_folder = Path.cwd() / "static"


def get_static_file_of_notebook(file_path: str) -> Path:
    """Get the static file path for a notebook.

    Args:
        file_path (str): The path to the notebook file relative to the static folder.

    Returns:
        Path: The path to the static file.

    Raises:
        ValueError: If the file path is invalid.
        FileNotFoundError: If the static file does not exist.

    """
    # Ensure the file path is safe and does not escape the static folder
    if ".." in file_path or file_path.startswith("/"):
        msg = f"Invalid file path: {file_path}"
        logger.error(msg)

        raise ValueError(msg)

    file = notebook_static_folder / file_path

    if not file.exists() or not file.is_file():
        msg = f"File not found: {file_path}"
        logger.error(msg)

        raise FileNotFoundError(msg)

    return file
