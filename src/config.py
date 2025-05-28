import os
from typing import Final

PORT: Final[int] = int(os.environ.get("PORT", "8000"))
HOST: Final[str] = os.environ.get("HOST", "127.0.0.1")
IS_DEV: Final[bool] = os.environ.get("DEV", "true").lower() == "true"
PREFIX: Final[str] = os.environ.get("PREFIX", "/api")
ORIGINS: Final[list[str]] = [
    "https://har.martindotpy.dev",
    "http://localhost:4321",
]
