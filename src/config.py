import os
from typing import Final

PORT: Final[int] = int(os.environ.get("PORT", "8000"))
HOST: Final[str] = os.environ.get("HOST", "127.0.0.1")
DEV: Final[bool] = os.environ.get("DEV", "false").lower() == "true"
PATH: Final[str] = os.environ.get("PREFIX", "/api")
