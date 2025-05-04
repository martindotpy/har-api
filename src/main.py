import os
from typing import Final

import uvicorn
from fastapi import FastAPI

from controller import routers

app = FastAPI()

for router in routers:
    app.include_router(router)


if __name__ == "__main__":
    PORT: Final[int] = int(os.environ.get("PORT", "8000"))
    HOST: Final[str] = os.environ.get("HOST", "127.0.0.1")

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
    )
