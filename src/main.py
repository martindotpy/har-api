import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, IS_DEV, ORIGINS, PORT, PREFIX
from controller import routers
from logger import configure_logger

app = FastAPI(
    docs_url=f"{PREFIX}/docs",
    redoc_url=f"{PREFIX}/redoc",
    openapi_url=f"{PREFIX}/openapi.json",
    generate_unique_id_function=lambda path: path.name,
)

# Routers
for router in routers:
    app.include_router(router, prefix=PREFIX)

# Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uvicorn logger
configure_logger(logging.getLogger("uvicorn"))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=IS_DEV,
        server_header=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
