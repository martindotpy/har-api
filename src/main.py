import uvicorn
from fastapi import FastAPI

from config import DEV, HOST, PATH, PORT
from controller import routers

app = FastAPI(
    docs_url=f"{PATH}/docs",
    redoc_url=f"{PATH}/redoc",
    openapi_url=f"{PATH}/openapi.json",
    generate_unique_id_function=lambda path: path.name,
)

# Routers
for router in routers:
    app.include_router(router, prefix=PATH)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEV,
    )
