import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import DEV, HOST, ORIGINS, PATH, PORT
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

# Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEV,
    )
