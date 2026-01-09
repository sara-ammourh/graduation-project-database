import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.app_routes import router as app_router
from api.auth_routes import router as api_router

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(title="GoAlgo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api")
app.include_router(app_router)


@app.get("/")
def read_root():
    return {"message": "API is running"}


if __name__ == "__main__":
    import uvicorn

    from db.config import create_db_and_tables

    create_db_and_tables()
    uvicorn.run(app, host="0.0.0.0", port=8000)
