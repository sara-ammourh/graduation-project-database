from contextlib import asynccontextmanager
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

load_dotenv()

from api.app_routes import router as app_router
from api.auth_routes import router as api_router
from db.config import create_db_and_tables, engine

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:4200,http://127.0.0.1:4200")
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("SKIP_DB_INIT", "false").lower() != "true":
        try:
            create_db_and_tables()
            logger.info("Database tables verified/initialized.")
        except Exception:
            logger.exception("Failed to initialize database tables on startup.")
            raise
    yield


app = FastAPI(title="GoAlgo API", lifespan=lifespan)

cors_origins = _parse_cors_origins()
cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
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


@app.get("/health")
def health_check():
    """Lightweight health check for Render and uptime monitors."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        logger.error("Health check database failure: %s", exc)
        return {"status": "degraded", "database": "disconnected"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
