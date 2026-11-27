import os
from typing import Generator

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

from db.models import *  # noqa: F401, F403 - register all SQLModel tables

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SQLITE_PATH = os.path.join(BASE_DIR, "db", "database.db")


def _build_database_url() -> str:
    raw_url = (os.getenv("DATABASE_URL") or "").strip()
    if not raw_url:
        return f"sqlite:///{DEFAULT_SQLITE_PATH}"

    # SQLAlchemy 2.x requires 'postgresql://' rather than legacy 'postgres://'
    if raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql://", 1)

    # Ensure SSL is required when connecting to Supabase PostgreSQL
    if (
        raw_url.startswith("postgresql://")
        and ("supabase.co" in raw_url or "supabase.com" in raw_url)
        and "sslmode=" not in raw_url
    ):
        separator = "&" if "?" in raw_url else "?"
        raw_url = f"{raw_url}{separator}sslmode=require"

    return raw_url


DATABASE_URL = _build_database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")
DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"

engine_kwargs: dict = {"echo": DB_ECHO}
if IS_SQLITE:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update(
        {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "5")),
        }
    )

engine = create_engine(DATABASE_URL, **engine_kwargs)


def create_db_and_tables() -> None:
    if IS_SQLITE:
        os.makedirs(os.path.dirname(DEFAULT_SQLITE_PATH), exist_ok=True)
    SQLModel.metadata.create_all(engine)


# Dependency function to yield a session for use in CRUD/Service functions
def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


if IS_SQLITE and not os.path.exists(DEFAULT_SQLITE_PATH):
    create_db_and_tables()
