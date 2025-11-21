from typing import Generator
from sqlmodel import create_engine, Session, SQLModel
from db.models import *  # Import all models

sqlite_file_name = "db/database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


# Dependency function to yield a session for use in CRUD/Service functions
def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
