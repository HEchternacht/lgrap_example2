"""
SQLAlchemy engine, session factory, and Base.

The database URL is read from the DB_URL environment variable.
Defaults to a local SQLite file so the app runs out of the box with
no external dependencies.

Supported URLs (examples):
  sqlite:///./lgrap.db                          ← default (local file)
  postgresql+psycopg2://user:pw@host/dbname
  mssql+pyodbc://user:pw@dsn
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_URL: str = os.getenv("DB_URL", "sqlite:///./lgrap.db")

# SQLite needs check_same_thread=False for FastAPI's threaded request handling.
# All other backends ignore this arg.
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(DB_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db():
    """Yield a DB session and close it when the request is done."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Table creation (called once at app startup)
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Create all tables that are not yet in the database."""
    # Import models here so their classes are registered on Base.metadata
    import db_app.models  # noqa: F401  — side-effect import
    Base.metadata.create_all(bind=engine)
