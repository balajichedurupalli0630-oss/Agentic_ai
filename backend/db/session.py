"""
session.py

SQLAlchemy engine, session maker, and database connection helpers.
"""

from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import settings
from logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


engine = None
SessionLocal = None
_db_available = False

try:
    db_url = settings.sync_database_url
    engine = create_engine(db_url, pool_pre_ping=True, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Ping test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    _db_available = True
    logger.info("PostgreSQL database connection established successfully.")
except Exception as e:
    logger.warning(f"PostgreSQL database connection inactive ({e}). DB features will operate in standby mode.")


def is_db_available() -> bool:
    """Check if PostgreSQL database connection is active."""
    return _db_available


def get_db() -> Generator:
    """FastAPI dependency yielding a SQLAlchemy DB session."""
    if not _db_available or SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
