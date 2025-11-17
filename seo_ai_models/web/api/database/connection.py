"""
Database connection and session management.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

logger = logging.getLogger(__name__)

# Database URL from environment or default to SQLite for development
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./seo_ai_models.db")

# PostgreSQL URL format: postgresql://user:password@host:port/database
# SQLite URL format: sqlite:///./database.db

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # Needed for SQLite
        echo=False,  # Set to True for SQL query logging
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections
        pool_size=10,
        max_overflow=20,
        echo=False,
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.

    Yields:
        Session: SQLAlchemy database session

    Example:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables.

    Call this once when setting up the application.
    """
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def drop_db():
    """
    Drop all tables (use with caution!).

    Only use in development/testing.
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("All tables dropped")
