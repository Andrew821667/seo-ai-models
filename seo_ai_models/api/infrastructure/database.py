"""
Database infrastructure for API v2.

Uses existing database connection from web.api.database.
"""

from seo_ai_models.web.api.database.connection import SessionLocal, engine
from seo_ai_models.web.api.database.models import Base


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ["get_db", "engine", "Base", "SessionLocal"]
