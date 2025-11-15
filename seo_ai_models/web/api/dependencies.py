"""
Dependencies for FastAPI application.
"""

import logging
from typing import Generator
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from .database.connection import SessionLocal
from .services.auth_service import AuthService
from .services.project_service_cached import ProjectServiceCached
from .services.webhook_service_cached import WebhookServiceCached
from .services.cache_service import get_cache_service
from .routers.auth import oauth2_scheme

logger = logging.getLogger(__name__)


def get_db() -> Generator[Session, None, None]:
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_auth_service() -> AuthService:
    """Get AuthService instance."""
    return AuthService()


def get_project_service(db: Session = Depends(get_db)) -> ProjectServiceCached:
    """Get ProjectServiceCached instance with Redis caching."""
    return ProjectServiceCached(db)


def get_webhook_service(db: Session = Depends(get_db)) -> WebhookServiceCached:
    """Get WebhookServiceCached instance with Redis caching."""
    return WebhookServiceCached(db)


def get_cache_service_dependency():
    """Get CacheService instance."""
    return get_cache_service()


def get_current_user_id(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service)
) -> str:
    """
    Extract current user ID from JWT token.

    Args:
        token: JWT access token
        auth_service: AuthService instance

    Returns:
        User ID from token

    Raises:
        HTTPException: 401 if token is invalid
    """
    try:
        payload = auth_service.verify_token(token)
        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user_id
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
