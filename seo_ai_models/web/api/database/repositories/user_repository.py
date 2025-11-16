"""
User repository for database operations.
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from ..models import User

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for User operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, user_data: dict) -> User:
        """Create a new user."""
        user = User(**user_data)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        logger.info(f"Created user: {user.username}")
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(User.username == username).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()

    def get_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination."""
        return self.db.query(User).offset(skip).limit(limit).all()

    def update(self, user_id: str, user_data: dict) -> Optional[User]:
        """Update user."""
        user = self.get_by_id(user_id)
        if not user:
            return None

        for key, value in user_data.items():
            if hasattr(user, key) and value is not None:
                setattr(user, key, value)

        self.db.commit()
        self.db.refresh(user)
        logger.info(f"Updated user: {user.username}")
        return user

    def delete(self, user_id: str) -> bool:
        """Delete user."""
        user = self.get_by_id(user_id)
        if not user:
            return False

        self.db.delete(user)
        self.db.commit()
        logger.info(f"Deleted user: {user.username}")
        return True
