"""
Authentication service with JWT tokens and password hashing.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .models import User, UserRole, TokenData, UserCreate
from ...common.config.settings import settings


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.get("jwt_secret_key", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.get("access_token_expire_minutes", 30)


class AuthService:
    """Authentication service."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(
        user_id: str,
        username: str,
        role: UserRole,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode = {
            "user_id": user_id,
            "username": username,
            "role": role.value,
            "exp": expire,
            "iat": datetime.utcnow()
        }

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_access_token(token: str) -> Optional[TokenData]:
        """Decode and verify JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            user_id: str = payload.get("user_id")
            username: str = payload.get("username")
            role: str = payload.get("role")
            exp: int = payload.get("exp")

            if user_id is None or username is None or role is None:
                return None

            return TokenData(
                user_id=user_id,
                username=username,
                role=UserRole(role),
                exp=datetime.fromtimestamp(exp)
            )
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user = db.query(User).filter(User.username == username).first()

        if not user:
            return None

        if not AuthService.verify_password(password, user.hashed_password):
            return None

        if not user.is_active:
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()

        return user

    @staticmethod
    def create_user(db: Session, user_create: UserCreate) -> User:
        """Create a new user."""
        hashed_password = AuthService.hash_password(user_create.password)

        user = User(
            id=secrets.token_urlsafe(16),
            email=user_create.email,
            username=user_create.username,
            full_name=user_create.full_name,
            hashed_password=hashed_password,
            role=user_create.role,
            is_active=True,
            is_verified=False
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def create_session(
        db: Session,
        user: User,
        token: str,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Create user session (stub - sessions not tracked)."""
        # TODO: Implement session tracking when needed
        return None

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def validate_session(db: Session, token: str):
        """Validate session token (stub - using JWT validation only)."""
        # Sessions not tracked in DB yet - rely on JWT expiration
        return True

    @staticmethod
    def revoke_session(db: Session, token: str) -> bool:
        """Revoke session (stub)."""
        # TODO: Implement when session tracking is added
        return True

    @staticmethod
    def cleanup_expired_sessions(db: Session):
        """Clean up expired sessions (stub)."""
        # TODO: Implement when session tracking is added
        pass
