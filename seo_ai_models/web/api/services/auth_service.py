"""
Authentication service with JWT token support.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for authentication and authorization."""

    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        logger.info("AuthService initialized")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            bool: True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """
        Hash a password.

        Args:
            password: Plain text password

        Returns:
            str: Hashed password
        """
        return pwd_context.hash(password)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.

        Args:
            data: Data to encode in token
            expires_delta: Custom expiration time

        Returns:
            str: Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        logger.debug(f"Created access token for user: {data.get('sub')}")
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create JWT refresh token.

        Args:
            data: Data to encode in token

        Returns:
            str: Encoded JWT refresh token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        logger.debug(f"Created refresh token for user: {data.get('sub')}")
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Optional[Dict[str, Any]]: Decoded token data or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return None

    def get_current_user_id(self, token: str) -> Optional[str]:
        """
        Extract user ID from JWT token.

        Args:
            token: JWT token string

        Returns:
            Optional[str]: User ID or None if invalid
        """
        payload = self.verify_token(token)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                return user_id

        logger.warning("Failed to extract user ID from token")
        return None

    def authenticate_user(
        self, username: str, password: str, get_user_func
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password.

        Args:
            username: Username
            password: Plain text password
            get_user_func: Function to get user by username from database

        Returns:
            Optional[Dict[str, Any]]: User data if authenticated, None otherwise
        """
        user = get_user_func(username)

        if not user:
            logger.warning(f"User not found: {username}")
            return None

        if not self.verify_password(password, user.get("hashed_password", "")):
            logger.warning(f"Invalid password for user: {username}")
            return None

        logger.info(f"User authenticated successfully: {username}")
        return user

    def create_tokens_for_user(self, user_id: str, username: str) -> Dict[str, str]:
        """
        Create both access and refresh tokens for a user.

        Args:
            user_id: User ID
            username: Username

        Returns:
            Dict[str, str]: Dictionary with access_token and refresh_token
        """
        token_data = {"sub": user_id, "username": username}

        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }
