"""
Модели данных для аутентификации и авторизации.
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from enum import Enum
from datetime import datetime


class UserRole(str, Enum):
    """Роли пользователей в системе."""

    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class UserCreate(BaseModel):
    """Модель для создания пользователя."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER

    @validator("username")
    def username_alphanumeric(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric")
        return v


class UserLogin(BaseModel):
    """Модель для входа пользователя."""

    username_or_email: str
    password: str


class TokenResponse(BaseModel):
    """Модель ответа с токеном доступа."""

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user_id: str


class UserResponse(BaseModel):
    """Модель ответа с данными пользователя."""

    id: str
    username: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class UserUpdate(BaseModel):
    """Модель для обновления данных пользователя."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class PasswordChange(BaseModel):
    """Модель для изменения пароля."""

    current_password: str
    new_password: str = Field(..., min_length=8)

    @validator("new_password")
    def password_strength(cls, v):
        # Проверка сложности пароля
        has_digit = any(c.isdigit() for c in v)
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)

        if not (has_digit and has_upper and has_lower):
            raise ValueError(
                "Password must contain at least one digit, one uppercase and one lowercase letter"
            )
        return v
