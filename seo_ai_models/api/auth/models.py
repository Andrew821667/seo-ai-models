"""
Authentication and Authorization models with roles.

Roles:
- ADMIN: Full access, user management
- USER: Standard access, can run analyses
- OBSERVER: Read-only access, can view results
- ANALYST: Can run analyses and view all results
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Column, String, DateTime, Boolean, Table, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship

from ..infrastructure.database import Base


class UserRole(str, Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    ANALYST = "analyst"
    USER = "user"
    OBSERVER = "observer"


class Permission(str, Enum):
    """Granular permissions."""
    # Analysis permissions
    RUN_ANALYSIS = "run_analysis"
    VIEW_ANALYSIS = "view_analysis"
    DELETE_ANALYSIS = "delete_analysis"

    # User management
    CREATE_USER = "create_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    VIEW_USERS = "view_users"
    ASSIGN_ROLES = "assign_roles"

    # AutoFix permissions
    EXECUTE_AUTOFIX = "execute_autofix"
    APPROVE_AUTOFIX = "approve_autofix"

    # System permissions
    VIEW_SYSTEM_STATS = "view_system_stats"
    MANAGE_SETTINGS = "manage_settings"


# Role to Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.DELETE_ANALYSIS,
        Permission.CREATE_USER,
        Permission.UPDATE_USER,
        Permission.DELETE_USER,
        Permission.VIEW_USERS,
        Permission.ASSIGN_ROLES,
        Permission.EXECUTE_AUTOFIX,
        Permission.APPROVE_AUTOFIX,
        Permission.VIEW_SYSTEM_STATS,
        Permission.MANAGE_SETTINGS,
    ],
    UserRole.ANALYST: [
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.DELETE_ANALYSIS,
        Permission.EXECUTE_AUTOFIX,
        Permission.APPROVE_AUTOFIX,
        Permission.VIEW_SYSTEM_STATS,
    ],
    UserRole.USER: [
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.EXECUTE_AUTOFIX,
    ],
    UserRole.OBSERVER: [
        Permission.VIEW_ANALYSIS,
        Permission.VIEW_SYSTEM_STATS,
    ],
}


# Database Models
class User(Base):
    """User database model."""
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    analyses = relationship("Analysis", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")


class UserSession(Base):
    """Active user sessions."""
    __tablename__ = "user_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String)
    user_agent = Column(String)

    user = relationship("User", back_populates="sessions")


# Pydantic Schemas
class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema."""
    password: str
    role: UserRole = UserRole.USER


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User in database schema."""
    id: str
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    """Public user response schema."""
    id: str
    email: str
    username: str
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT Token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    username: str
    role: UserRole
    exp: datetime


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    old_password: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    """Reset password request."""
    email: EmailStr


def has_permission(role: UserRole, permission: Permission) -> bool:
    """Check if role has permission."""
    return permission in ROLE_PERMISSIONS.get(role, [])


def get_user_permissions(role: UserRole) -> List[Permission]:
    """Get all permissions for a role."""
    return ROLE_PERMISSIONS.get(role, [])
