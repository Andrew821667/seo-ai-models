"""
Authentication and Authorization models with roles.

Uses existing User model from web.api.database.models
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr

# Use existing database models
from ...web.api.database.models import User, UserRole


class Permission(str):
    """Granular permissions - using string constants instead of Enum"""

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
    UserRole.VIEWER: [  # Legacy support
        Permission.VIEW_ANALYSIS,
        Permission.VIEW_SYSTEM_STATS,
    ],
}


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


def has_permission(role: UserRole, permission: str) -> bool:
    """Check if role has permission."""
    return permission in ROLE_PERMISSIONS.get(role, [])


def get_user_permissions(role: UserRole) -> List[str]:
    """Get all permissions for a role."""
    return ROLE_PERMISSIONS.get(role, [])
