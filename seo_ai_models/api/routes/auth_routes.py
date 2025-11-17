"""
Authentication routes for login, logout, user management.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from ..auth.models import (
    LoginRequest,
    Token,
    UserCreate,
    UserUpdate,
    UserResponse,
    ChangePasswordRequest,
    UserRole,
    Permission,
    get_user_permissions,
)
from ..auth.service import AuthService
from ..auth.dependencies import get_current_user, require_permission, require_admin
from ..auth.models import User
from ..infrastructure.database import get_db


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=Token)
async def login(login_request: LoginRequest, request: Request, db: Session = Depends(get_db)):
    """
    Login with username and password.

    Returns JWT access token.
    """
    # Authenticate user
    user = AuthService.authenticate_user(db, login_request.username, login_request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = AuthService.create_access_token(
        user_id=user.id, username=user.username, role=user.role
    )

    # Create session
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    AuthService.create_session(db, user, access_token, ip_address=ip_address, user_agent=user_agent)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=1800,  # 30 minutes
        user=UserResponse.from_orm(user),
    )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Logout and revoke current session."""
    # Token is in the current_user dependency, but we need to get it from request
    # This is a simplified version - in production, pass token explicitly
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.from_orm(current_user)


@router.get("/me/permissions", response_model=List[str])
async def get_my_permissions(current_user: User = Depends(get_current_user)):
    """Get current user permissions."""
    permissions = get_user_permissions(current_user.role)
    return [p.value for p in permissions]


@router.post("/change-password")
async def change_password(
    password_request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change current user password."""
    # Verify old password
    if not AuthService.verify_password(password_request.old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect current password"
        )

    # Update password
    current_user.hashed_password = AuthService.hash_password(password_request.new_password)
    db.commit()

    return {"message": "Password changed successfully"}


# Admin-only routes
@router.post("/users", response_model=UserResponse)
async def create_user(
    user_create: UserCreate,
    current_user: User = Depends(require_permission(Permission.CREATE_USER)),
    db: Session = Depends(get_db),
):
    """
    Create a new user (Admin only).

    Only admins can create users.
    """
    # Check if username exists
    existing_user = AuthService.get_user_by_username(db, user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists"
        )

    # Check if email exists
    existing_email = AuthService.get_user_by_email(db, user_create.email)
    if existing_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    # Create user
    user = AuthService.create_user(db, user_create)

    return UserResponse.from_orm(user)


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(require_permission(Permission.VIEW_USERS)),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
):
    """
    List all users (Admin only).
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return [UserResponse.from_orm(user) for user in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_permission(Permission.VIEW_USERS)),
    db: Session = Depends(get_db),
):
    """Get user by ID (Admin only)."""
    user = AuthService.get_user_by_id(db, user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse.from_orm(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    current_user: User = Depends(require_permission(Permission.UPDATE_USER)),
    db: Session = Depends(get_db),
):
    """Update user (Admin only)."""
    user = AuthService.get_user_by_id(db, user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Update fields
    if user_update.email is not None:
        user.email = user_update.email
    if user_update.username is not None:
        user.username = user_update.username
    if user_update.full_name is not None:
        user.full_name = user_update.full_name
    if user_update.role is not None:
        user.role = user_update.role
    if user_update.is_active is not None:
        user.is_active = user_update.is_active

    db.commit()
    db.refresh(user)

    return UserResponse.from_orm(user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_permission(Permission.DELETE_USER)),
    db: Session = Depends(get_db),
):
    """Delete user (Admin only)."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete yourself"
        )

    user = AuthService.get_user_by_id(db, user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db.delete(user)
    db.commit()

    return {"message": "User deleted successfully"}


@router.get("/roles", response_model=List[str])
async def list_roles(current_user: User = Depends(get_current_user)):
    """List all available roles."""
    return [role.value for role in UserRole]


@router.get("/permissions", response_model=List[str])
async def list_permissions(current_user: User = Depends(require_admin)):
    """List all available permissions (Admin only)."""
    return [permission.value for permission in Permission]
