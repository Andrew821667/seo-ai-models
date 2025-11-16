"""
Unit tests for AuthService.
"""

import pytest
from datetime import datetime, timedelta
from jose import jwt

from seo_ai_models.web.api.services.auth_service import AuthService


class TestAuthService:
    """Tests for AuthService."""

    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance."""
        return AuthService()

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "test_password_123"
        hashed = auth_service.get_password_hash(password)

        assert hashed != password
        assert len(hashed) > 20
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self, auth_service):
        """Test password verification with correct password."""
        password = "test_password_123"
        hashed = auth_service.get_password_hash(password)

        assert auth_service.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self, auth_service):
        """Test password verification with incorrect password."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = auth_service.get_password_hash(password)

        assert auth_service.verify_password(wrong_password, hashed) is False

    def test_create_access_token(self, auth_service):
        """Test access token creation."""
        data = {"sub": "user123", "email": "test@example.com"}
        token = auth_service.create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 50

        # Decode and verify
        payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert payload["sub"] == "user123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_create_refresh_token(self, auth_service):
        """Test refresh token creation."""
        data = {"sub": "user123"}
        token = auth_service.create_refresh_token(data)

        assert isinstance(token, str)
        assert len(token) > 50

        # Decode and verify
        payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"
        assert "exp" in payload

    def test_create_token_with_custom_expiration(self, auth_service):
        """Test token creation with custom expiration."""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=5)
        token = auth_service.create_access_token(data, expires_delta)

        payload = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        exp_time = datetime.fromtimestamp(payload["exp"])
        now = datetime.utcnow()

        # Should expire in approximately 5 minutes
        time_diff = (exp_time - now).total_seconds()
        assert 4 * 60 < time_diff < 6 * 60

    def test_verify_token_valid(self, auth_service):
        """Test token verification with valid token."""
        data = {"sub": "user123", "email": "test@example.com"}
        token = auth_service.create_access_token(data)

        payload = auth_service.verify_token(token)

        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["email"] == "test@example.com"

    def test_verify_token_invalid(self, auth_service):
        """Test token verification with invalid token."""
        invalid_token = "invalid.token.here"

        payload = auth_service.verify_token(invalid_token)

        assert payload is None

    def test_verify_token_expired(self, auth_service):
        """Test token verification with expired token."""
        data = {"sub": "user123"}
        # Create token that expires immediately
        expires_delta = timedelta(seconds=-1)
        token = auth_service.create_access_token(data, expires_delta)

        payload = auth_service.verify_token(token)

        assert payload is None

    def test_authenticate_user_correct_credentials(self, auth_service):
        """Test user authentication with correct credentials."""
        password = "test_password"
        hashed = auth_service.get_password_hash(password)

        user_data = {
            "id": "user123",
            "username": "testuser",
            "hashed_password": hashed
        }

        def get_user_func(username):
            return user_data if username == "testuser" else None

        result = auth_service.authenticate_user("testuser", password, get_user_func)

        assert result is not None
        assert result["username"] == "testuser"

    def test_authenticate_user_wrong_password(self, auth_service):
        """Test user authentication with wrong password."""
        password = "test_password"
        wrong_password = "wrong_password"
        hashed = auth_service.get_password_hash(password)

        user_data = {
            "id": "user123",
            "username": "testuser",
            "hashed_password": hashed
        }

        def get_user_func(username):
            return user_data if username == "testuser" else None

        result = auth_service.authenticate_user("testuser", wrong_password, get_user_func)

        assert result is None

    def test_create_tokens_for_user(self, auth_service):
        """Test creating both access and refresh tokens for user."""
        tokens = auth_service.create_tokens_for_user("user123", "testuser")

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

        # Verify both tokens
        access_payload = auth_service.verify_token(tokens["access_token"])
        refresh_payload = auth_service.verify_token(tokens["refresh_token"])

        assert access_payload["sub"] == "user123"
        assert access_payload["username"] == "testuser"
        assert access_payload["type"] == "access"

        assert refresh_payload["sub"] == "user123"
        assert refresh_payload["type"] == "refresh"
