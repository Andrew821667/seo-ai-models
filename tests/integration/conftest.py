"""
Pytest configuration for integration tests.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from seo_ai_models.web.api.app import create_app
from seo_ai_models.web.api.database.models import Base
from seo_ai_models.web.api.dependencies import get_db


# Create test database
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database override."""
    app = create_app()

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    """Get authentication headers with valid token."""
    # Create a test user and get token
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "Test User"
    }

    # Register user
    response = client.post("/api/auth/register", json=user_data)

    if response.status_code == 201:
        # Login to get token
        login_response = client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "testpass123"}
        )

        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}

    # Fallback if registration fails (user might exist)
    login_response = client.post(
        "/api/auth/login",
        data={"username": "testuser", "password": "testpass123"}
    )

    if login_response.status_code == 200:
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    return {}
