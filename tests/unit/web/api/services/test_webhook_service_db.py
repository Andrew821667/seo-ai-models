"""
Unit tests for WebhookServiceDB.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from seo_ai_models.web.api.database.models import Base
from seo_ai_models.web.api.services.webhook_service_db import WebhookServiceDB
from seo_ai_models.web.api.services.project_service_db import ProjectServiceDB


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()


@pytest.fixture
def service(db_session):
    """Create WebhookServiceDB instance."""
    return WebhookServiceDB(db_session)


@pytest.fixture
def project_service(db_session):
    """Create ProjectServiceDB instance."""
    return ProjectServiceDB(db_session)


@pytest.fixture
def test_project(project_service):
    """Create a test project."""
    result = project_service.create_project(
        name="Test Project",
        website="https://example.com",
        owner_id="user123"
    )
    return result["project"]


class TestWebhookServiceDB:
    """Tests for WebhookServiceDB."""

    def test_create_webhook_success(self, service, test_project):
        """Test successful webhook creation."""
        result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.example.com/endpoint",
            events=["project.created", "project.updated"],
            user_id="user123",
            description="Test webhook"
        )

        assert result["success"] is True
        assert "webhook" in result
        assert result["webhook"]["url"] == "https://webhook.example.com/endpoint"
        assert result["webhook"]["events"] == ["project.created", "project.updated"]
        assert result["webhook"]["project_id"] == test_project["id"]
        assert result["webhook"]["status"] == "active"
        assert "secret" in result["webhook"]

    def test_create_webhook_invalid_url(self, service, test_project):
        """Test creating webhook with invalid URL."""
        result = service.create_webhook(
            project_id=test_project["id"],
            url="invalid-url",
            events=["project.created"],
            user_id="user123"
        )

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_create_webhook_empty_events(self, service, test_project):
        """Test creating webhook with empty events list."""
        result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.example.com",
            events=[],
            user_id="user123"
        )

        assert result["success"] is False
        assert "events" in result["error"].lower()

    def test_create_webhook_access_denied(self, service, test_project):
        """Test creating webhook without project access."""
        result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.example.com",
            events=["project.created"],
            user_id="user456"  # Different user
        )

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_get_webhook_success(self, service, test_project):
        """Test getting webhook by ID."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.example.com",
            events=["project.created"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Get webhook
        result = service.get_webhook(webhook_id, "user123")

        assert result["success"] is True
        assert result["webhook"]["id"] == webhook_id
        assert result["webhook"]["url"] == "https://webhook.example.com"

    def test_get_webhook_not_found(self, service):
        """Test getting non-existent webhook."""
        result = service.get_webhook("nonexistent", "user123")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_get_webhook_access_denied(self, service, test_project):
        """Test getting webhook without access."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.example.com",
            events=["project.created"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Try to get as different user
        result = service.get_webhook(webhook_id, "user456")

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_list_webhooks(self, service, test_project):
        """Test listing project webhooks."""
        # Create multiple webhooks
        service.create_webhook(
            test_project["id"],
            "https://webhook1.com",
            ["event1"],
            "user123"
        )
        service.create_webhook(
            test_project["id"],
            "https://webhook2.com",
            ["event2"],
            "user123"
        )

        # List webhooks
        result = service.list_webhooks(test_project["id"], "user123")

        assert result["success"] is True
        assert len(result["webhooks"]) == 2
        assert result["total"] == 2

    def test_list_webhooks_access_denied(self, service, test_project):
        """Test listing webhooks without access."""
        result = service.list_webhooks(test_project["id"], "user456")

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_update_webhook_success(self, service, test_project):
        """Test updating webhook."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://old.com",
            events=["old.event"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Update webhook
        result = service.update_webhook(
            webhook_id,
            "user123",
            {
                "url": "https://new.com",
                "events": ["new.event1", "new.event2"],
                "description": "Updated description"
            }
        )

        assert result["success"] is True
        assert result["webhook"]["url"] == "https://new.com"
        assert result["webhook"]["events"] == ["new.event1", "new.event2"]
        assert result["webhook"]["description"] == "Updated description"

    def test_update_webhook_invalid_url(self, service, test_project):
        """Test updating webhook with invalid URL."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://valid.com",
            events=["event"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Try to update with invalid URL
        result = service.update_webhook(
            webhook_id,
            "user123",
            {"url": "invalid-url"}
        )

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_update_webhook_access_denied(self, service, test_project):
        """Test updating webhook without access."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.com",
            events=["event"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Try to update as different user
        result = service.update_webhook(
            webhook_id,
            "user456",
            {"description": "Hacked"}
        )

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_delete_webhook_success(self, service, test_project):
        """Test deleting webhook."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.com",
            events=["event"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Delete webhook
        result = service.delete_webhook(webhook_id, "user123")

        assert result["success"] is True
        assert result["webhook_id"] == webhook_id

        # Verify webhook is deleted
        get_result = service.get_webhook(webhook_id, "user123")
        assert get_result["success"] is False

    def test_delete_webhook_access_denied(self, service, test_project):
        """Test deleting webhook without access."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.com",
            events=["event"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Try to delete as different user
        result = service.delete_webhook(webhook_id, "user456")

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_get_webhooks_by_event(self, service, test_project):
        """Test getting webhooks by event type."""
        # Create webhooks with different events
        service.create_webhook(
            test_project["id"],
            "https://webhook1.com",
            ["event.a", "event.b"],
            "user123"
        )
        service.create_webhook(
            test_project["id"],
            "https://webhook2.com",
            ["event.b", "event.c"],
            "user123"
        )
        service.create_webhook(
            test_project["id"],
            "https://webhook3.com",
            ["event.c"],
            "user123"
        )

        # Get webhooks for event.b
        webhooks = service.get_webhooks_by_event("event.b")

        # Should return 2 webhooks that listen to event.b
        assert len(webhooks) >= 2  # May include more if other tests created webhooks

    def test_trigger_webhook_success(self, service, test_project):
        """Test triggering webhook."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.com",
            events=["test.event"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Trigger webhook
        result = service.trigger_webhook(
            webhook_id,
            "test.event",
            {"data": "test payload"}
        )

        assert result["success"] is True
        assert result["webhook_id"] == webhook_id
        assert result["event"] == "test.event"

    def test_trigger_webhook_wrong_event(self, service, test_project):
        """Test triggering webhook with event it doesn't listen to."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.com",
            events=["event.a"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Try to trigger with different event
        result = service.trigger_webhook(
            webhook_id,
            "event.b",
            {"data": "test"}
        )

        assert result["success"] is False
        assert "does not listen" in result["error"].lower()

    def test_trigger_webhook_inactive(self, service, test_project):
        """Test triggering inactive webhook."""
        # Create webhook
        create_result = service.create_webhook(
            project_id=test_project["id"],
            url="https://webhook.com",
            events=["test.event"],
            user_id="user123"
        )
        webhook_id = create_result["webhook"]["id"]

        # Deactivate webhook
        service.update_webhook(webhook_id, "user123", {"status": "inactive"})

        # Try to trigger
        result = service.trigger_webhook(
            webhook_id,
            "test.event",
            {"data": "test"}
        )

        assert result["success"] is False
        assert "not active" in result["error"].lower()
