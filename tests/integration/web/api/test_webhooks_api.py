"""
Integration tests for Webhooks API endpoints.
"""

import pytest


class TestWebhooksAPI:
    """Integration tests for webhook endpoints."""

    @pytest.fixture
    def test_project(self, client, auth_headers):
        """Create a test project for webhook tests."""
        response = client.post(
            "/api/projects/",
            json={
                "name": "Webhook Test Project",
                "website": "https://webhook-test.com",
                "description": "For testing webhooks"
            },
            headers=auth_headers
        )
        return response.json()

    def test_create_webhook(self, client, auth_headers, test_project):
        """Test creating a webhook."""
        webhook_data = {
            "url": "https://webhook.example.com/endpoint",
            "events": ["project.created", "project.updated"],
            "project_id": test_project["id"],
            "description": "Test webhook"
        }

        response = client.post(
            "/api/webhooks/",
            json=webhook_data,
            headers=auth_headers
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert data["url"] == "https://webhook.example.com/endpoint"
        assert data["events"] == ["project.created", "project.updated"]
        assert "id" in data

    def test_list_webhooks(self, client, auth_headers, test_project):
        """Test listing webhooks for a project."""
        # Create webhook first
        client.post(
            "/api/webhooks/",
            json={
                "url": "https://list-test.com/hook",
                "events": ["task.created"],
                "project_id": test_project["id"]
            },
            headers=auth_headers
        )

        # List webhooks
        response = client.get(
            f"/api/webhooks/?project_id={test_project['id']}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_webhook(self, client, auth_headers, test_project):
        """Test getting a specific webhook."""
        # Create webhook
        create_response = client.post(
            "/api/webhooks/",
            json={
                "url": "https://get-test.com/hook",
                "events": ["project.deleted"],
                "project_id": test_project["id"],
                "description": "Get test webhook"
            },
            headers=auth_headers
        )

        webhook_id = create_response.json()["id"]

        # Get webhook
        response = client.get(
            f"/api/webhooks/{webhook_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == webhook_id
        assert data["url"] == "https://get-test.com/hook"

    def test_update_webhook(self, client, auth_headers, test_project):
        """Test updating a webhook."""
        # Create webhook
        create_response = client.post(
            "/api/webhooks/",
            json={
                "url": "https://update-test.com/old",
                "events": ["old.event"],
                "project_id": test_project["id"]
            },
            headers=auth_headers
        )

        webhook_id = create_response.json()["id"]

        # Update webhook
        update_data = {
            "url": "https://update-test.com/new",
            "events": ["new.event"],
            "description": "Updated webhook"
        }

        response = client.put(
            f"/api/webhooks/{webhook_id}",
            json=update_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://update-test.com/new"
        assert data["events"] == ["new.event"]

    def test_delete_webhook(self, client, auth_headers, test_project):
        """Test deleting a webhook."""
        # Create webhook
        create_response = client.post(
            "/api/webhooks/",
            json={
                "url": "https://delete-test.com/hook",
                "events": ["test.event"],
                "project_id": test_project["id"]
            },
            headers=auth_headers
        )

        webhook_id = create_response.json()["id"]

        # Delete webhook
        response = client.delete(
            f"/api/webhooks/{webhook_id}",
            headers=auth_headers
        )

        assert response.status_code == 204

        # Verify webhook is deleted
        get_response = client.get(
            f"/api/webhooks/{webhook_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

    def test_test_webhook(self, client, auth_headers, test_project):
        """Test triggering a webhook test."""
        # Create webhook
        create_response = client.post(
            "/api/webhooks/",
            json={
                "url": "https://test.com/hook",
                "events": ["test.event"],
                "project_id": test_project["id"]
            },
            headers=auth_headers
        )

        webhook_id = create_response.json()["id"]

        # Test webhook
        response = client.post(
            f"/api/webhooks/{webhook_id}/test",
            json={
                "event": "test.event",
                "payload": {"data": "test"}
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_invalid_webhook_url(self, client, auth_headers, test_project):
        """Test webhook creation with invalid URL."""
        invalid_data = {
            "url": "not-a-valid-url",
            "events": ["test.event"],
            "project_id": test_project["id"]
        }

        response = client.post(
            "/api/webhooks/",
            json=invalid_data,
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_empty_events_list(self, client, auth_headers, test_project):
        """Test webhook creation with empty events."""
        invalid_data = {
            "url": "https://valid.com/hook",
            "events": [],
            "project_id": test_project["id"]
        }

        response = client.post(
            "/api/webhooks/",
            json=invalid_data,
            headers=auth_headers
        )

        assert response.status_code == 400
