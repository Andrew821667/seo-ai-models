"""
Интеграционные тесты для API webhooks.
"""

import pytest
from fastapi.testclient import TestClient
from seo_ai_models.web.api.app import app  # type: ignore

client = TestClient(app)


class TestWebhooksAPI:
    """Тесты для API webhooks."""

    def test_delete_webhook_not_found(self):
        """Тест удаления несуществующего webhook."""
        response = client.delete(
            "/webhooks/nonexistent-webhook-id",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_webhook_unauthorized(self):
        """Тест удаления webhook без токена."""
        response = client.delete("/webhooks/some-id")
        assert response.status_code in [401, 403]
