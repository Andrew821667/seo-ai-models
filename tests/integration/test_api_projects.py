"""
Интеграционные тесты для API проектов.
"""

import pytest
from fastapi.testclient import TestClient
from seo_ai_models.web.api.app import app  # type: ignore

# Создаем тестовый клиент
client = TestClient(app)


class TestProjectsAPI:
    """Тесты для API проектов."""

    def test_delete_project_not_found(self):
        """Тест удаления несуществующего проекта."""
        response = client.delete(
            "/projects/nonexistent-id",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_project_unauthorized(self):
        """Тест удаления проекта без токена."""
        response = client.delete("/projects/some-id")
        assert response.status_code in [401, 403]

    def test_delete_task_not_found(self):
        """Тест удаления несуществующей задачи."""
        response = client.delete(
            "/projects/some-project/tasks/nonexistent-task",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 404

    def test_delete_task_unauthorized(self):
        """Тест удаления задачи без токена."""
        response = client.delete("/projects/some-project/tasks/some-task")
        assert response.status_code in [401, 403]


@pytest.mark.asyncio
class TestProjectsLifecycle:
    """Тесты жизненного цикла проекта."""

    @pytest.mark.skip(reason="Requires full API implementation")
    async def test_create_and_delete_project(self):
        """Тест создания и удаления проекта."""
        # Создаем проект
        create_response = client.post(
            "/projects/",
            json={
                "name": "Test Project",
                "website": "https://example.com",
                "description": "Test",
                "status": "active"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        assert create_response.status_code == 201
        project_id = create_response.json()["id"]

        # Удаляем проект
        delete_response = client.delete(
            f"/projects/{project_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        assert delete_response.status_code == 204

    @pytest.mark.skip(reason="Requires full API implementation")
    async def test_create_and_delete_task(self):
        """Тест создания и удаления задачи."""
        # Предполагаем, что проект уже существует
        project_id = "test-project-id"

        # Создаем задачу
        create_response = client.post(
            f"/projects/{project_id}/tasks",
            json={
                "title": "Test Task",
                "description": "Test",
                "status": "pending",
                "priority": "medium"
            },
            headers={"Authorization": "Bearer test_token"}
        )

        if create_response.status_code == 201:
            task_id = create_response.json()["id"]

            # Удаляем задачу
            delete_response = client.delete(
                f"/projects/{project_id}/tasks/{task_id}",
                headers={"Authorization": "Bearer test_token"}
            )
            assert delete_response.status_code == 204
