"""
Integration tests for Projects API endpoints.
"""

import pytest


class TestProjectsAPI:
    """Integration tests for project endpoints."""

    def test_create_project(self, client, auth_headers):
        """Test creating a project via API."""
        project_data = {
            "name": "Test Project",
            "website": "https://example.com",
            "description": "Test description",
            "status": "active"
        }

        response = client.post(
            "/api/projects/",
            json=project_data,
            headers=auth_headers
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["website"] == "https://example.com"
        assert "id" in data

    def test_list_projects(self, client, auth_headers):
        """Test listing projects."""
        # Create a project first
        project_data = {
            "name": "List Test Project",
            "website": "https://example.com",
            "description": "Test"
        }

        client.post("/api/projects/", json=project_data, headers=auth_headers)

        # List projects
        response = client.get("/api/projects/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_project(self, client, auth_headers):
        """Test getting a specific project."""
        # Create project
        create_response = client.post(
            "/api/projects/",
            json={
                "name": "Get Test",
                "website": "https://test.com",
                "description": "Test"
            },
            headers=auth_headers
        )

        project_id = create_response.json()["id"]

        # Get project
        response = client.get(f"/api/projects/{project_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project_id
        assert data["name"] == "Get Test"

    def test_update_project(self, client, auth_headers):
        """Test updating a project."""
        # Create project
        create_response = client.post(
            "/api/projects/",
            json={
                "name": "Update Test",
                "website": "https://old.com",
                "description": "Old"
            },
            headers=auth_headers
        )

        project_id = create_response.json()["id"]

        # Update project
        update_data = {
            "name": "Updated Project",
            "description": "New description"
        }

        response = client.put(
            f"/api/projects/{project_id}",
            json=update_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Project"
        assert data["description"] == "New description"

    def test_delete_project(self, client, auth_headers):
        """Test deleting a project."""
        # Create project
        create_response = client.post(
            "/api/projects/",
            json={
                "name": "Delete Test",
                "website": "https://delete.com",
                "description": "To be deleted"
            },
            headers=auth_headers
        )

        project_id = create_response.json()["id"]

        # Delete project
        response = client.delete(
            f"/api/projects/{project_id}",
            headers=auth_headers
        )

        assert response.status_code == 204

        # Verify project is deleted
        get_response = client.get(
            f"/api/projects/{project_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

    def test_create_task(self, client, auth_headers):
        """Test creating a task for a project."""
        # Create project first
        project_response = client.post(
            "/api/projects/",
            json={
                "name": "Task Test Project",
                "website": "https://tasks.com",
                "description": "For tasks"
            },
            headers=auth_headers
        )

        project_id = project_response.json()["id"]

        # Create task
        task_data = {
            "title": "Test Task",
            "description": "Task description",
            "status": "pending",
            "priority": "high"
        }

        response = client.post(
            f"/api/projects/{project_id}/tasks",
            json=task_data,
            headers=auth_headers
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert data["title"] == "Test Task"
        assert data["project_id"] == project_id

    def test_list_tasks(self, client, auth_headers):
        """Test listing tasks for a project."""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={
                "name": "Tasks List Project",
                "website": "https://list.com",
                "description": "Tasks"
            },
            headers=auth_headers
        )

        project_id = project_response.json()["id"]

        # Create tasks
        client.post(
            f"/api/projects/{project_id}/tasks",
            json={"title": "Task 1", "description": "T1"},
            headers=auth_headers
        )

        client.post(
            f"/api/projects/{project_id}/tasks",
            json={"title": "Task 2", "description": "T2"},
            headers=auth_headers
        )

        # List tasks
        response = client.get(
            f"/api/projects/{project_id}/tasks",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_unauthorized_access(self, client):
        """Test that endpoints require authentication."""
        # Try to list projects without auth
        response = client.get("/api/projects/")

        assert response.status_code == 401

    def test_create_project_validation(self, client, auth_headers):
        """Test project creation validation."""
        # Missing required fields
        invalid_data = {
            "name": "Test"
            # Missing website
        }

        response = client.post(
            "/api/projects/",
            json=invalid_data,
            headers=auth_headers
        )

        assert response.status_code == 422  # Validation error
