"""
Unit tests for ProjectServiceDB.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from seo_ai_models.web.api.database.models import Base, ProjectStatus, TaskStatus
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
    """Create ProjectServiceDB instance."""
    return ProjectServiceDB(db_session)


class TestProjectServiceDB:
    """Tests for ProjectServiceDB."""

    def test_create_project_success(self, service):
        """Test successful project creation."""
        result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123",
            description="Test description"
        )

        assert result["success"] is True
        assert "project" in result
        assert result["project"]["name"] == "Test Project"
        assert result["project"]["website"] == "https://example.com"
        assert result["project"]["owner_id"] == "user123"
        assert result["project"]["status"] == "active"
        assert "id" in result["project"]

    def test_get_project_success(self, service):
        """Test getting project by ID."""
        # Create project
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Get project
        result = service.get_project(project_id, "user123")

        assert result["success"] is True
        assert result["project"]["id"] == project_id
        assert result["project"]["name"] == "Test Project"
        assert result["project"]["tasks_count"] == 0

    def test_get_project_not_found(self, service):
        """Test getting non-existent project."""
        result = service.get_project("nonexistent", "user123")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_get_project_access_denied(self, service):
        """Test getting project without access."""
        # Create project for user123
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Try to access as different user
        result = service.get_project(project_id, "user456")

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_list_projects(self, service):
        """Test listing user's projects."""
        # Create multiple projects
        service.create_project("Project 1", "https://example1.com", "user123")
        service.create_project("Project 2", "https://example2.com", "user123")
        service.create_project("Project 3", "https://example3.com", "user456")

        # List projects for user123
        result = service.list_projects("user123")

        assert result["success"] is True
        assert len(result["projects"]) == 2
        assert result["total"] == 2

    def test_update_project_success(self, service):
        """Test updating project."""
        # Create project
        create_result = service.create_project(
            name="Old Name",
            website="https://old.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Update project
        result = service.update_project(
            project_id,
            "user123",
            {"name": "New Name", "description": "New description"}
        )

        assert result["success"] is True
        assert result["project"]["name"] == "New Name"
        assert result["project"]["description"] == "New description"
        assert result["project"]["website"] == "https://old.com"  # Unchanged

    def test_update_project_access_denied(self, service):
        """Test updating project without access."""
        # Create project
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Try to update as different user
        result = service.update_project(
            project_id,
            "user456",
            {"name": "Hacked Name"}
        )

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_delete_project_success(self, service):
        """Test deleting project."""
        # Create project
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Delete project
        result = service.delete_project(project_id, "user123")

        assert result["success"] is True
        assert result["project_id"] == project_id
        assert "tasks_deleted" in result

        # Verify project is marked as deleted
        get_result = service.get_project(project_id, "user123")
        assert get_result["success"] is False

    def test_delete_project_with_tasks(self, service):
        """Test deleting project cascades to tasks."""
        # Create project
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Create tasks
        service.create_task(project_id, "Task 1", "user123")
        service.create_task(project_id, "Task 2", "user123")

        # Delete project
        result = service.delete_project(project_id, "user123")

        assert result["success"] is True
        assert result["tasks_deleted"] == 2

    def test_create_task_success(self, service):
        """Test creating task."""
        # Create project
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Create task
        result = service.create_task(
            project_id=project_id,
            title="Test Task",
            user_id="user123",
            description="Test description",
            priority="high"
        )

        assert result["success"] is True
        assert result["task"]["title"] == "Test Task"
        assert result["task"]["project_id"] == project_id
        assert result["task"]["status"] == "pending"
        assert result["task"]["priority"] == "high"

    def test_create_task_access_denied(self, service):
        """Test creating task without project access."""
        # Create project
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Try to create task as different user
        result = service.create_task(
            project_id=project_id,
            title="Unauthorized Task",
            user_id="user456"
        )

        assert result["success"] is False
        assert "access denied" in result["error"].lower()

    def test_list_tasks(self, service):
        """Test listing project tasks."""
        # Create project
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        # Create tasks
        service.create_task(project_id, "Task 1", "user123", priority="high")
        service.create_task(project_id, "Task 2", "user123", priority="low")
        service.create_task(project_id, "Task 3", "user123", status="completed")

        # List all tasks
        result = service.list_tasks(project_id, "user123")

        assert result["success"] is True
        assert len(result["tasks"]) == 3

        # Filter by priority
        result = service.list_tasks(project_id, "user123", priority="high")
        assert len(result["tasks"]) == 1

    def test_get_task_success(self, service):
        """Test getting task by ID."""
        # Create project and task
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        task_result = service.create_task(project_id, "Test Task", "user123")
        task_id = task_result["task"]["id"]

        # Get task
        result = service.get_task(project_id, task_id, "user123")

        assert result["success"] is True
        assert result["task"]["id"] == task_id
        assert result["task"]["title"] == "Test Task"

    def test_update_task_success(self, service):
        """Test updating task."""
        # Create project and task
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        task_result = service.create_task(project_id, "Old Title", "user123")
        task_id = task_result["task"]["id"]

        # Update task
        result = service.update_task(
            project_id,
            task_id,
            "user123",
            {"title": "New Title", "status": "in_progress"}
        )

        assert result["success"] is True
        assert result["task"]["title"] == "New Title"
        assert result["task"]["status"] == "in_progress"

    def test_delete_task_success(self, service):
        """Test deleting task."""
        # Create project and task
        create_result = service.create_project(
            name="Test Project",
            website="https://example.com",
            owner_id="user123"
        )
        project_id = create_result["project"]["id"]

        task_result = service.create_task(project_id, "Test Task", "user123")
        task_id = task_result["task"]["id"]

        # Delete task
        result = service.delete_task(project_id, task_id, "user123")

        assert result["success"] is True
        assert result["task_id"] == task_id

        # Verify task is deleted
        get_result = service.get_task(project_id, task_id, "user123")
        assert get_result["success"] is False
