"""
Project service with database support - Full CRUD implementation.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from ..database.repositories import ProjectRepository, TaskRepository
from ..database.models import ProjectStatus, TaskStatus

logger = logging.getLogger(__name__)


class ProjectServiceDB:
    """
    Service for managing projects and tasks with database backend.
    Full CRUD implementation with proper validation and access control.
    """

    def __init__(self, db: Session):
        self.db = db
        self.project_repo = ProjectRepository(db)
        self.task_repo = TaskRepository(db)
        logger.info("ProjectServiceDB initialized with database backend")

    def check_project_access(self, project_id: str, user_id: str) -> bool:
        """Check if user has access to project."""
        project = self.project_repo.get_by_id(project_id)
        if not project:
            return False
        return project.owner_id == user_id

    # ===== PROJECT CRUD =====

    def create_project(
        self,
        name: str,
        website: str,
        owner_id: str,
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a new project."""
        try:
            project_data = {
                "id": str(uuid.uuid4()),
                "name": name,
                "website": website,
                "description": description,
                "owner_id": owner_id,
                "status": ProjectStatus.ACTIVE,
                "metadata": metadata or {},
            }

            project = self.project_repo.create(project_data)

            logger.info(f"Project created: {project.id} by user {owner_id}")
            return {
                "success": True,
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "website": project.website,
                    "description": project.description,
                    "status": project.status.value,
                    "owner_id": project.owner_id,
                    "created_at": project.created_at.isoformat(),
                    "updated_at": project.updated_at.isoformat(),
                    "metadata": project.extra_metadata,
                },
            }
        except Exception as e:
            logger.error(f"Failed to create project: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_project(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Get project by ID."""
        project = self.project_repo.get_by_id(project_id)

        if not project:
            return {"success": False, "error": "Project not found"}

        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied"}

        tasks_count = self.task_repo.count_by_project(project_id)

        return {
            "success": True,
            "project": {
                "id": project.id,
                "name": project.name,
                "website": project.website,
                "description": project.description,
                "status": project.status.value,
                "owner_id": project.owner_id,
                "created_at": project.created_at.isoformat(),
                "updated_at": project.updated_at.isoformat(),
                "tasks_count": tasks_count,
                "metadata": project.extra_metadata,
            },
        }

    def list_projects(
        self, user_id: str, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        """List user's projects."""
        try:
            projects = self.project_repo.get_by_owner(user_id, status, skip, limit)

            project_list = []
            for project in projects:
                tasks_count = self.task_repo.count_by_project(project.id)
                project_list.append(
                    {
                        "id": project.id,
                        "name": project.name,
                        "website": project.website,
                        "description": project.description,
                        "status": project.status.value,
                        "owner_id": project.owner_id,
                        "created_at": project.created_at.isoformat(),
                        "updated_at": project.updated_at.isoformat(),
                        "tasks_count": tasks_count,
                        "metadata": project.extra_metadata,
                    }
                )

            return {"success": True, "projects": project_list, "total": len(project_list)}
        except Exception as e:
            logger.error(f"Failed to list projects: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_project(
        self, project_id: str, user_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update project."""
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied"}

        try:
            # Remove None values
            cleaned_data = {k: v for k, v in update_data.items() if v is not None}

            project = self.project_repo.update(project_id, cleaned_data)

            if not project:
                return {"success": False, "error": "Project not found"}

            logger.info(f"Project updated: {project_id} by user {user_id}")
            return {
                "success": True,
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "website": project.website,
                    "description": project.description,
                    "status": project.status.value,
                    "owner_id": project.owner_id,
                    "created_at": project.created_at.isoformat(),
                    "updated_at": project.updated_at.isoformat(),
                    "metadata": project.extra_metadata,
                },
            }
        except Exception as e:
            logger.error(f"Failed to update project {project_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def delete_project(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Delete project (soft delete)."""
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied. Only project owner can delete the project."}

        project = self.project_repo.get_by_id(project_id)
        if not project:
            return {"success": False, "error": "Project not found"}

        try:
            # Soft delete the project
            success = self.project_repo.soft_delete(project_id)

            if success:
                # Also soft delete all tasks
                tasks = self.task_repo.get_by_project(project_id)
                tasks_deleted = 0
                for task in tasks:
                    if self.task_repo.soft_delete(task.id):
                        tasks_deleted += 1

                logger.info(
                    f"Project {project_id} soft deleted by user {user_id}. {tasks_deleted} tasks deleted."
                )
                return {
                    "success": True,
                    "project_id": project_id,
                    "tasks_deleted": tasks_deleted,
                    "message": "Project and associated tasks have been marked as deleted",
                }
            else:
                return {"success": False, "error": "Failed to delete project"}
        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    # ===== TASK CRUD =====

    def create_task(
        self,
        project_id: str,
        title: str,
        user_id: str,
        description: str = "",
        status: str = "pending",
        priority: str = "medium",
        assignee_id: Optional[str] = None,
        due_date: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a new task."""
        # Check project access
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied to project"}

        try:
            task_data = {
                "id": str(uuid.uuid4()),
                "project_id": project_id,
                "title": title,
                "description": description,
                "status": TaskStatus[status.upper()],
                "priority": priority,
                "assignee_id": assignee_id,
                "due_date": due_date,
                "metadata": metadata or {},
            }

            task = self.task_repo.create(task_data)

            logger.info(f"Task created: {task.id} in project {project_id}")
            return {
                "success": True,
                "task": {
                    "id": task.id,
                    "project_id": task.project_id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "assignee_id": task.assignee_id,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                    "metadata": task.extra_metadata,
                },
            }
        except Exception as e:
            logger.error(f"Failed to create task: {str(e)}")
            return {"success": False, "error": str(e)}

    def list_tasks(
        self,
        project_id: str,
        user_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List tasks for a project."""
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied"}

        try:
            tasks = self.task_repo.get_by_project(project_id, status, priority, skip, limit)

            task_list = []
            for task in tasks:
                task_list.append(
                    {
                        "id": task.id,
                        "project_id": task.project_id,
                        "title": task.title,
                        "description": task.description,
                        "status": task.status.value,
                        "priority": task.priority.value,
                        "assignee_id": task.assignee_id,
                        "due_date": task.due_date.isoformat() if task.due_date else None,
                        "created_at": task.created_at.isoformat(),
                        "updated_at": task.updated_at.isoformat(),
                        "metadata": task.extra_metadata,
                    }
                )

            return {"success": True, "tasks": task_list, "total": len(task_list)}
        except Exception as e:
            logger.error(f"Failed to list tasks: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_task(self, project_id: str, task_id: str, user_id: str) -> Dict[str, Any]:
        """Get task by ID."""
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied"}

        task = self.task_repo.get_by_id(task_id)

        if not task:
            return {"success": False, "error": "Task not found"}

        if task.project_id != project_id:
            return {"success": False, "error": "Task does not belong to this project"}

        return {
            "success": True,
            "task": {
                "id": task.id,
                "project_id": task.project_id,
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "assignee_id": task.assignee_id,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "metadata": task.extra_metadata,
            },
        }

    def update_task(
        self, project_id: str, task_id: str, user_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update task."""
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied"}

        task = self.task_repo.get_by_id(task_id)
        if not task or task.project_id != project_id:
            return {"success": False, "error": "Task not found"}

        try:
            cleaned_data = {k: v for k, v in update_data.items() if v is not None}
            task = self.task_repo.update(task_id, cleaned_data)

            logger.info(f"Task updated: {task_id} by user {user_id}")
            return {
                "success": True,
                "task": {
                    "id": task.id,
                    "project_id": task.project_id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "assignee_id": task.assignee_id,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                    "metadata": task.extra_metadata,
                },
            }
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def delete_task(self, project_id: str, task_id: str, user_id: str) -> Dict[str, Any]:
        """Delete task (soft delete)."""
        if not self.check_project_access(project_id, user_id):
            return {"success": False, "error": "Access denied"}

        task = self.task_repo.get_by_id(task_id)
        if not task:
            return {"success": False, "error": "Task not found"}

        if task.project_id != project_id:
            return {"success": False, "error": "Task does not belong to this project"}

        try:
            success = self.task_repo.soft_delete(task_id)

            if success:
                logger.info(f"Task {task_id} soft deleted by user {user_id}")
                return {
                    "success": True,
                    "task_id": task_id,
                    "message": "Task has been marked as deleted",
                }
            else:
                return {"success": False, "error": "Failed to delete task"}
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {str(e)}")
            return {"success": False, "error": str(e)}
