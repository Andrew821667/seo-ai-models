"""
Task repository for database operations.
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from ..models import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskRepository:
    """Repository for Task operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, task_data: dict) -> Task:
        """Create a new task."""
        task = Task(**task_data)
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        logger.info(f"Created task: {task.title} (ID: {task.id})")
        return task

    def get_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.db.query(Task).filter(Task.id == task_id).first()

    def get_by_project(
        self,
        project_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Task]:
        """Get tasks by project with optional filters."""
        query = self.db.query(Task).filter(Task.project_id == project_id)

        if status:
            query = query.filter(Task.status == status)

        if priority:
            query = query.filter(Task.priority == priority)

        return query.offset(skip).limit(limit).all()

    def get_by_assignee(
        self, assignee_id: str, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Task]:
        """Get tasks by assignee."""
        query = self.db.query(Task).filter(Task.assignee_id == assignee_id)

        if status:
            query = query.filter(Task.status == status)

        return query.offset(skip).limit(limit).all()

    def update(self, task_id: str, task_data: dict) -> Optional[Task]:
        """Update task."""
        task = self.get_by_id(task_id)
        if not task:
            return None

        for key, value in task_data.items():
            if hasattr(task, key) and value is not None:
                setattr(task, key, value)

        self.db.commit()
        self.db.refresh(task)
        logger.info(f"Updated task: {task.title} (ID: {task.id})")
        return task

    def soft_delete(self, task_id: str) -> bool:
        """Soft delete task (mark as deleted)."""
        task = self.get_by_id(task_id)
        if not task:
            return False

        task.status = TaskStatus.DELETED
        self.db.commit()
        logger.info(f"Soft deleted task: {task.title} (ID: {task.id})")
        return True

    def delete(self, task_id: str) -> bool:
        """Hard delete task."""
        task = self.get_by_id(task_id)
        if not task:
            return False

        self.db.delete(task)
        self.db.commit()
        logger.info(f"Deleted task: {task.title} (ID: {task.id})")
        return True

    def count_by_project(self, project_id: str, status: Optional[str] = None) -> int:
        """Count tasks by project."""
        query = self.db.query(Task).filter(Task.project_id == project_id)

        if status:
            query = query.filter(Task.status == status)

        return query.count()
