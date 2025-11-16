"""
Project repository for database operations.
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from ..models import Project, ProjectStatus

logger = logging.getLogger(__name__)


class ProjectRepository:
    """Repository for Project operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, project_data: dict) -> Project:
        """Create a new project."""
        project = Project(**project_data)
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        logger.info(f"Created project: {project.name} (ID: {project.id})")
        return project

    def get_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        return self.db.query(Project).filter(Project.id == project_id).first()

    def get_by_owner(
        self, owner_id: str, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Project]:
        """Get projects by owner with optional status filter."""
        query = self.db.query(Project).filter(Project.owner_id == owner_id)

        if status:
            query = query.filter(Project.status == status)

        return query.offset(skip).limit(limit).all()

    def get_all(
        self, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Project]:
        """Get all projects with pagination and optional status filter."""
        query = self.db.query(Project)

        if status:
            query = query.filter(Project.status == status)

        return query.offset(skip).limit(limit).all()

    def update(self, project_id: str, project_data: dict) -> Optional[Project]:
        """Update project."""
        project = self.get_by_id(project_id)
        if not project:
            return None

        for key, value in project_data.items():
            if hasattr(project, key) and value is not None:
                setattr(project, key, value)

        self.db.commit()
        self.db.refresh(project)
        logger.info(f"Updated project: {project.name} (ID: {project.id})")
        return project

    def soft_delete(self, project_id: str) -> bool:
        """Soft delete project (mark as deleted)."""
        project = self.get_by_id(project_id)
        if not project:
            return False

        project.status = ProjectStatus.DELETED
        self.db.commit()
        logger.info(f"Soft deleted project: {project.name} (ID: {project.id})")
        return True

    def delete(self, project_id: str) -> bool:
        """Hard delete project."""
        project = self.get_by_id(project_id)
        if not project:
            return False

        self.db.delete(project)
        self.db.commit()
        logger.info(f"Deleted project: {project.name} (ID: {project.id})")
        return True

    def count_by_owner(self, owner_id: str, status: Optional[str] = None) -> int:
        """Count projects by owner."""
        query = self.db.query(Project).filter(Project.owner_id == owner_id)

        if status:
            query = query.filter(Project.status == status)

        return query.count()
