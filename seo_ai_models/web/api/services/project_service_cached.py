"""
Cached version of ProjectServiceDB with Redis caching.
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from .project_service_db import ProjectServiceDB
from .cache_service import get_cache_service, cache_result, invalidate_cache

logger = logging.getLogger(__name__)


class ProjectServiceCached(ProjectServiceDB):
    """
    ProjectService with Redis caching layer.
    Extends ProjectServiceDB with caching for read operations.
    """

    def __init__(self, db: Session):
        super().__init__(db)
        self.cache = get_cache_service()
        logger.info("ProjectServiceCached initialized with Redis")

    def _make_cache_key(self, entity: str, *args) -> str:
        """Generate cache key."""
        return f"project:{entity}:{':'.join(str(a) for a in args)}"

    def get_project(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Get project with caching."""
        cache_key = self._make_cache_key("get", project_id, user_id)

        # Try cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Get from database
        result = super().get_project(project_id, user_id)

        # Cache if successful
        if result.get("success"):
            self.cache.set(cache_key, result, ttl=300)  # 5 minutes

        return result

    def list_projects(
        self,
        owner_id: str,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List projects with caching."""
        cache_key = self._make_cache_key("list", owner_id, status or "all", skip, limit)

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = super().list_projects(owner_id, status, skip, limit)

        if result.get("success"):
            self.cache.set(cache_key, result, ttl=60)  # 1 minute for lists

        return result

    def create_project(self, name: str, website: str, owner_id: str, **kwargs) -> Dict[str, Any]:
        """Create project and invalidate cache."""
        result = super().create_project(name, website, owner_id, **kwargs)

        if result.get("success"):
            # Invalidate user's project list cache
            invalidate_cache(f"project:list:{owner_id}:*")

        return result

    def update_project(
        self, project_id: str, user_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update project and invalidate cache."""
        result = super().update_project(project_id, user_id, update_data)

        if result.get("success"):
            # Invalidate project cache
            invalidate_cache(f"project:get:{project_id}:*")
            invalidate_cache(f"project:list:{user_id}:*")

        return result

    def delete_project(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Delete project and invalidate cache."""
        result = super().delete_project(project_id, user_id)

        if result.get("success"):
            # Invalidate all related caches
            invalidate_cache(f"project:get:{project_id}:*")
            invalidate_cache(f"project:list:{user_id}:*")
            invalidate_cache(f"project:tasks:{project_id}:*")

        return result

    def get_task(self, project_id: str, task_id: str, user_id: str) -> Dict[str, Any]:
        """Get task with caching."""
        cache_key = self._make_cache_key("task", project_id, task_id, user_id)

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = super().get_task(project_id, task_id, user_id)

        if result.get("success"):
            self.cache.set(cache_key, result, ttl=300)

        return result

    def list_tasks(
        self,
        project_id: str,
        user_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List tasks with caching."""
        cache_key = self._make_cache_key(
            "tasks", project_id, user_id, status or "all", priority or "all", skip, limit
        )

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = super().list_tasks(project_id, user_id, status, priority, skip, limit)

        if result.get("success"):
            self.cache.set(cache_key, result, ttl=60)

        return result

    def create_task(self, project_id: str, title: str, user_id: str, **kwargs) -> Dict[str, Any]:
        """Create task and invalidate cache."""
        result = super().create_task(project_id, title, user_id, **kwargs)

        if result.get("success"):
            # Invalidate project and task list caches
            invalidate_cache(f"project:get:{project_id}:*")
            invalidate_cache(f"project:tasks:{project_id}:*")

        return result

    def update_task(
        self,
        project_id: str,
        task_id: str,
        user_id: str,
        update_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update task and invalidate cache."""
        result = super().update_task(project_id, task_id, user_id, update_data)

        if result.get("success"):
            # Invalidate task caches
            invalidate_cache(f"project:task:{project_id}:{task_id}:*")
            invalidate_cache(f"project:tasks:{project_id}:*")
            invalidate_cache(f"project:get:{project_id}:*")

        return result

    def delete_task(self, project_id: str, task_id: str, user_id: str) -> Dict[str, Any]:
        """Delete task and invalidate cache."""
        result = super().delete_task(project_id, task_id, user_id)

        if result.get("success"):
            # Invalidate all related caches
            invalidate_cache(f"project:task:{project_id}:{task_id}:*")
            invalidate_cache(f"project:tasks:{project_id}:*")
            invalidate_cache(f"project:get:{project_id}:*")

        return result
