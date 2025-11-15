"""
Роутер для управления проектами и задачами.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
import logging

from ..models.projects import (
    ProjectCreate, ProjectResponse, ProjectUpdate,
    TaskCreate, TaskResponse, TaskUpdate
)
from ..services.project_service_db import ProjectServiceDB
from ..dependencies import get_project_service, get_current_user_id

# Создаем объект роутера
router = APIRouter()

# Логгер
logger = logging.getLogger(__name__)


# ===== PROJECT ENDPOINTS =====

@router.get("/", response_model=List[ProjectResponse])
async def get_projects(
    status: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Получение списка проектов пользователя с фильтрацией и пагинацией.

    Args:
        status: Фильтр по статусу проекта.
        skip: Количество пропускаемых записей (для пагинации).
        limit: Максимальное количество возвращаемых записей.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        List[ProjectResponse]: Список проектов.
    """
    result = service.list_projects(user_id, status, skip, limit)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Failed to retrieve projects")
        )

    return result["projects"]


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Создание нового проекта.

    Args:
        project_data: Данные для создания проекта.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        ProjectResponse: Данные созданного проекта.
    """
    result = service.create_project(
        name=project_data.name,
        website=project_data.website,
        owner_id=user_id,
        description=project_data.description or "",
        metadata=project_data.metadata
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to create project")
        )

    return result["project"]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Получение информации о проекте по ID.

    Args:
        project_id: ID проекта.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        ProjectResponse: Данные проекта.
    """
    result = service.get_project(project_id, user_id)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        elif "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )

    return result["project"]


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Обновление проекта по ID.

    Args:
        project_id: ID проекта.
        project_data: Данные для обновления проекта.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        ProjectResponse: Обновленные данные проекта.
    """
    update_data = {}
    if project_data.name is not None:
        update_data["name"] = project_data.name
    if project_data.description is not None:
        update_data["description"] = project_data.description
    if project_data.website is not None:
        update_data["website"] = project_data.website
    if project_data.status is not None:
        update_data["status"] = project_data.status
    if project_data.metadata is not None:
        update_data["metadata"] = project_data.metadata

    result = service.update_project(project_id, user_id, update_data)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        elif "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )

    return result["project"]


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Удаление проекта по ID (soft delete).

    Проект помечается как удаленный, но физически не удаляется из хранилища.
    Все связанные задачи также помечаются как удаленные.

    Args:
        project_id: ID проекта.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Raises:
        HTTPException: 404 Not Found - проект не найден.
        HTTPException: 403 Forbidden - недостаточно прав для удаления.

    Returns:
        Статус 204 No Content при успешном удалении.
    """
    logger.info(f"User {user_id} attempting to delete project {project_id}")

    result = service.delete_project(project_id, user_id)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            logger.warning(f"Project {project_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        elif "access denied" in error.lower():
            logger.warning(f"User {user_id} access denied for project {project_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only project owner can delete the project."
            )
        else:
            logger.error(f"Error deleting project {project_id}: {error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete project: {error}"
            )

    logger.info(f"Project {project_id} successfully deleted. {result.get('tasks_deleted', 0)} tasks deleted.")


# ===== TASK ENDPOINTS =====

@router.get("/{project_id}/tasks", response_model=List[TaskResponse])
async def get_project_tasks(
    project_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Получение списка задач проекта с фильтрацией и пагинацией.

    Args:
        project_id: ID проекта.
        status: Фильтр по статусу задачи.
        priority: Фильтр по приоритету задачи.
        skip: Количество пропускаемых записей (для пагинации).
        limit: Максимальное количество возвращаемых записей.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        List[TaskResponse]: Список задач проекта.
    """
    result = service.list_tasks(project_id, user_id, status, priority, skip, limit)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )

    return result["tasks"]


@router.post("/{project_id}/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    project_id: str,
    task_data: TaskCreate,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Создание новой задачи в проекте.

    Args:
        project_id: ID проекта.
        task_data: Данные для создания задачи.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        TaskResponse: Данные созданной задачи.
    """
    result = service.create_task(
        project_id=project_id,
        title=task_data.title,
        user_id=user_id,
        description=task_data.description or "",
        status=task_data.status or "pending",
        priority=task_data.priority or "medium",
        assignee_id=task_data.assignee_id,
        due_date=task_data.due_date,
        metadata=task_data.metadata
    )

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to project"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )

    return result["task"]


@router.get("/{project_id}/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    project_id: str,
    task_id: str,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Получение информации о задаче по ID.

    Args:
        project_id: ID проекта.
        task_id: ID задачи.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        TaskResponse: Данные задачи.
    """
    result = service.get_task(project_id, task_id, user_id)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        elif "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        elif "does not belong" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task {task_id} does not belong to project {project_id}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )

    return result["task"]


@router.put("/{project_id}/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    project_id: str,
    task_id: str,
    task_data: TaskUpdate,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Обновление задачи по ID.

    Args:
        project_id: ID проекта.
        task_id: ID задачи.
        task_data: Данные для обновления задачи.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Returns:
        TaskResponse: Обновленные данные задачи.
    """
    update_data = {}
    if task_data.title is not None:
        update_data["title"] = task_data.title
    if task_data.description is not None:
        update_data["description"] = task_data.description
    if task_data.status is not None:
        update_data["status"] = task_data.status
    if task_data.priority is not None:
        update_data["priority"] = task_data.priority
    if task_data.assignee_id is not None:
        update_data["assignee_id"] = task_data.assignee_id
    if task_data.due_date is not None:
        update_data["due_date"] = task_data.due_date
    if task_data.metadata is not None:
        update_data["metadata"] = task_data.metadata

    result = service.update_task(project_id, task_id, user_id, update_data)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        elif "access denied" in error.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )

    return result["task"]


@router.delete("/{project_id}/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    project_id: str,
    task_id: str,
    user_id: str = Depends(get_current_user_id),
    service: ProjectServiceDB = Depends(get_project_service)
):
    """
    Удаление задачи по ID (soft delete).

    Задача помечается как удаленная, но физически не удаляется из хранилища.

    Args:
        project_id: ID проекта.
        task_id: ID задачи.
        user_id: ID текущего пользователя (из токена).
        service: Сервис проектов (dependency injection).

    Raises:
        HTTPException: 404 Not Found - задача не найдена.
        HTTPException: 403 Forbidden - недостаточно прав для удаления.
        HTTPException: 400 Bad Request - задача не принадлежит проекту.

    Returns:
        Статус 204 No Content при успешном удалении.
    """
    logger.info(f"User {user_id} attempting to delete task {task_id} from project {project_id}")

    result = service.delete_task(project_id, task_id, user_id)

    if not result["success"]:
        error = result.get("error", "Unknown error")

        if "not found" in error.lower():
            logger.warning(f"Task {task_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        elif "access denied" in error.lower():
            logger.warning(f"User {user_id} access denied for task {task_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You don't have permission to delete this task."
            )
        elif "does not belong" in error.lower():
            logger.warning(f"Task {task_id} does not belong to project {project_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task {task_id} does not belong to project {project_id}"
            )
        else:
            logger.error(f"Error deleting task {task_id}: {error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete task: {error}"
            )

    logger.info(f"Task {task_id} successfully deleted from project {project_id}")
