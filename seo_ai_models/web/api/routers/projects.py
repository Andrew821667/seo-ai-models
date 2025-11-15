
"""
Роутер для управления проектами и задачами.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
import logging

from ..models.projects import (
    ProjectCreate, ProjectResponse, ProjectUpdate,
    TaskCreate, TaskResponse, TaskUpdate
)
from .auth import oauth2_scheme
from ..services.project_service import ProjectService

# Создаем объект роутера
router = APIRouter()

# Логгер
logger = logging.getLogger(__name__)

# Глобальный экземпляр сервиса (в production будет через Dependency Injection)
_project_service = None


def get_project_service() -> ProjectService:
    """Получает экземпляр ProjectService (Dependency)."""
    global _project_service
    if _project_service is None:
        _project_service = ProjectService()
    return _project_service


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    """
    Извлекает ID текущего пользователя из токена.

    В реальной системе здесь будет декодирование JWT токена.
    Пока возвращаем тестовый ID.
    """
    # TODO: Implement actual JWT token decoding
    return "user123"


# Маршрут для получения списка проектов
@router.get("/", response_model=List[ProjectResponse])
async def get_projects(
    status: Optional[str] = None,
    search: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    token: str = Depends(oauth2_scheme)
):
    """
    Получение списка проектов с возможностью фильтрации и пагинации.
    
    Args:
        status: Фильтр по статусу проекта.
        search: Поисковый запрос.
        skip: Количество пропускаемых записей (для пагинации).
        limit: Максимальное количество возвращаемых записей.
        token: Токен доступа.
        
    Returns:
        List[ProjectResponse]: Список проектов.
    """
    # Здесь будет код для получения проектов
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return [
        {
            "id": "project123",
            "name": "Test Project",
            "description": "Test project description",
            "website": "https://example.com",
            "status": "active",
            "owner_id": "user123",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "tasks_count": 5,
            "metadata": {}
        }
    ]


# Маршрут для создания нового проекта
@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project_data: ProjectCreate, token: str = Depends(oauth2_scheme)):
    """
    Создание нового проекта.
    
    Args:
        project_data: Данные для создания проекта.
        token: Токен доступа.
        
    Returns:
        ProjectResponse: Данные созданного проекта.
    """
    # Здесь будет код для создания проекта
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return {
        "id": "project123",
        "name": project_data.name,
        "description": project_data.description,
        "website": project_data.website,
        "status": project_data.status,
        "owner_id": "user123",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "tasks_count": 0,
        "metadata": project_data.metadata or {}
    }


# Маршрут для получения информации о проекте
@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, token: str = Depends(oauth2_scheme)):
    """
    Получение информации о проекте по ID.
    
    Args:
        project_id: ID проекта.
        token: Токен доступа.
        
    Returns:
        ProjectResponse: Данные проекта.
    """
    # Здесь будет код для получения проекта по ID
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return {
        "id": project_id,
        "name": "Test Project",
        "description": "Test project description",
        "website": "https://example.com",
        "status": "active",
        "owner_id": "user123",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "tasks_count": 5,
        "metadata": {}
    }


# Маршрут для обновления проекта
@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    token: str = Depends(oauth2_scheme)
):
    """
    Обновление проекта по ID.
    
    Args:
        project_id: ID проекта.
        project_data: Данные для обновления проекта.
        token: Токен доступа.
        
    Returns:
        ProjectResponse: Обновленные данные проекта.
    """
    # Здесь будет код для обновления проекта
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return {
        "id": project_id,
        "name": project_data.name or "Test Project",
        "description": project_data.description or "Test project description",
        "website": project_data.website or "https://example.com",
        "status": project_data.status or "active",
        "owner_id": "user123",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "tasks_count": 5,
        "metadata": project_data.metadata or {}
    }


# Маршрут для удаления проекта
@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    user_id: str = Depends(get_current_user_id),
    service: ProjectService = Depends(get_project_service)
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
    # Возвращаем 204 No Content (FastAPI автоматически обработает при успехе)


# Маршруты для задач

# Маршрут для получения списка задач проекта
@router.get("/{project_id}/tasks", response_model=List[TaskResponse])
async def get_project_tasks(
    project_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    search: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    token: str = Depends(oauth2_scheme)
):
    """
    Получение списка задач проекта с возможностью фильтрации и пагинации.
    
    Args:
        project_id: ID проекта.
        status: Фильтр по статусу задачи.
        priority: Фильтр по приоритету задачи.
        search: Поисковый запрос.
        skip: Количество пропускаемых записей (для пагинации).
        limit: Максимальное количество возвращаемых записей.
        token: Токен доступа.
        
    Returns:
        List[TaskResponse]: Список задач проекта.
    """
    # Здесь будет код для получения задач проекта
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return [
        {
            "id": "task123",
            "title": "Test Task",
            "project_id": project_id,
            "description": "Test task description",
            "status": "in_progress",
            "priority": "high",
            "assignee_id": "user123",
            "due_date": datetime.now(),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": {}
        }
    ]


# Маршрут для создания задачи
@router.post("/{project_id}/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    project_id: str,
    task_data: TaskCreate,
    token: str = Depends(oauth2_scheme)
):
    """
    Создание новой задачи в проекте.
    
    Args:
        project_id: ID проекта.
        task_data: Данные для создания задачи.
        token: Токен доступа.
        
    Returns:
        TaskResponse: Данные созданной задачи.
    """
    # Здесь будет код для создания задачи
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return {
        "id": "task123",
        "title": task_data.title,
        "project_id": project_id,
        "description": task_data.description,
        "status": task_data.status,
        "priority": task_data.priority,
        "assignee_id": task_data.assignee_id,
        "due_date": task_data.due_date,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "metadata": task_data.metadata or {}
    }


# Маршрут для получения информации о задаче
@router.get("/{project_id}/tasks/{task_id}", response_model=TaskResponse)
async def get_task(project_id: str, task_id: str, token: str = Depends(oauth2_scheme)):
    """
    Получение информации о задаче по ID.
    
    Args:
        project_id: ID проекта.
        task_id: ID задачи.
        token: Токен доступа.
        
    Returns:
        TaskResponse: Данные задачи.
    """
    # Здесь будет код для получения задачи по ID
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return {
        "id": task_id,
        "title": "Test Task",
        "project_id": project_id,
        "description": "Test task description",
        "status": "in_progress",
        "priority": "high",
        "assignee_id": "user123",
        "due_date": datetime.now(),
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "metadata": {}
    }


# Маршрут для обновления задачи
@router.put("/{project_id}/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    project_id: str,
    task_id: str,
    task_data: TaskUpdate,
    token: str = Depends(oauth2_scheme)
):
    """
    Обновление задачи по ID.
    
    Args:
        project_id: ID проекта.
        task_id: ID задачи.
        task_data: Данные для обновления задачи.
        token: Токен доступа.
        
    Returns:
        TaskResponse: Обновленные данные задачи.
    """
    # Здесь будет код для обновления задачи
    # с использованием ProjectManager
    
    # Пока просто заглушка
    return {
        "id": task_id,
        "title": task_data.title or "Test Task",
        "project_id": project_id,
        "description": task_data.description or "Test task description",
        "status": task_data.status or "in_progress",
        "priority": task_data.priority or "high",
        "assignee_id": task_data.assignee_id or "user123",
        "due_date": task_data.due_date or datetime.now(),
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "metadata": task_data.metadata or {}
    }


# Маршрут для удаления задачи
@router.delete("/{project_id}/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    project_id: str,
    task_id: str,
    user_id: str = Depends(get_current_user_id),
    service: ProjectService = Depends(get_project_service)
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
    # Возвращаем 204 No Content (FastAPI автоматически обработает при успехе)
