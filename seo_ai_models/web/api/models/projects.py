"""
Модели данных для проектов и задач.
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ProjectStatus(str, Enum):
    """Статусы проекта."""

    DRAFT = "draft"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class TaskPriority(str, Enum):
    """Приоритеты задач."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Статусы задач."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ProjectCreate(BaseModel):
    """Модель для создания проекта."""

    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    website: Optional[HttpUrl] = None
    status: ProjectStatus = ProjectStatus.DRAFT
    metadata: Optional[Dict[str, Any]] = None


class ProjectResponse(BaseModel):
    """Модель ответа с данными проекта."""

    id: str
    name: str
    description: Optional[str] = None
    website: Optional[HttpUrl] = None
    status: ProjectStatus
    owner_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    tasks_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    """Модель для обновления проекта."""

    name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = None
    website: Optional[HttpUrl] = None
    status: Optional[ProjectStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskCreate(BaseModel):
    """Модель для создания задачи."""

    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assignee_id: Optional[str] = None
    due_date: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    """Модель ответа с данными задачи."""

    id: str
    title: str
    project_id: str
    description: Optional[str] = None
    status: TaskStatus
    priority: TaskPriority
    assignee_id: Optional[str] = None
    due_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class TaskUpdate(BaseModel):
    """Модель для обновления задачи."""

    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee_id: Optional[str] = None
    due_date: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
