"""
Сервисный слой для работы с проектами и задачами.
Обеспечивает бизнес-логику и взаимодействие с ProjectManagement.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from seo_ai_models.web.dashboard.project_management import ProjectManagement, Project

logger = logging.getLogger(__name__)


class Task:
    """Класс, представляющий задачу проекта."""

    def __init__(
        self,
        task_id: str,
        project_id: str,
        title: str,
        description: str = "",
        status: str = "pending",
        priority: str = "medium",
        assignee_id: Optional[str] = None,
        due_date: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализирует задачу.

        Args:
            task_id: Уникальный идентификатор задачи
            project_id: ID проекта, к которому относится задача
            title: Название задачи
            description: Описание задачи
            status: Статус задачи (pending, in_progress, completed, cancelled)
            priority: Приоритет (low, medium, high, critical)
            assignee_id: ID исполнителя
            due_date: Срок выполнения
            created_at: Время создания
            updated_at: Время последнего обновления
            metadata: Дополнительные метаданные
        """
        self.task_id = task_id
        self.project_id = project_id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.assignee_id = assignee_id
        self.due_date = due_date
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует задачу в словарь."""
        return {
            "id": self.task_id,
            "project_id": self.project_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "assignee_id": self.assignee_id,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


class ProjectService:
    """
    Сервис для управления проектами и задачами.
    Обеспечивает бизнес-логику, валидацию и проверку прав доступа.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Инициализирует сервис проектов.

        Args:
            data_dir: Директория для хранения данных
        """
        self.project_manager = ProjectManagement(data_dir=data_dir)
        self.tasks: Dict[str, Task] = {}
        logger.info("ProjectService initialized")

    def check_project_access(self, project_id: str, user_id: str) -> bool:
        """
        Проверяет права доступа пользователя к проекту.

        Args:
            project_id: ID проекта
            user_id: ID пользователя

        Returns:
            bool: True, если доступ разрешен
        """
        project = self.project_manager.get_project(project_id)
        if not project:
            return False

        # В реальной системе здесь будет проверка ролей и прав
        # Пока проверяем только владельца
        return project.owner_id == user_id or project.owner_id is None

    def delete_project(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """
        Удаляет проект (soft delete).

        Args:
            project_id: ID проекта
            user_id: ID пользователя, выполняющего операцию

        Returns:
            Dict[str, Any]: Результат операции
        """
        # Проверяем существование проекта
        project = self.project_manager.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found for deletion")
            return {"success": False, "error": "Project not found"}

        # Проверяем права доступа
        if not self.check_project_access(project_id, user_id):
            logger.warning(f"User {user_id} denied access to delete project {project_id}")
            return {
                "success": False,
                "error": "Access denied. Only project owner can delete the project.",
            }

        # Удаляем проект (soft delete)
        success = self.project_manager.delete_project(project_id)

        if success:
            # Также помечаем все связанные задачи как удаленные
            tasks_deleted = 0
            for task in self.get_project_tasks(project_id):
                if self._delete_task_internal(task.task_id):
                    tasks_deleted += 1

            logger.info(
                f"Project {project_id} deleted by user {user_id}. {tasks_deleted} tasks marked as deleted."
            )

            return {
                "success": True,
                "project_id": project_id,
                "tasks_deleted": tasks_deleted,
                "message": "Project and associated tasks have been marked as deleted",
            }
        else:
            logger.error(f"Failed to delete project {project_id}")
            return {"success": False, "error": "Failed to delete project"}

    def create_task(
        self,
        project_id: str,
        title: str,
        description: str = "",
        status: str = "pending",
        priority: str = "medium",
        assignee_id: Optional[str] = None,
        due_date: Optional[datetime] = None,
    ) -> Optional[Task]:
        """
        Создает новую задачу для проекта.

        Args:
            project_id: ID проекта
            title: Название задачи
            description: Описание задачи
            status: Статус задачи
            priority: Приоритет задачи
            assignee_id: ID исполнителя
            due_date: Срок выполнения

        Returns:
            Optional[Task]: Созданная задача или None
        """
        # Проверяем существование проекта
        project = self.project_manager.get_project(project_id)
        if not project:
            logger.warning(f"Cannot create task: project {project_id} not found")
            return None

        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            project_id=project_id,
            title=title,
            description=description,
            status=status,
            priority=priority,
            assignee_id=assignee_id,
            due_date=due_date,
        )

        self.tasks[task_id] = task
        logger.info(f"Task {task_id} created for project {project_id}")

        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Получает задачу по ID."""
        return self.tasks.get(task_id)

    def get_project_tasks(
        self, project_id: str, status: Optional[str] = None, priority: Optional[str] = None
    ) -> List[Task]:
        """
        Получает список задач проекта с возможностью фильтрации.

        Args:
            project_id: ID проекта
            status: Фильтр по статусу
            priority: Фильтр по приоритету

        Returns:
            List[Task]: Список задач
        """
        result = []
        for task in self.tasks.values():
            if task.project_id != project_id:
                continue
            if status and task.status != status:
                continue
            if priority and task.priority != priority:
                continue
            result.append(task)

        return result

    def _delete_task_internal(self, task_id: str) -> bool:
        """
        Внутренний метод для удаления задачи.

        Args:
            task_id: ID задачи

        Returns:
            bool: True если успешно
        """
        task = self.get_task(task_id)
        if task:
            task.status = "deleted"
            task.updated_at = datetime.now()
            return True
        return False

    def delete_task(self, project_id: str, task_id: str, user_id: str) -> Dict[str, Any]:
        """
        Удаляет задачу (soft delete).

        Args:
            project_id: ID проекта
            task_id: ID задачи
            user_id: ID пользователя

        Returns:
            Dict[str, Any]: Результат операции
        """
        # Проверяем существование задачи
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found for deletion")
            return {"success": False, "error": "Task not found"}

        # Проверяем что задача принадлежит проекту
        if task.project_id != project_id:
            logger.warning(f"Task {task_id} does not belong to project {project_id}")
            return {"success": False, "error": "Task does not belong to this project"}

        # Проверяем права доступа к проекту
        if not self.check_project_access(project_id, user_id):
            logger.warning(f"User {user_id} denied access to delete task {task_id}")
            return {"success": False, "error": "Access denied"}

        # Удаляем задачу (soft delete)
        success = self._delete_task_internal(task_id)

        if success:
            logger.info(f"Task {task_id} deleted by user {user_id}")
            return {
                "success": True,
                "task_id": task_id,
                "message": "Task has been marked as deleted",
            }
        else:
            logger.error(f"Failed to delete task {task_id}")
            return {"success": False, "error": "Failed to delete task"}
