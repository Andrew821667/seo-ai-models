
"""
ProjectManagement - Модуль для управления проектами SEO-оптимизации.
Предоставляет функциональность для создания, редактирования и отслеживания
проектов и заданий по SEO-оптимизации.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime
from uuid import uuid4, UUID
from enum import Enum
from pathlib import Path


class ProjectStatus(Enum):
    """Статусы проекта."""
    DRAFT = "draft"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class TaskPriority(Enum):
    """Приоритеты задач."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(Enum):
    """Статусы задач."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Project:
    """Класс проекта SEO-оптимизации."""
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 website: str = "",
                 status: ProjectStatus = ProjectStatus.DRAFT,
                 owner_id: Optional[Union[str, UUID]] = None,
                 created_at: Optional[datetime] = None):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.website = website
        self.status = status
        self.owner_id = str(owner_id) if owner_id else None
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at
        self.tasks = []
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует проект в словарь."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "website": self.website,
            "status": self.status.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tasks_count": len(self.tasks),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Создает проект из словаря."""
        project = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            website=data.get("website", ""),
            status=ProjectStatus(data.get("status", "draft")),
            owner_id=data.get("owner_id")
        )
        
        project.id = data.get("id", str(uuid4()))
        project.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        project.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        project.metadata = data.get("metadata", {})
        
        return project


class Task:
    """Класс задачи SEO-оптимизации."""
    
    def __init__(self,
                 title: str,
                 project_id: str,
                 description: str = "",
                 status: TaskStatus = TaskStatus.PENDING,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 assignee_id: Optional[str] = None,
                 due_date: Optional[datetime] = None,
                 created_at: Optional[datetime] = None):
        self.id = str(uuid4())
        self.title = title
        self.project_id = project_id
        self.description = description
        self.status = status
        self.priority = priority
        self.assignee_id = assignee_id
        self.due_date = due_date
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует задачу в словарь."""
        return {
            "id": self.id,
            "title": self.title,
            "project_id": self.project_id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "assignee_id": self.assignee_id,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Создает задачу из словаря."""
        task = cls(
            title=data.get("title", ""),
            project_id=data.get("project_id", ""),
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", "medium")),
            assignee_id=data.get("assignee_id"),
            due_date=datetime.fromisoformat(data.get("due_date")) if data.get("due_date") else None
        )
        
        task.id = data.get("id", str(uuid4()))
        task.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        task.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        task.metadata = data.get("metadata", {})
        
        return task


class ProjectManager:
    """Менеджер проектов для управления проектами и задачами."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("./data/projects")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects: Dict[str, Project] = {}
        self.tasks: Dict[str, Task] = {}
        
    def load_projects(self):
        """Загружает проекты из файлов."""
        projects_dir = self.data_dir / "projects"
        if not projects_dir.exists():
            projects_dir.mkdir(parents=True, exist_ok=True)
            return
            
        for project_file in projects_dir.glob("*.json"):
            try:
                with open(project_file, 'r') as f:
                    project_data = json.load(f)
                project = Project.from_dict(project_data)
                self.projects[project.id] = project
            except Exception as e:
                logging.error(f"Failed to load project from {project_file}: {str(e)}")
                
    def load_tasks(self):
        """Загружает задачи из файлов."""
        tasks_dir = self.data_dir / "tasks"
        if not tasks_dir.exists():
            tasks_dir.mkdir(parents=True, exist_ok=True)
            return
            
        for task_file in tasks_dir.glob("*.json"):
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                task = Task.from_dict(task_data)
                self.tasks[task.id] = task
                
                # Добавляем задачу в соответствующий проект
                if task.project_id in self.projects:
                    self.projects[task.project_id].tasks.append(task.id)
            except Exception as e:
                logging.error(f"Failed to load task from {task_file}: {str(e)}")
                
    def create_project(self, name: str, **kwargs) -> Project:
        """Создает новый проект."""
        project = Project(name=name, **kwargs)
        self.projects[project.id] = project
        self._save_project(project)
        return project
        
    def update_project(self, project_id: str, **kwargs) -> Optional[Project]:
        """Обновляет существующий проект."""
        if project_id not in self.projects:
            return None
            
        project = self.projects[project_id]
        for key, value in kwargs.items():
            if hasattr(project, key):
                setattr(project, key, value)
                
        project.updated_at = datetime.now()
        self._save_project(project)
        return project
        
    def _save_project(self, project: Project):
        """Сохраняет проект в файл."""
        projects_dir = self.data_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = projects_dir / f"{project.id}.json"
        with open(file_path, 'w') as f:
            json.dump(project.to_dict(), f, indent=2)
            
    def create_task(self, title: str, project_id: str, **kwargs) -> Optional[Task]:
        """Создает новую задачу."""
        if project_id not in self.projects:
            return None
            
        task = Task(title=title, project_id=project_id, **kwargs)
        self.tasks[task.id] = task
        self.projects[project_id].tasks.append(task.id)
        self._save_task(task)
        return task
        
    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        """Обновляет существующую задачу."""
        if task_id not in self.tasks:
            return None
            
        task = self.tasks[task_id]
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
                
        task.updated_at = datetime.now()
        self._save_task(task)
        return task
        
    def _save_task(self, task: Task):
        """Сохраняет задачу в файл."""
        tasks_dir = self.data_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = tasks_dir / f"{task.id}.json"
        with open(file_path, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)
            
    def get_project(self, project_id: str) -> Optional[Project]:
        """Возвращает проект по ID."""
        return self.projects.get(project_id)
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Возвращает задачу по ID."""
        return self.tasks.get(task_id)
        
    def get_project_tasks(self, project_id: str) -> List[Task]:
        """Возвращает список задач проекта."""
        if project_id not in self.projects:
            return []
            
        project = self.projects[project_id]
        return [self.tasks.get(task_id) for task_id in project.tasks if task_id in self.tasks]
        
    def delete_project(self, project_id: str) -> bool:
        """Удаляет проект и все его задачи."""
        if project_id not in self.projects:
            return False
            
        # Удаляем задачи проекта
        project = self.projects[project_id]
        for task_id in project.tasks:
            if task_id in self.tasks:
                del self.tasks[task_id]
                task_file = self.data_dir / "tasks" / f"{task_id}.json"
                if task_file.exists():
                    task_file.unlink()
                    
        # Удаляем проект
        del self.projects[project_id]
        project_file = self.data_dir / "projects" / f"{project_id}.json"
        if project_file.exists():
            project_file.unlink()
            
        return True
        
    def delete_task(self, task_id: str) -> bool:
        """Удаляет задачу."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        if task.project_id in self.projects:
            project = self.projects[task.project_id]
            if task_id in project.tasks:
                project.tasks.remove(task_id)
                self._save_project(project)
                
        del self.tasks[task_id]
        task_file = self.data_dir / "tasks" / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink()
            
        return True


# Функция для создания экземпляра ProjectManager
def create_project_manager(data_dir: Optional[str] = None) -> ProjectManager:
    """Создает экземпляр менеджера проектов."""
    manager = ProjectManager(data_dir)
    manager.load_projects()
    manager.load_tasks()
    return manager
