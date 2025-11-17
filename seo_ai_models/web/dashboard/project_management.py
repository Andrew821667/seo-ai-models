"""
ProjectManagement - Модуль для управления проектами через панель управления.
Обеспечивает функциональность создания, редактирования, удаления и анализа проектов.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class Project:
    """Класс, представляющий проект в системе."""

    def __init__(
        self,
        project_id: str,
        name: str,
        url: str,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        owner_id: Optional[str] = None,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
        status: str = "active",
    ):
        """
        Инициализирует проект.

        Args:
            project_id: Уникальный идентификатор проекта
            name: Название проекта
            url: URL сайта или страницы проекта
            created_at: Время создания проекта
            updated_at: Время последнего обновления проекта
            owner_id: ID владельца проекта
            description: Описание проекта
            settings: Настройки проекта
            status: Статус проекта (active, archived, deleted)
        """
        self.project_id = project_id
        self.name = name
        self.url = url
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.owner_id = owner_id
        self.description = description
        self.settings = settings or {}
        self.status = status
        self.analyses = []

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует проект в словарь."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "url": self.url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner_id": self.owner_id,
            "description": self.description,
            "settings": self.settings,
            "status": self.status,
            "analyses_count": len(self.analyses),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Создает проект из словаря."""
        # Обрабатываем даты, которые приходят в виде строк
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at")
        )
        updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at")
        )

        return cls(
            project_id=data["project_id"],
            name=data["name"],
            url=data["url"],
            created_at=created_at,
            updated_at=updated_at,
            owner_id=data.get("owner_id"),
            description=data.get("description", ""),
            settings=data.get("settings", {}),
            status=data.get("status", "active"),
        )


class Analysis:
    """Класс, представляющий анализ проекта."""

    def __init__(
        self,
        analysis_id: str,
        project_id: str,
        created_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        status: str = "pending",
        settings: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализирует анализ.

        Args:
            analysis_id: Уникальный идентификатор анализа
            project_id: ID проекта, к которому относится анализ
            created_at: Время создания анализа
            completed_at: Время завершения анализа
            status: Статус анализа (pending, running, completed, failed)
            settings: Настройки анализа
            results: Результаты анализа
        """
        self.analysis_id = analysis_id
        self.project_id = project_id
        self.created_at = created_at or datetime.now()
        self.completed_at = completed_at
        self.status = status
        self.settings = settings or {}
        self.results = results or {}

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует анализ в словарь."""
        return {
            "analysis_id": self.analysis_id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "settings": self.settings,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Analysis":
        """Создает анализ из словаря."""
        # Обрабатываем даты, которые приходят в виде строк
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at")
        )
        completed_at = (
            datetime.fromisoformat(data["completed_at"])
            if isinstance(data.get("completed_at"), str) and data.get("completed_at")
            else None
        )

        return cls(
            analysis_id=data["analysis_id"],
            project_id=data["project_id"],
            created_at=created_at,
            completed_at=completed_at,
            status=data.get("status", "pending"),
            settings=data.get("settings", {}),
            results=data.get("results", {}),
        )


class ProjectManagement:
    """
    Класс для управления проектами в панели управления.
    """

    def __init__(self, data_dir: Optional[str] = None, api_client=None):
        """
        Инициализирует управление проектами.

        Args:
            data_dir: Директория для хранения данных проектов (для локального режима)
            api_client: Клиент API для взаимодействия с бэкендом
        """
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"), ".seo_ai_models", "projects"
        )
        self.api_client = api_client
        self.projects = {}
        self.analyses = {}

        # Создаем директорию для данных, если она не существует
        os.makedirs(self.data_dir, exist_ok=True)

        # Загружаем существующие проекты
        self._load_projects()

    def _load_projects(self):
        """Загружает существующие проекты из хранилища."""
        projects_dir = os.path.join(self.data_dir, "projects")
        analyses_dir = os.path.join(self.data_dir, "analyses")

        # Создаем директории, если они не существуют
        os.makedirs(projects_dir, exist_ok=True)
        os.makedirs(analyses_dir, exist_ok=True)

        # Загружаем проекты
        for project_file in Path(projects_dir).glob("*.json"):
            try:
                with open(project_file, "r", encoding="utf-8") as f:
                    project_data = json.load(f)
                    project = Project.from_dict(project_data)
                    self.projects[project.project_id] = project
            except Exception as e:
                logger.error(f"Failed to load project from {project_file}: {str(e)}")

        # Загружаем анализы
        for analysis_file in Path(analyses_dir).glob("*.json"):
            try:
                with open(analysis_file, "r", encoding="utf-8") as f:
                    analysis_data = json.load(f)
                    analysis = Analysis.from_dict(analysis_data)
                    self.analyses[analysis.analysis_id] = analysis

                    # Добавляем анализ к соответствующему проекту
                    if analysis.project_id in self.projects:
                        self.projects[analysis.project_id].analyses.append(analysis.analysis_id)
            except Exception as e:
                logger.error(f"Failed to load analysis from {analysis_file}: {str(e)}")

    def create_project(
        self,
        name: str,
        url: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
        owner_id: Optional[str] = None,
    ) -> Project:
        """
        Создает новый проект.

        Args:
            name: Название проекта
            url: URL сайта или страницы проекта
            description: Описание проекта
            settings: Настройки проекта
            owner_id: ID владельца проекта

        Returns:
            Project: Созданный проект
        """
        # Генерируем уникальный ID для проекта
        import uuid

        project_id = str(uuid.uuid4())

        # Создаем проект
        project = Project(
            project_id=project_id,
            name=name,
            url=url,
            description=description,
            settings=settings,
            owner_id=owner_id,
        )

        # Сохраняем проект
        self.projects[project_id] = project
        self._save_project(project)

        return project

    def _save_project(self, project: Project):
        """
        Сохраняет проект в хранилище.

        Args:
            project: Проект для сохранения
        """
        projects_dir = os.path.join(self.data_dir, "projects")
        os.makedirs(projects_dir, exist_ok=True)

        project_file = os.path.join(projects_dir, f"{project.project_id}.json")

        with open(project_file, "w", encoding="utf-8") as f:
            json.dump(project.to_dict(), f, indent=2, ensure_ascii=False)

    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Получает проект по ID.

        Args:
            project_id: ID проекта

        Returns:
            Optional[Project]: Проект, если найден, иначе None
        """
        return self.projects.get(project_id)

    def get_projects(
        self, owner_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Project]:
        """
        Получает список проектов с возможностью фильтрации.

        Args:
            owner_id: Фильтр по ID владельца
            status: Фильтр по статусу

        Returns:
            List[Project]: Список проектов
        """
        result = []

        for project in self.projects.values():
            if owner_id and project.owner_id != owner_id:
                continue
            if status and project.status != status:
                continue
            result.append(project)

        return result

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        url: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Optional[Project]:
        """
        Обновляет проект.

        Args:
            project_id: ID проекта
            name: Новое название проекта
            url: Новый URL сайта или страницы
            description: Новое описание
            settings: Новые настройки
            status: Новый статус

        Returns:
            Optional[Project]: Обновленный проект, если найден, иначе None
        """
        project = self.get_project(project_id)
        if not project:
            return project

        if name:
            project.name = name
        if url:
            project.url = url
        if description:
            project.description = description
        if settings:
            project.settings.update(settings)
        if status:
            project.status = status

        project.updated_at = datetime.now()

        # Сохраняем проект
        self._save_project(project)

        return project

    def update_project_status(self, project_id: str, status: str) -> Optional[Project]:
        """Обновляет статус проекта."""
        project = self.get_project(project_id)
        if not project:
            print(f"❌ Проект {project_id} не найден")
            return None

        old_status = project.status
        project.status = status
        project.updated_at = datetime.now()

        self._save_project(project)

        print(f'✅ Статус проекта {project_id} изменен с "{old_status}" на "{status}"')
        return project

    def schedule_analysis(
        self,
        project_id: str,
        analysis_name: str,
        analysis_type: str = "full_seo",
        scheduled_time: Optional[datetime] = None,
        priority: str = "normal",
    ) -> Optional[Analysis]:
        """Планирует выполнение анализа проекта."""
        from datetime import timedelta
        import uuid

        project = self.get_project(project_id)
        if not project:
            print(f"❌ Проект {project_id} не найден для планирования анализа")
            return None

        analysis_id = str(uuid.uuid4())

        if not scheduled_time:
            scheduled_time = datetime.now() + timedelta(minutes=5)

        analysis = Analysis(
            analysis_id=analysis_id,
            project_id=project_id,
            name=analysis_name,
            type=analysis_type,
            status="scheduled",
            created_at=datetime.now(),
            settings={"scheduled_time": scheduled_time.isoformat(), "priority": priority},
        )

        self.analyses[analysis_id] = analysis
        project.analyses.append(analysis_id)
        self._save_analysis(analysis)

        print(
            f'✅ Анализ "{analysis_name}" запланирован на {scheduled_time.strftime("%Y-%m-%d %H:%M:%S")}'
        )
        return analysis

    def delete_project(self, project_id: str) -> bool:
        """
        Удаляет проект.

        Args:
            project_id: ID проекта

        Returns:
            bool: True, если проект успешно удален, иначе False
        """
        project = self.get_project(project_id)
        if not project:
            return False

        # Помечаем проект как удаленный
        project.status = "deleted"
        project.updated_at = datetime.now()

        # Сохраняем проект
        self._save_project(project)

        return True

    def create_analysis(
        self, project_id: str, settings: Optional[Dict[str, Any]] = None
    ) -> Optional[Analysis]:
        """
        Создает новый анализ для проекта.

        Args:
            project_id: ID проекта
            settings: Настройки анализа

        Returns:
            Optional[Analysis]: Созданный анализ, если проект найден, иначе None
        """
        project = self.get_project(project_id)
        if not project:
            return None

        # Генерируем уникальный ID для анализа
        import uuid

        analysis_id = str(uuid.uuid4())

        # Создаем анализ
        analysis = Analysis(analysis_id=analysis_id, project_id=project_id, settings=settings)

        # Сохраняем анализ
        self.analyses[analysis_id] = analysis
        project.analyses.append(analysis_id)
        self._save_analysis(analysis)

        return analysis

    def _save_analysis(self, analysis: Analysis):
        """
        Сохраняет анализ в хранилище.

        Args:
            analysis: Анализ для сохранения
        """
        analyses_dir = os.path.join(self.data_dir, "analyses")
        os.makedirs(analyses_dir, exist_ok=True)

        analysis_file = os.path.join(analyses_dir, f"{analysis.analysis_id}.json")

        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)

    def get_analysis(self, analysis_id: str) -> Optional[Analysis]:
        """
        Получает анализ по ID.

        Args:
            analysis_id: ID анализа

        Returns:
            Optional[Analysis]: Анализ, если найден, иначе None
        """
        return self.analyses.get(analysis_id)

    def get_project_analyses(self, project_id: str) -> List[Analysis]:
        """
        Получает список анализов для проекта.

        Args:
            project_id: ID проекта

        Returns:
            List[Analysis]: Список анализов
        """
        project = self.get_project(project_id)
        if not project:
            return []

        return [
            self.analyses.get(analysis_id)
            for analysis_id in project.analyses
            if analysis_id in self.analyses
        ]

    def update_analysis_status(
        self, analysis_id: str, status: str, results: Optional[Dict[str, Any]] = None
    ) -> Optional[Analysis]:
        """
        Обновляет статус анализа.

        Args:
            analysis_id: ID анализа
            status: Новый статус
            results: Результаты анализа (если status == "completed")

        Returns:
            Optional[Analysis]: Обновленный анализ, если найден, иначе None
        """
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            return analysis

        analysis.status = status

        if status == "completed":
            analysis.completed_at = datetime.now()
            if results:
                analysis.results = results

        # Сохраняем анализ
        self._save_analysis(analysis)

        return analysis

    def get_recent_analyses(self, limit: int = 10) -> List[Analysis]:
        """
        Получает список последних анализов.

        Args:
            limit: Максимальное количество анализов

        Returns:
            List[Analysis]: Список анализов
        """
        # Сортируем анализы по дате создания (от новых к старым)
        sorted_analyses = sorted(self.analyses.values(), key=lambda x: x.created_at, reverse=True)

        return sorted_analyses[:limit]

    def get_active_analyses(self) -> List[Analysis]:
        """
        Получает список активных анализов (pending, running).

        Returns:
            List[Analysis]: Список активных анализов
        """
        return [a for a in self.analyses.values() if a.status in ["pending", "running"]]

    def get_completed_analyses(self, limit: int = 100) -> List[Analysis]:
        """
        Получает список завершенных анализов.

        Args:
            limit: Максимальное количество анализов

        Returns:
            List[Analysis]: Список завершенных анализов
        """
        # Сортируем завершенные анализы по дате завершения (от новых к старым)
        sorted_analyses = sorted(
            [a for a in self.analyses.values() if a.status == "completed" and a.completed_at],
            key=lambda x: x.completed_at,
            reverse=True,
        )

        return sorted_analyses[:limit]

    def get_project_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику по проектам.

        Returns:
            Dict[str, Any]: Статистика по проектам
        """
        active_projects = len([p for p in self.projects.values() if p.status == "active"])
        active_analyses = len(self.get_active_analyses())
        completed_analyses = len([a for a in self.analyses.values() if a.status == "completed"])
        failed_analyses = len([a for a in self.analyses.values() if a.status == "failed"])

        return {
            "total_projects": len(self.projects),
            "active_projects": active_projects,
            "total_analyses": len(self.analyses),
            "active_analyses": active_analyses,
            "completed_analyses": completed_analyses,
            "failed_analyses": failed_analyses,
        }
