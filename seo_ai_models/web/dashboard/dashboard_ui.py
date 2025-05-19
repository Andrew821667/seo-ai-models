
"""
DashboardUI - Основной модуль для панели управления SEO AI Models.
Этот модуль предоставляет интерфейс для визуализации и управления 
всеми аспектами SEO-оптимизации через веб-интерфейс.
"""

from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from pathlib import Path
import os
import traceback

from .project_management import ProjectManagement
from .report_generator import ReportGenerator
from .user_management import UserManagement

logger = logging.getLogger(__name__)

class DashboardConfig:
    """Конфигурация для панели управления."""
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000",
                 theme: str = "light",
                 refresh_interval: int = 60,
                 default_views: List[str] = None,
                 data_dir: Optional[str] = None):
        """
        Инициализирует конфигурацию панели управления.
        
        Args:
            api_url: URL API сервера
            theme: Тема оформления (light, dark)
            refresh_interval: Интервал обновления данных (секунды)
            default_views: Список представлений по умолчанию
            data_dir: Директория для хранения данных
        """
        self.api_url = api_url
        self.theme = theme
        self.refresh_interval = refresh_interval
        self.default_views = default_views or ["overview", "projects", "reports", "users"]
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".seo_ai_models", "dashboard")
        self.user_settings = {}
        
    def save_to_file(self, file_path: str) -> bool:
        """Сохраняет конфигурацию в файл."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Failed to save dashboard config: {str(e)}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'DashboardConfig':
        """Загружает конфигурацию из файла."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            config = cls()
            for key, value in config_data.items():
                setattr(config, key, value)
            return config
        except Exception as e:
            logging.error(f"Failed to load dashboard config: {str(e)}")
            return cls()


class DashboardWidget:
    """Базовый класс для виджетов панели управления."""
    
    def __init__(self, widget_id: str, title: str, type: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует виджет.
        
        Args:
            widget_id: Уникальный идентификатор виджета
            title: Заголовок виджета
            type: Тип виджета (chart, table, summary, etc.)
            config: Конфигурация виджета
        """
        self.widget_id = widget_id
        self.title = title
        self.type = type
        self.config = config or {}
        self.data = {}
        
    def update_data(self, data: Dict[str, Any]):
        """
        Обновляет данные виджета.
        
        Args:
            data: Новые данные
        """
        self.data = data
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует виджет в словарь."""
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "type": self.type,
            "config": self.config,
            "data": self.data
        }


class DashboardView:
    """Класс, представляющий представление панели управления."""
    
    def __init__(self, view_id: str, title: str, icon: str = "", 
                 description: str = "", order: int = 0):
        """
        Инициализирует представление.
        
        Args:
            view_id: Уникальный идентификатор представления
            title: Заголовок представления
            icon: Иконка представления
            description: Описание представления
            order: Порядок отображения
        """
        self.view_id = view_id
        self.title = title
        self.icon = icon
        self.description = description
        self.order = order
        self.widgets = []
        
    def add_widget(self, widget: DashboardWidget, position: Optional[int] = None):
        """
        Добавляет виджет в представление.
        
        Args:
            widget: Виджет
            position: Позиция виджета (если None, то в конец)
        """
        if position is None:
            self.widgets.append(widget)
        else:
            self.widgets.insert(position, widget)
        
    def remove_widget(self, widget_id: str) -> bool:
        """
        Удаляет виджет из представления.
        
        Args:
            widget_id: ID виджета
            
        Returns:
            bool: True, если виджет успешно удален, иначе False
        """
        for i, widget in enumerate(self.widgets):
            if widget.widget_id == widget_id:
                del self.widgets[i]
                return True
        return False
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует представление в словарь."""
        return {
            "view_id": self.view_id,
            "title": self.title,
            "icon": self.icon,
            "description": self.description,
            "order": self.order,
            "widgets": [widget.to_dict() for widget in self.widgets]
        }


class DashboardUI:
    """
    Основной класс для панели управления, который координирует
    визуализацию данных и взаимодействие с другими модулями.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Инициализирует панель управления.
        
        Args:
            config: Конфигурация панели управления
        """
        self.config = config or DashboardConfig()
        self.views = {}
        self.active_view = "overview"
        self.last_refresh = datetime.now()
        self.data_cache = {}
        
        # Компоненты
        self.project_management = None
        self.report_generator = None
        self.user_management = None
        
        # Инициализация логгера
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def initialize(self) -> bool:
        """
        Инициализирует панель управления.
        
        Returns:
            bool: True, если инициализация выполнена успешно, иначе False
        """
        try:
            logger.info("Initializing Dashboard UI")
            
            # Создаем директорию для данных, если она не существует
            os.makedirs(self.config.data_dir, exist_ok=True)
            
            # Загружаем компоненты
            self._load_components()
            
            # Настраиваем представления и виджеты
            self._setup_views()
            
            # Обновляем данные
            self.refresh_data()
            
            logger.info("Dashboard UI initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Dashboard UI: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def _load_components(self):
        """Загружает компоненты для панели управления."""
        try:
            # Создаем экземпляры компонентов
            self.project_management = ProjectManagement(
                data_dir=os.path.join(self.config.data_dir, "projects")
            )
            
            self.report_generator = ReportGenerator(
                data_dir=os.path.join(self.config.data_dir, "reports"),
                project_management=self.project_management
            )
            
            self.user_management = UserManagement(
                data_dir=os.path.join(self.config.data_dir, "users")
            )
            
            logger.info("Components loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load components: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def _setup_views(self):
        """Настраивает представления и виджеты для панели управления."""
        try:
            # Создаем основные представления
            self._setup_overview_view()
            self._setup_projects_view()
            self._setup_reports_view()
            self._setup_users_view()
            self._setup_settings_view()
            
            logger.info("Views and widgets set up successfully")
        except Exception as e:
            logger.error(f"Failed to set up views and widgets: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def _setup_overview_view(self):
        """Настраивает представление обзора."""
        # Создаем представление
        overview_view = DashboardView(
            view_id="overview",
            title="Обзор",
            icon="dashboard",
            description="Общий обзор состояния проектов и анализов",
            order=0
        )
        
        # Добавляем виджеты
        
        # Виджет с общей статистикой
        overview_view.add_widget(DashboardWidget(
            widget_id="overview_stats",
            title="Общая статистика",
            type="summary",
            config={
                "columns": 3,
                "items": [
                    {"label": "Всего проектов", "value_key": "total_projects", "icon": "folder"},
                    {"label": "Активных проектов", "value_key": "active_projects", "icon": "folder-open"},
                    {"label": "Всего анализов", "value_key": "total_analyses", "icon": "analytics"},
                    {"label": "Активных анализов", "value_key": "active_analyses", "icon": "trending-up"},
                    {"label": "Завершенных анализов", "value_key": "completed_analyses", "icon": "check-circle"},
                    {"label": "Всего отчетов", "value_key": "total_reports", "icon": "file-text"}
                ]
            }
        ))
        
        # Виджет с последними анализами
        overview_view.add_widget(DashboardWidget(
            widget_id="recent_analyses",
            title="Последние анализы",
            type="table",
            config={
                "columns": [
                    {"field": "analysis_id", "header": "ID", "visible": False},
                    {"field": "project_name", "header": "Проект"},
                    {"field": "created_at", "header": "Дата создания", "type": "datetime"},
                    {"field": "status", "header": "Статус", "type": "status"},
                    {"field": "actions", "header": "Действия", "type": "actions"}
                ],
                "actions": [
                    {"id": "view", "label": "Просмотр", "icon": "eye"},
                    {"id": "report", "label": "Создать отчет", "icon": "file-text"}
                ]
            }
        ))
        
        # Виджет с последними отчетами
        overview_view.add_widget(DashboardWidget(
            widget_id="recent_reports",
            title="Последние отчеты",
            type="table",
            config={
                "columns": [
                    {"field": "report_id", "header": "ID", "visible": False},
                    {"field": "name", "header": "Название"},
                    {"field": "project_name", "header": "Проект"},
                    {"field": "updated_at", "header": "Дата обновления", "type": "datetime"},
                    {"field": "status", "header": "Статус", "type": "status"},
                    {"field": "actions", "header": "Действия", "type": "actions"}
                ],
                "actions": [
                    {"id": "view", "label": "Просмотр", "icon": "eye"},
                    {"id": "edit", "label": "Редактировать", "icon": "edit"},
                    {"id": "delete", "label": "Удалить", "icon": "trash"}
                ]
            }
        ))
        
        # Виджет с графиком активности
        overview_view.add_widget(DashboardWidget(
            widget_id="activity_chart",
            title="Активность за последние 30 дней",
            type="chart",
            config={
                "chart_type": "line",
                "x_axis": {"field": "date", "label": "Дата"},
                "y_axis": {"field": "count", "label": "Количество"},
                "series": [
                    {"field": "projects", "label": "Проекты", "color": "#4CAF50"},
                    {"field": "analyses", "label": "Анализы", "color": "#2196F3"},
                    {"field": "reports", "label": "Отчеты", "color": "#FFC107"}
                ]
            }
        ))
        
        # Добавляем представление
        self.views["overview"] = overview_view
        
    def _setup_projects_view(self):
        """Настраивает представление проектов."""
        # Создаем представление
        projects_view = DashboardView(
            view_id="projects",
            title="Проекты",
            icon="folder",
            description="Управление проектами и их анализами",
            order=1
        )
        
        # Добавляем виджеты
        
        # Виджет со списком проектов
        projects_view.add_widget(DashboardWidget(
            widget_id="projects_list",
            title="Список проектов",
            type="table",
            config={
                "columns": [
                    {"field": "project_id", "header": "ID", "visible": False},
                    {"field": "name", "header": "Название"},
                    {"field": "url", "header": "URL"},
                    {"field": "analyses_count", "header": "Анализы", "type": "count"},
                    {"field": "status", "header": "Статус", "type": "status"},
                    {"field": "updated_at", "header": "Обновлено", "type": "datetime"},
                    {"field": "actions", "header": "Действия", "type": "actions"}
                ],
                "actions": [
                    {"id": "view", "label": "Просмотр", "icon": "eye"},
                    {"id": "edit", "label": "Редактировать", "icon": "edit"},
                    {"id": "analyze", "label": "Анализировать", "icon": "search"},
                    {"id": "delete", "label": "Удалить", "icon": "trash"}
                ],
                "filters": [
                    {"field": "status", "label": "Статус", "type": "select", "options": [
                        {"value": "active", "label": "Активные"},
                        {"value": "archived", "label": "Архивные"},
                        {"value": "deleted", "label": "Удаленные"}
                    ]},
                    {"field": "name", "label": "Поиск по названию", "type": "text"}
                ],
                "pagination": True,
                "sorting": True
            }
        ))
        
        # Добавляем представление
        self.views["projects"] = projects_view
        
    def _setup_reports_view(self):
        """Настраивает представление отчетов."""
        # Создаем представление
        reports_view = DashboardView(
            view_id="reports",
            title="Отчеты",
            icon="file-text",
            description="Управление отчетами и шаблонами",
            order=2
        )
        
        # Добавляем виджеты
        
        # Виджет со списком отчетов
        reports_view.add_widget(DashboardWidget(
            widget_id="reports_list",
            title="Список отчетов",
            type="table",
            config={
                "columns": [
                    {"field": "report_id", "header": "ID", "visible": False},
                    {"field": "name", "header": "Название"},
                    {"field": "project_name", "header": "Проект"},
                    {"field": "type", "header": "Тип", "type": "tag"},
                    {"field": "status", "header": "Статус", "type": "status"},
                    {"field": "updated_at", "header": "Обновлено", "type": "datetime"},
                    {"field": "actions", "header": "Действия", "type": "actions"}
                ],
                "actions": [
                    {"id": "view", "label": "Просмотр", "icon": "eye"},
                    {"id": "edit", "label": "Редактировать", "icon": "edit"},
                    {"id": "delete", "label": "Удалить", "icon": "trash"}
                ],
                "filters": [
                    {"field": "status", "label": "Статус", "type": "select", "options": [
                        {"value": "draft", "label": "Черновики"},
                        {"value": "published", "label": "Опубликованные"},
                        {"value": "archived", "label": "Архивные"}
                    ]},
                    {"field": "type", "label": "Тип", "type": "select", "options": [
                        {"value": "analysis", "label": "Анализ"},
                        {"value": "comparison", "label": "Сравнение"},
                        {"value": "llm_optimization", "label": "LLM-оптимизация"}
                    ]},
                    {"field": "name", "label": "Поиск по названию", "type": "text"}
                ],
                "pagination": True,
                "sorting": True
            }
        ))
        
        # Виджет со списком шаблонов
        reports_view.add_widget(DashboardWidget(
            widget_id="templates_list",
            title="Шаблоны отчетов",
            type="table",
            config={
                "columns": [
                    {"field": "template_id", "header": "ID", "visible": False},
                    {"field": "name", "header": "Название"},
                    {"field": "type", "header": "Тип", "type": "tag"},
                    {"field": "description", "header": "Описание"},
                    {"field": "actions", "header": "Действия", "type": "actions"}
                ],
                "actions": [
                    {"id": "view", "label": "Просмотр", "icon": "eye"},
                    {"id": "edit", "label": "Редактировать", "icon": "edit"},
                    {"id": "create", "label": "Создать отчет", "icon": "file-plus"},
                    {"id": "delete", "label": "Удалить", "icon": "trash"}
                ],
                "filters": [
                    {"field": "type", "label": "Тип", "type": "select", "options": [
                        {"value": "analysis", "label": "Анализ"},
                        {"value": "comparison", "label": "Сравнение"},
                        {"value": "llm_optimization", "label": "LLM-оптимизация"}
                    ]},
                    {"field": "name", "label": "Поиск по названию", "type": "text"}
                ],
                "pagination": True,
                "sorting": True
            }
        ))
        
        # Добавляем представление
        self.views["reports"] = reports_view
        
    def _setup_users_view(self):
        """Настраивает представление пользователей."""
        # Создаем представление
        users_view = DashboardView(
            view_id="users",
            title="Пользователи",
            icon="users",
            description="Управление пользователями и их ролями",
            order=3
        )
        
        # Добавляем виджеты
        
        # Виджет со списком пользователей
        users_view.add_widget(DashboardWidget(
            widget_id="users_list",
            title="Список пользователей",
            type="table",
            config={
                "columns": [
                    {"field": "user_id", "header": "ID", "visible": False},
                    {"field": "username", "header": "Имя пользователя"},
                    {"field": "full_name", "header": "Полное имя"},
                    {"field": "email", "header": "Email"},
                    {"field": "role", "header": "Роль", "type": "tag"},
                    {"field": "status", "header": "Статус", "type": "status"},
                    {"field": "last_login", "header": "Последний вход", "type": "datetime"},
                    {"field": "actions", "header": "Действия", "type": "actions"}
                ],
                "actions": [
                    {"id": "view", "label": "Просмотр", "icon": "eye"},
                    {"id": "edit", "label": "Редактировать", "icon": "edit"},
                    {"id": "reset_password", "label": "Сбросить пароль", "icon": "key"},
                    {"id": "delete", "label": "Удалить", "icon": "trash"}
                ],
                "filters": [
                    {"field": "status", "label": "Статус", "type": "select", "options": [
                        {"value": "active", "label": "Активные"},
                        {"value": "inactive", "label": "Неактивные"},
                        {"value": "blocked", "label": "Заблокированные"}
                    ]},
                    {"field": "role", "label": "Роль", "type": "select", "options": [
                        {"value": "admin", "label": "Администраторы"},
                        {"value": "manager", "label": "Менеджеры"},
                        {"value": "user", "label": "Пользователи"}
                    ]},
                    {"field": "username", "label": "Поиск", "type": "text"}
                ],
                "pagination": True,
                "sorting": True
            }
        ))
        
        # Виджет с сессиями
        users_view.add_widget(DashboardWidget(
            widget_id="active_sessions",
            title="Активные сессии",
            type="table",
            config={
                "columns": [
                    {"field": "session_id", "header": "ID", "visible": False},
                    {"field": "username", "header": "Пользователь"},
                    {"field": "ip_address", "header": "IP-адрес"},
                    {"field": "user_agent", "header": "User-Agent"},
                    {"field": "created_at", "header": "Начало сессии", "type": "datetime"},
                    {"field": "expires_at", "header": "Истекает", "type": "datetime"},
                    {"field": "actions", "header": "Действия", "type": "actions"}
                ],
                "actions": [
                    {"id": "terminate", "label": "Завершить", "icon": "x-circle"}
                ],
                "pagination": True,
                "sorting": True
            }
        ))
        
        # Добавляем представление
        self.views["users"] = users_view
        
    def _setup_settings_view(self):
        """Настраивает представление настроек."""
        # Создаем представление
        settings_view = DashboardView(
            view_id="settings",
            title="Настройки",
            icon="settings",
            description="Настройки панели управления",
            order=4
        )
        
        # Добавляем виджеты
        
        # Виджет с настройками панели управления
        settings_view.add_widget(DashboardWidget(
            widget_id="dashboard_settings",
            title="Настройки панели управления",
            type="form",
            config={
                "fields": [
                    {"name": "api_url", "label": "URL API", "type": "text"},
                    {"name": "theme", "label": "Тема оформления", "type": "select", "options": [
                        {"value": "light", "label": "Светлая"},
                        {"value": "dark", "label": "Темная"}
                    ]},
                    {"name": "refresh_interval", "label": "Интервал обновления (секунды)", "type": "number"},
                    {"name": "default_views", "label": "Представления по умолчанию", "type": "multi_select", "options": [
                        {"value": "overview", "label": "Обзор"},
                        {"value": "projects", "label": "Проекты"},
                        {"value": "reports", "label": "Отчеты"},
                        {"value": "users", "label": "Пользователи"},
                        {"value": "settings", "label": "Настройки"}
                    ]}
                ],
                "submit_label": "Сохранить"
            }
        ))
        
        # Виджет с настройками пользователя
        settings_view.add_widget(DashboardWidget(
            widget_id="user_profile",
            title="Профиль пользователя",
            type="form",
            config={
                "fields": [
                    {"name": "username", "label": "Имя пользователя", "type": "text", "readonly": True},
                    {"name": "email", "label": "Email", "type": "email"},
                    {"name": "first_name", "label": "Имя", "type": "text"},
                    {"name": "last_name", "label": "Фамилия", "type": "text"},
                    {"name": "password", "label": "Новый пароль", "type": "password"},
                    {"name": "password_confirm", "label": "Подтверждение пароля", "type": "password"}
                ],
                "submit_label": "Сохранить"
            }
        ))
        
        # Добавляем представление
        self.views["settings"] = settings_view
        
    def refresh_data(self) -> bool:
        """
        Обновляет данные в панели управления.
        
        Returns:
            bool: True, если данные успешно обновлены, иначе False
        """
        try:
            logger.info("Refreshing dashboard data")
            self.last_refresh = datetime.now()
            
            # Обновляем данные для каждого представления
            self._refresh_overview_data()
            self._refresh_projects_data()
            self._refresh_reports_data()
            self._refresh_users_data()
            
            logger.info("Dashboard data refreshed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh dashboard data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def _refresh_overview_data(self):
        """Обновляет данные для представления обзора."""
        view = self.views.get("overview")
        if not view:
            return
        
        # Обновляем данные для виджета с общей статистикой
        overview_stats_widget = next((w for w in view.widgets if w.widget_id == "overview_stats"), None)
        if overview_stats_widget:
            # Получаем статистику по проектам
            project_stats = self.project_management.get_project_statistics()
            
            # Получаем статистику по отчетам
            report_stats = self.report_generator.get_report_statistics()
            
            # Объединяем статистику
            overview_stats = {**project_stats, **report_stats}
            
            # Обновляем данные виджета
            overview_stats_widget.update_data(overview_stats)
        
        # Обновляем данные для виджета с последними анализами
        recent_analyses_widget = next((w for w in view.widgets if w.widget_id == "recent_analyses"), None)
        if recent_analyses_widget:
            # Получаем последние анализы
            recent_analyses = self.project_management.get_recent_analyses(10)
            
            # Преобразуем анализы в формат для виджета
            analyses_data = []
            for analysis in recent_analyses:
                project = self.project_management.get_project(analysis.project_id)
                project_name = project.name if project else "Неизвестный проект"
                
                analyses_data.append({
                    "analysis_id": analysis.analysis_id,
                    "project_id": analysis.project_id,
                    "project_name": project_name,
                    "created_at": analysis.created_at,
                    "status": analysis.status
                })
            
            # Обновляем данные виджета
            recent_analyses_widget.update_data({"items": analyses_data})
        
        # Обновляем данные для виджета с последними отчетами
        recent_reports_widget = next((w for w in view.widgets if w.widget_id == "recent_reports"), None)
        if recent_reports_widget:
            # Получаем последние отчеты
            recent_reports = self.report_generator.get_recent_reports(10)
            
            # Преобразуем отчеты в формат для виджета
            reports_data = []
            for report in recent_reports:
                project = self.project_management.get_project(report.project_id) if report.project_id else None
                project_name = project.name if project else "Не указан"
                
                reports_data.append({
                    "report_id": report.report_id,
                    "name": report.name,
                    "project_id": report.project_id,
                    "project_name": project_name,
                    "updated_at": report.updated_at,
                    "status": report.status
                })
            
            # Обновляем данные виджета
            recent_reports_widget.update_data({"items": reports_data})
        
        # Обновляем данные для виджета с графиком активности
        activity_chart_widget = next((w for w in view.widgets if w.widget_id == "activity_chart"), None)
        if activity_chart_widget:
            # Заглушка для графика активности
            # В реальном приложении здесь должен быть код для получения данных активности
            
            # Генерируем тестовые данные
            from datetime import timedelta
            import random
            
            activity_data = []
            for i in range(30, 0, -1):
                date = datetime.now() - timedelta(days=i)
                activity_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "projects": random.randint(0, 5),
                    "analyses": random.randint(0, 10),
                    "reports": random.randint(0, 8)
                })
            
            # Обновляем данные виджета
            activity_chart_widget.update_data({"items": activity_data})
        
    def _refresh_projects_data(self):
        """Обновляет данные для представления проектов."""
        view = self.views.get("projects")
        if not view:
            return
        
        # Обновляем данные для виджета со списком проектов
        projects_list_widget = next((w for w in view.widgets if w.widget_id == "projects_list"), None)
        if projects_list_widget:
            # Получаем проекты
            projects = self.project_management.get_projects()
            
            # Преобразуем проекты в формат для виджета
            projects_data = []
            for project in projects:
                projects_data.append({
                    "project_id": project.project_id,
                    "name": project.name,
                    "url": project.url,
                    "description": project.description,
                    "status": project.status,
                    "analyses_count": len(project.analyses),
                    "created_at": project.created_at,
                    "updated_at": project.updated_at
                })
            
            # Обновляем данные виджета
            projects_list_widget.update_data({"items": projects_data})
        
    def _refresh_reports_data(self):
        """Обновляет данные для представления отчетов."""
        view = self.views.get("reports")
        if not view:
            return
        
        # Обновляем данные для виджета со списком отчетов
        reports_list_widget = next((w for w in view.widgets if w.widget_id == "reports_list"), None)
        if reports_list_widget:
            # Получаем отчеты
            reports = self.report_generator.get_reports()
            
            # Преобразуем отчеты в формат для виджета
            reports_data = []
            for report in reports:
                project = self.project_management.get_project(report.project_id) if report.project_id else None
                project_name = project.name if project else "Не указан"
                
                reports_data.append({
                    "report_id": report.report_id,
                    "name": report.name,
                    "description": report.description,
                    "project_id": report.project_id,
                    "project_name": project_name,
                    "type": report.type,
                    "status": report.status,
                    "created_at": report.created_at,
                    "updated_at": report.updated_at
                })
            
            # Обновляем данные виджета
            reports_list_widget.update_data({"items": reports_data})
        
        # Обновляем данные для виджета со списком шаблонов
        templates_list_widget = next((w for w in view.widgets if w.widget_id == "templates_list"), None)
        if templates_list_widget:
            # Получаем шаблоны
            templates = self.report_generator.get_templates()
            
            # Преобразуем шаблоны в формат для виджета
            templates_data = []
            for template in templates:
                templates_data.append({
                    "template_id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "type": template.type,
                    "created_at": template.created_at,
                    "updated_at": template.updated_at
                })
            
            # Обновляем данные виджета
            templates_list_widget.update_data({"items": templates_data})
        
    def _refresh_users_data(self):
        """Обновляет данные для представления пользователей."""
        view = self.views.get("users")
        if not view:
            return
        
        # Обновляем данные для виджета со списком пользователей
        users_list_widget = next((w for w in view.widgets if w.widget_id == "users_list"), None)
        if users_list_widget:
            # Получаем пользователей
            users = self.user_management.get_users()
            
            # Преобразуем пользователей в формат для виджета
            users_data = []
            for user in users:
                users_data.append({
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.get_full_name(),
                    "role": user.role,
                    "status": user.status,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                    "last_login": user.last_login
                })
            
            # Обновляем данные виджета
            users_list_widget.update_data({"items": users_data})
        
        # Обновляем данные для виджета с сессиями
        active_sessions_widget = next((w for w in view.widgets if w.widget_id == "active_sessions"), None)
        if active_sessions_widget:
            # Получаем активные сессии
            sessions = self.user_management.get_active_sessions()
            
            # Преобразуем сессии в формат для виджета
            sessions_data = []
            for session in sessions:
                user = self.user_management.get_user(session.user_id)
                username = user.username if user else "Неизвестный пользователь"
                
                sessions_data.append({
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "username": username,
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                    "created_at": session.created_at,
                    "expires_at": session.expires_at
                })
            
            # Обновляем данные виджета
            active_sessions_widget.update_data({"items": sessions_data})
        
    def switch_view(self, view_id: str) -> bool:
        """
        Переключает текущий вид панели управления.
        
        Args:
            view_id: ID представления
            
        Returns:
            bool: True, если представление успешно переключено, иначе False
        """
        if view_id in self.views:
            self.active_view = view_id
            return True
        return False
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку ключевых метрик.
        
        Returns:
            Dict[str, Any]: Ключевые метрики
        """
        # Получаем статистику по проектам
        project_stats = self.project_management.get_project_statistics() if self.project_management else {}
        
        # Получаем статистику по отчетам
        report_stats = self.report_generator.get_report_statistics() if self.report_generator else {}
        
        # Получаем статистику по пользователям
        user_stats = self.user_management.get_user_statistics() if self.user_management else {}
        
        # Объединяем статистику
        metrics = {**project_stats, **report_stats, **user_stats}
        
        # Добавляем вычисляемые метрики
        
        return metrics
        
    def generate_dashboard_state(self) -> Dict[str, Any]:
        """
        Генерирует текущее состояние панели управления для рендеринга.
        
        Returns:
            Dict[str, Any]: Состояние панели управления
        """
        active_view = self.views.get(self.active_view)
        
        return {
            "active_view_id": self.active_view,
            "active_view": active_view.to_dict() if active_view else None,
            "views": [view.to_dict() for view in sorted(self.views.values(), key=lambda v: v.order)],
            "metrics": self.get_metrics_summary(),
            "last_refresh": self.last_refresh.isoformat(),
            "config": {
                "theme": self.config.theme,
                "refresh_interval": self.config.refresh_interval
            }
        }
    
    def create_project(self, name: str, url: str, description: str = "", settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Создает новый проект.
        
        Args:
            name: Название проекта
            url: URL сайта или страницы проекта
            description: Описание проекта
            settings: Настройки проекта
            
        Returns:
            Dict[str, Any]: Данные созданного проекта
        """
        if not self.project_management:
            raise ValueError("ProjectManagement is not initialized")
        
        project = self.project_management.create_project(
            name=name,
            url=url,
            description=description,
            settings=settings
        )
        
        return project.to_dict()
    
    def create_report(self, name: str, template_id: str, project_id: str = None, analysis_id: str = None, description: str = "") -> Dict[str, Any]:
        """
        Создает новый отчет.
        
        Args:
            name: Название отчета
            template_id: ID шаблона
            project_id: ID проекта
            analysis_id: ID анализа
            description: Описание отчета
            
        Returns:
            Dict[str, Any]: Данные созданного отчета
        """
        if not self.report_generator:
            raise ValueError("ReportGenerator is not initialized")
        
        report = self.report_generator.create_report(
            name=name,
            template_id=template_id,
            project_id=project_id,
            analysis_id=analysis_id,
            description=description
        )
        
        return report.to_dict()
    
    def create_user(self, username: str, email: str, password: str, first_name: str = "", last_name: str = "", role: str = "user") -> Dict[str, Any]:
        """
        Создает нового пользователя.
        
        Args:
            username: Имя пользователя
            email: Email пользователя
            password: Пароль
            first_name: Имя
            last_name: Фамилия
            role: Роль пользователя
            
        Returns:
            Dict[str, Any]: Данные созданного пользователя
        """
        if not self.user_management:
            raise ValueError("UserManagement is not initialized")
        
        user = self.user_management.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            role=role
        )
        
        # Не возвращаем хеш пароля
        user_dict = user.to_dict()
        user_dict.pop("password_hash", None)
        
        return user_dict


# Функция для создания экземпляра DashboardUI с настройками по умолчанию
def create_dashboard(config_path: Optional[str] = None) -> DashboardUI:
    """
    Создает экземпляр панели управления с настройками.
    
    Args:
        config_path: Путь к файлу конфигурации
    
    Returns:
        DashboardUI: Экземпляр панели управления
    """
    config = None
    if config_path and Path(config_path).exists():
        config = DashboardConfig.load_from_file(config_path)
    
    dashboard = DashboardUI(config)
    dashboard.initialize()
    return dashboard
