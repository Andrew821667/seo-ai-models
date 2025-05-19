
"""
Тестовый скрипт для проверки компонентов панели управления.
"""

import os
import sys
from pathlib import Path
import json

# Добавляем корневую директорию проекта в путь для импорта
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from seo_ai_models.web.dashboard.dashboard_ui import DashboardConfig, DashboardUI, create_dashboard
from seo_ai_models.web.dashboard.project_management import ProjectManager, Project, Task, ProjectStatus, TaskStatus, TaskPriority, create_project_manager
from seo_ai_models.web.dashboard.report_generator import ReportGenerator, Report, Visualization, ReportType, VisualizationType, create_report_generator
from seo_ai_models.web.dashboard.user_management import UserManager, User, UserRole, create_user_manager

# Функция для проверки DashboardUI
def test_dashboard_ui():
    print("Testing DashboardUI...")
    # Создаем конфигурацию
    config = DashboardConfig(
        api_url="http://localhost:8000",
        theme="dark",
        refresh_interval=30,
        default_views=["overview", "projects", "reports", "settings"]
    )
    
    # Создаем экземпляр панели управления
    dashboard = DashboardUI(config)
    dashboard.initialize()
    
    # Получаем состояние панели
    state = dashboard.generate_dashboard_state()
    print(f"Dashboard state: {json.dumps(state, indent=2, default=str)}")
    
    print("DashboardUI test completed successfully!")

# Функция для проверки ProjectManager
def test_project_manager():
    print("\nTesting ProjectManager...")
    # Создаем временную директорию для данных
    data_dir = Path("./test_data/projects")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем менеджер проектов
    manager = create_project_manager(str(data_dir))
    
    # Создаем проект
    project = manager.create_project(
        name="Test Project",
        description="Project for testing",
        website="https://example.com",
        status=ProjectStatus.ACTIVE
    )
    print(f"Created project: {project.to_dict()}")
    
    # Создаем задачи
    task1 = manager.create_task(
        title="Task 1",
        project_id=project.id,
        description="First test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.HIGH
    )
    print(f"Created task: {task1.to_dict()}")
    
    task2 = manager.create_task(
        title="Task 2",
        project_id=project.id,
        description="Second test task",
        status=TaskStatus.IN_PROGRESS,
        priority=TaskPriority.MEDIUM
    )
    print(f"Created task: {task2.to_dict()}")
    
    # Получаем задачи проекта
    tasks = manager.get_project_tasks(project.id)
    print(f"Project tasks count: {len(tasks)}")
    
    print("ProjectManager test completed successfully!")

# Функция для проверки ReportGenerator
def test_report_generator():
    print("\nTesting ReportGenerator...")
    # Создаем временную директорию для данных
    data_dir = Path("./test_data/reports")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем генератор отчетов
    generator = create_report_generator(str(data_dir))
    
    # Создаем отчет
    report = generator.create_report(
        title="Test Report",
        project_id="project123",
        report_type=ReportType.OVERVIEW,
        description="Report for testing"
    )
    print(f"Created report: {report.to_dict()}")
    
    # Добавляем визуализации
    visualization1 = generator.add_visualization(
        report_id=report.id,
        title="Test Visualization 1",
        visualization_type=VisualizationType.BAR_CHART,
        data={"labels": ["A", "B", "C"], "values": [10, 20, 30]},
        description="First test visualization"
    )
    print(f"Added visualization: {visualization1.to_dict()}")
    
    visualization2 = generator.add_visualization(
        report_id=report.id,
        title="Test Visualization 2",
        visualization_type=VisualizationType.PIE_CHART,
        data={"labels": ["X", "Y", "Z"], "values": [15, 25, 35]},
        description="Second test visualization"
    )
    print(f"Added visualization: {visualization2.to_dict()}")
    
    # Получаем отчет с визуализациями
    updated_report = generator.get_report(report.id)
    print(f"Report visualizations count: {len(updated_report.visualizations)}")
    
    print("ReportGenerator test completed successfully!")

# Функция для проверки UserManager
def test_user_manager():
    print("\nTesting UserManager...")
    # Создаем временную директорию для данных
    data_dir = Path("./test_data/users")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем менеджер пользователей
    manager = create_user_manager(str(data_dir))
    
    # Создаем пользователя
    user = manager.create_user(
        username="testuser",
        email="test@example.com",
        password="Test123!",
        first_name="Test",
        last_name="User",
        role=UserRole.ANALYST
    )
    print(f"Created user: {user.to_dict()}")
    
    # Проверяем аутентификацию
    auth_user = manager.authenticate("testuser", "Test123!")
    if auth_user:
        print(f"Authentication successful for user: {auth_user.username}")
    else:
        print("Authentication failed")
    
    # Создаем сессию
    session = manager.create_session(
        user_id=user.id,
        expires_in=3600,
        ip_address="127.0.0.1",
        user_agent="Test Script"
    )
    print(f"Created session: {session.to_dict()}")
    
    # Проверяем валидацию сессии
    validated_user = manager.validate_session(session.token)
    if validated_user:
        print(f"Session validation successful for user: {validated_user.username}")
    else:
        print("Session validation failed")
    
    print("UserManager test completed successfully!")

# Запускаем тесты
if __name__ == "__main__":
    print("Starting tests for dashboard components...")
    
    try:
        test_dashboard_ui()
        test_project_manager()
        test_report_generator()
        test_user_manager()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Error during tests: {str(e)}")
        import traceback
        traceback.print_exc()
