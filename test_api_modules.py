
"""
Тестовый скрипт для проверки модулей API без запуска сервера.
"""

import os
import sys
from pathlib import Path
import json

# Добавляем корневую директорию проекта в путь для импорта
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_api_models():
    print("Testing API Models...")
    
    # Импортируем и проверяем модели auth
    from seo_ai_models.web.api.models.auth import UserCreate, UserLogin, TokenResponse, UserResponse, UserRole
    print("Auth models loaded successfully")
    
    # Импортируем и проверяем модели projects
    from seo_ai_models.web.api.models.projects import ProjectCreate, ProjectResponse, TaskCreate, TaskResponse, ProjectStatus
    print("Project models loaded successfully")
    
    # Импортируем и проверяем модели reports
    from seo_ai_models.web.api.models.reports import ReportCreate, ReportResponse, VisualizationCreate, ReportType, VisualizationType
    print("Report models loaded successfully")
    
    print("API Models test completed successfully!")

def test_api_routers():
    print("\nTesting API Routers...")
    
    # Импортируем и проверяем роутер auth
    from seo_ai_models.web.api.routers.auth import router as auth_router
    print("Auth router loaded successfully")
    
    # Импортируем и проверяем роутер projects
    from seo_ai_models.web.api.routers.projects import router as projects_router
    print("Projects router loaded successfully")
    
    # Импортируем и проверяем роутер cms_connectors
    from seo_ai_models.web.api.routers.cms_connectors import router as cms_router
    print("CMS Connectors router loaded successfully")
    
    # Импортируем и проверяем роутер webhooks
    from seo_ai_models.web.api.routers.webhooks import router as webhooks_router
    print("Webhooks router loaded successfully")
    
    print("API Routers test completed successfully!")

def test_api_app():
    print("\nTesting API Application...")
    
    # Импортируем и создаем приложение
    from seo_ai_models.web.api.app import create_app
    app = create_app()
    print("API application created successfully!")
    
    print("API routes:")
    for route in app.routes:
        print(f" - {route.path} [{','.join(route.methods)}]")
    
    print("API Application test completed successfully!")

# Запускаем тесты
if __name__ == "__main__":
    print("Starting tests for API modules...")
    
    try:
        test_api_models()
        test_api_routers()
        test_api_app()
        print("\nAll API module tests completed successfully!")
    except Exception as e:
        print(f"Error during tests: {str(e)}")
        import traceback
        traceback.print_exc()
