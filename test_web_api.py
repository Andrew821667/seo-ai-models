
"""
Тестовый скрипт для проверки API.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь для импорта
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from seo_ai_models.web.api.app import create_app

# Создаем экземпляр приложения FastAPI
app = create_app()

if __name__ == "__main__":
    print("Starting SEO AI Models API server...")
    print("Access the API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
