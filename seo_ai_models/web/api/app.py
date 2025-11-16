
"""
Модуль инициализации FastAPI приложения для RESTful API.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional, Dict, Any

# Конфигурация логгера (СНАЧАЛА!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импортируем роутеры
from .routers import auth, projects, cms_connectors, webhooks

# Новые роутеры для enhanced functionality (опционально)
try:
    from ...api.routes import auth_routes, enhanced_analysis_routes
    from ...api.websocket import routes as ws_routes
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Создает и настраивает FastAPI приложение.
    
    Args:
        config: Словарь с настройками приложения.
        
    Returns:
        FastAPI: Настроенное приложение.
    """
    app = FastAPI(
        title="SEO AI Models API",
        description="API для доступа к функциональности SEO AI Models",
        version="0.1.0",
    )
    
    # Настройка CORS для доступа из браузера
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # В продакшене лучше указать конкретные домены
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Добавление роутеров
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
    app.include_router(cms_connectors.router, prefix="/api/cms", tags=["CMS Connectors"])
    app.include_router(webhooks.router, prefix="/api/webhooks", tags=["Webhooks"])

    # Новые enhanced роутеры (если доступны)
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            app.include_router(auth_routes.router, prefix="/api/v2", tags=["Auth V2"])
            app.include_router(enhanced_analysis_routes.router, prefix="/api/v2", tags=["Enhanced Analysis"])
            app.include_router(ws_routes.router, tags=["WebSocket"])
            logger.info("Enhanced features (Auth V2, WebSocket) enabled")
        except Exception as e:
            logger.error(f"Failed to load enhanced features: {e}")
    else:
        logger.info("Running in basic mode (enhanced features disabled)")
    
    @app.get("/api/health")
    async def health_check():
        """Проверка работоспособности API."""
        return {"status": "ok", "version": app.version}
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting up SEO AI Models API")
        # Инициализация необходимых сервисов
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down SEO AI Models API")
        # Освобождение ресурсов
    
    return app
