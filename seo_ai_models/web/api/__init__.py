"""
API для SEO AI Models - предоставляет доступ к функциональности
через RESTful интерфейс с использованием FastAPI.
"""

from .app import create_app

__all__ = ["create_app"]
