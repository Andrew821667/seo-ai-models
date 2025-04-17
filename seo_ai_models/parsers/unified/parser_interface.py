"""
Интерфейс для парсеров в проекте SEO AI Models.
Определяет базовые методы, которые должны быть реализованы всеми парсерами.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class ParserInterface(ABC):
    """
    Абстрактный базовый класс, определяющий интерфейс парсера.
    Все парсеры должны реализовывать эти методы.
    """
    
    @abstractmethod
    def parse_url(self, url: str, **options) -> Dict[str, Any]:
        """
        Парсинг одного URL, извлечение контента и метаданных.
        
        Args:
            url: URL для парсинга
            **options: Дополнительные опции парсинга
            
        Returns:
            Dict[str, Any]: Результат парсинга
        """
        pass
    
    @abstractmethod
    def parse_html(self, html: str, base_url: Optional[str] = None, **options) -> Dict[str, Any]:
        """
        Парсинг HTML-контента.
        
        Args:
            html: HTML-контент для парсинга
            base_url: Базовый URL для относительных ссылок
            **options: Дополнительные опции парсинга
            
        Returns:
            Dict[str, Any]: Результат парсинга
        """
        pass
    
    @abstractmethod
    def crawl_site(self, base_url: str, **options) -> Dict[str, Any]:
        """
        Сканирование сайта и парсинг его страниц.
        
        Args:
            base_url: Начальный URL для сканирования
            **options: Дополнительные опции сканирования
            
        Returns:
            Dict[str, Any]: Результаты сканирования и парсинга
        """
        pass
    
    @abstractmethod
    def parse_search_results(self, query: str, **options) -> Dict[str, Any]:
        """
        Парсинг результатов поиска по запросу.
        
        Args:
            query: Поисковый запрос
            **options: Дополнительные опции парсинга
            
        Returns:
            Dict[str, Any]: Результаты парсинга поисковой выдачи
        """
        pass
