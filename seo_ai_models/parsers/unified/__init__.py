"""
Унифицированный парсер-модуль для проекта SEO AI Models.
Обеспечивает современный парсинг сайтов с расширенными возможностями:
- Поддержка SPA-сайтов через Playwright
- Многопоточный параллельный парсинг
- Интеграция с API поисковых систем
- Расширенный семантический анализ с NLP
"""

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.parsers.unified.parser_result import (
    ParserResult, PageData, SiteData, SearchResultData, 
    TextContent, PageStructure, MetaData
)
from seo_ai_models.parsers.unified.site_analyzer import SiteAnalyzer

# Экспортируем ключевые компоненты
__all__ = [
    'UnifiedParser',
    'SiteAnalyzer',
    'ParserResult',
    'PageData',
    'SiteData',
    'SearchResultData',
    'TextContent',
    'PageStructure',
    'MetaData'
]
