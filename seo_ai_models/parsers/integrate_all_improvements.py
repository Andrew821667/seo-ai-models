"""
Интеграция всех улучшений для парсера SPA.
"""

# Импортируем все компоненты
from seo_ai_models.parsers.parsing_pipeline import ParsingPipeline
from seo_ai_models.parsers.parsing_pipeline_update import integrate_spa_support
from seo_ai_models.parsers.extractors.spa_content_extractor_ajax import (
    update_spa_extractor_with_ajax,
)
from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor
from seo_ai_models.parsers.extractors.spa_content_extractor_fix import fix_spa_content_extractor
from seo_ai_models.parsers.extractors.meta_extractor_update import update_meta_extractor
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.adaptive_parsing_pipeline_cache import add_caching_support
from seo_ai_models.parsers.adaptive_parsing_pipeline import AdaptiveParsingPipeline


def integrate_all_improvements():
    """
    Интегрирует все улучшения в модули парсера.
    """
    # Сначала исправляем SPAContentExtractor
    FixedSPAContentExtractor = fix_spa_content_extractor(SPAContentExtractor)

    # Затем обновляем его с поддержкой AJAX
    SPAContentExtractorWithAJAX = update_spa_extractor_with_ajax(FixedSPAContentExtractor)

    # Обновляем MetaExtractor
    update_meta_extractor(MetaExtractor)

    # Обновляем AdaptiveParsingPipeline с поддержкой кэширования
    CachedAdaptiveParsingPipeline = add_caching_support(AdaptiveParsingPipeline)

    # Готовим функцию для интеграции всех улучшений в ParsePipeline
    def get_enhanced_pipeline(user_agent="SEOAIModels EnhancedParser/1.0", **kwargs):
        """
        Создает полностью улучшенный конвейер парсинга.

        Args:
            user_agent: User-Agent для запросов
            **kwargs: Дополнительные аргументы для конфигурации

        Returns:
            Улучшенный ParsePipeline
        """
        # Создаем базовый конвейер
        pipeline = ParsingPipeline(user_agent=user_agent)

        # Добавляем поддержку SPA
        pipeline = integrate_spa_support(pipeline)

        # Конфигурируем параметры SPA
        headless = kwargs.get("headless", True)
        wait_for_idle = kwargs.get("wait_for_idle", 2000)
        wait_for_timeout = kwargs.get("wait_for_timeout", 10000)
        browser_type = kwargs.get("browser_type", "chromium")

        pipeline.configure_spa_options(
            wait_for_idle=wait_for_idle,
            wait_for_timeout=wait_for_timeout,
            headless=headless,
            browser_type=browser_type,
        )

        # Настраиваем включение/выключение режима SPA
        force_spa_mode = kwargs.get("force_spa_mode", False)
        pipeline.set_spa_mode(force_spa_mode)

        return pipeline

    # Возвращаем обновленные классы и функцию создания
    return {
        "FixedSPAContentExtractor": FixedSPAContentExtractor,
        "SPAContentExtractorWithAJAX": SPAContentExtractorWithAJAX,
        "CachedAdaptiveParsingPipeline": CachedAdaptiveParsingPipeline,
        "get_enhanced_pipeline": get_enhanced_pipeline,
    }
