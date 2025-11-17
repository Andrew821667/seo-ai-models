"""
Демонстрационный скрипт для локальных оптимизаций LLM.

Скрипт демонстрирует использование компонентов локальных оптимизаций для:
1. Работы с локальными LLM-моделями
2. Гибридной обработки (облако + локальные ресурсы)
3. Интеллектуального кэширования
4. Офлайн-режима работы
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

from seo_ai_models.models.llm_integration.service.llm_service import LLMService
from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
from seo_ai_models.models.llm_integration.service.cost_estimator import CostEstimator
from seo_ai_models.models.llm_integration.local_optimizations.local_llm_manager import (
    LocalLLMManager,
)
from seo_ai_models.models.llm_integration.local_optimizations.hybrid_processing_pipeline import (
    HybridProcessingPipeline,
)
from seo_ai_models.models.llm_integration.local_optimizations.intelligent_cache import (
    IntelligentCache,
)
from seo_ai_models.models.llm_integration.local_optimizations.offline_analysis_mode import (
    OfflineAnalysisMode,
)

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_services(openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Настраивает сервисы для демонстрации.

    Args:
        openai_api_key: API ключ OpenAI (опционально)

    Returns:
        Dict[str, Any]: Словарь с сервисами
    """
    # Если API ключ не указан, пытаемся получить его из переменной окружения
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Создаем экземпляры базовых сервисов
    llm_service = LLMService(openai_api_key=openai_api_key)
    prompt_generator = PromptGenerator()
    cost_estimator = CostEstimator()

    # Создаем экземпляры компонентов локальных оптимизаций
    local_llm_manager = LocalLLMManager()
    cache = IntelligentCache()
    hybrid_pipeline = HybridProcessingPipeline(llm_service, local_llm_manager, cost_estimator)
    offline_mode = OfflineAnalysisMode(local_llm_manager, cache, cost_estimator)

    # Возвращаем словарь с сервисами
    return {
        "llm_service": llm_service,
        "prompt_generator": prompt_generator,
        "cost_estimator": cost_estimator,
        "local_llm_manager": local_llm_manager,
        "cache": cache,
        "hybrid_pipeline": hybrid_pipeline,
        "offline_mode": offline_mode,
    }


def demonstrate_local_llm_manager(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу LocalLLMManager.

    Args:
        services: Словарь с сервисами
    """
    logger.info("=== Демонстрация LocalLLMManager ===")

    local_llm_manager = services["local_llm_manager"]

    # Получаем информацию о доступных моделях
    models_info = local_llm_manager.get_available_models()

    logger.info(f"Доступные модели: {', '.join(models_info.get('available_models', []))}")
    logger.info(f"Загруженная модель: {models_info.get('loaded_model')}")

    # Пример выбора оптимальной модели
    prompt = "Расскажи о применении машинного обучения в SEO"

    logger.info("\n--- Выбор оптимальной модели ---")
    selected_model = local_llm_manager.select_optimal_model(prompt, "medium")
    logger.info(f"Выбранная модель: {selected_model}")

    # Можно добавить загрузку и запрос к модели, если она доступна
    if models_info.get("available_models"):
        # Для демонстрации пропускаем загрузку модели, которая может занять много времени
        logger.info("В демонстрационном режиме пропускаем загрузку и запрос к модели.")
    else:
        logger.info(
            "Локальные модели не найдены. Для использования LocalLLMManager необходимо загрузить модели."
        )


def demonstrate_hybrid_pipeline(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу HybridProcessingPipeline.

    Args:
        services: Словарь с сервисами
    """
    logger.info("\n=== Демонстрация HybridProcessingPipeline ===")

    hybrid_pipeline = services["hybrid_pipeline"]

    # Настраиваем параметры гибридного пайплайна
    hybrid_pipeline.configure(
        fallback_mode=True, quality_threshold=0.7, cost_optimization_level="medium"
    )

    # Пример запроса с разными стратегиями
    prompt = "Объясни, как оптимизировать контент для LLM-поисковиков"

    logger.info("\n--- Стратегия auto ---")
    # В режиме демонстрации пропускаем реальные запросы, которые требуют API ключи и модели
    logger.info("В демонстрационном режиме пропускаем реальные запросы.")
    logger.info("Стратегия auto автоматически выбирает между облачной и локальной обработкой.")

    logger.info("\n--- Стратегия cloud_first ---")
    logger.info("Стратегия cloud_first приоритизирует использование облачных моделей.")

    logger.info("\n--- Стратегия local_first ---")
    logger.info("Стратегия local_first приоритизирует использование локальных моделей.")

    logger.info("\n--- Стратегия cost_optimized ---")
    logger.info("Стратегия cost_optimized выбирает наиболее дешевый вариант обработки.")


def demonstrate_intelligent_cache(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу IntelligentCache.

    Args:
        services: Словарь с сервисами
    """
    logger.info("\n=== Демонстрация IntelligentCache ===")

    cache = services["cache"]

    # Получаем статистику кэша
    stats = cache.get_cache_stats()

    logger.info(
        f"Размер кэша: {stats.get('cache_size_mb', 0):.2f} МБ из {stats.get('max_cache_size_mb', 0)} МБ"
    )
    logger.info(f"Количество записей: {stats.get('total_entries', 0)}")
    logger.info(f"Количество попаданий в кэш: {stats.get('total_cache_hits', 0)}")
    logger.info(f"Количество промахов кэша: {stats.get('total_cache_misses', 0)}")
    logger.info(f"Процент попаданий: {stats.get('cache_hit_rate', 0) * 100:.2f}%")
    logger.info(f"Сэкономлено: {stats.get('total_cost_saved', 0):.6f} руб.")

    # Пример сохранения и получения результата из кэша
    logger.info("\n--- Пример работы с кэшем ---")

    # Тестовый контент
    content = "Это тестовый контент для демонстрации работы кэша."

    # Тестовый результат анализа
    test_result = {"test_score": 7, "provider": "test", "cost": 0.01}

    # Сохраняем результат в кэш
    cache.save_analysis_to_cache(content, "test_analysis", test_result)
    logger.info("Результат сохранен в кэш")

    # Получаем результат из кэша
    cached_result = cache.get_analysis_from_cache(content, "test_analysis")

    if cached_result:
        logger.info("Результат получен из кэша")
        logger.info(f"Тестовая оценка: {cached_result.get('test_score')}")
        logger.info(f"Из кэша: {cached_result.get('from_cache', False)}")
    else:
        logger.info("Результат не найден в кэше")

    # Очистка тестового кэша
    # cache.clear_cache()
    # logger.info("Кэш очищен")


def demonstrate_offline_mode(services: Dict[str, Any]) -> None:
    """
    Демонстрирует работу OfflineAnalysisMode.

    Args:
        services: Словарь с сервисами
    """
    logger.info("\n=== Демонстрация OfflineAnalysisMode ===")

    offline_mode = services["offline_mode"]

    # Включаем офлайн-режим
    offline_mode.enable_offline_mode()
    logger.info(f"Офлайн-режим включен: {offline_mode.is_offline_mode_enabled()}")

    # Настраиваем параметры офлайн-режима
    offline_mode.configure(
        default_quality="medium",
        fallback_to_cached=True,
        cache_all_results=True,
        prefetch_enabled=True,
    )

    # Пример анализа в офлайн-режиме
    logger.info("\n--- Пример анализа в офлайн-режиме ---")

    # Тестовый контент
    content = """
    # Оптимизация контента для поисковых систем
    
    Поисковая оптимизация (SEO) - это процесс улучшения видимости сайта
    в результатах поисковых систем. Хорошо оптимизированный контент
    имеет больше шансов привлечь целевую аудиторию.
    
    ## Ключевые факторы оптимизации
    
    - Релевантные ключевые слова
    - Качественный и уникальный контент
    - Понятная структура с заголовками
    - Оптимизированные мета-теги
    """

    # В режиме демонстрации пропускаем реальные запросы, которые требуют локальные модели
    logger.info("В демонстрационном режиме пропускаем реальные запросы.")
    logger.info("Офлайн-режим позволяет выполнять анализ контента без доступа к облачным API.")
    logger.info(
        "Поддерживаемые типы анализа: citability, content_structure, keyword_analysis, eeat"
    )

    # Выключаем офлайн-режим
    offline_mode.disable_offline_mode()
    logger.info(f"Офлайн-режим выключен: {offline_mode.is_offline_mode_enabled()}")


def main():
    parser = argparse.ArgumentParser(description="Демонстрация локальных оптимизаций LLM")
    parser.add_argument("--api_key", help="API ключ OpenAI")
    parser.add_argument(
        "--demo",
        choices=["all", "local_llm", "hybrid", "cache", "offline"],
        default="all",
        help="Какую демонстрацию запустить",
    )

    args = parser.parse_args()

    # Настраиваем сервисы
    services = setup_services(args.api_key)

    # Запускаем выбранную демонстрацию
    if args.demo in ["all", "local_llm"]:
        demonstrate_local_llm_manager(services)

    if args.demo in ["all", "hybrid"]:
        demonstrate_hybrid_pipeline(services)

    if args.demo in ["all", "cache"]:
        demonstrate_intelligent_cache(services)

    if args.demo in ["all", "offline"]:
        demonstrate_offline_mode(services)

    logger.info("\n=== Демонстрация завершена ===")


if __name__ == "__main__":
    main()
