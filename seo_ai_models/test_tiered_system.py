"""
Тестирование многоуровневой системы TieredAdvisor.

Скрипт выполняет базовое тестирование функциональности
компонентов многоуровневой системы.
"""

import os
import sys
import logging
from typing import Dict, Any

# Добавляем родительскую директорию в Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Импорт компонентов многоуровневой системы
from seo_ai_models.models.tiered_system.core.tiered_advisor import TieredAdvisor, TierPlan
from seo_ai_models.models.tiered_system.core.resource_optimizer import (
    ResourceOptimizer,
    ResourceType,
)
from seo_ai_models.models.tiered_system.core.feature_gating import FeatureGating
from seo_ai_models.models.tiered_system.core.llm_token_manager import LLMTokenManager


def test_tiered_advisor():
    """Тестирование TieredAdvisor."""
    logger.info("=== Тестирование TieredAdvisor ===")

    # Тестируем создание advisor для разных планов
    plans = ["micro", "basic", "professional", "enterprise"]

    for plan in plans:
        try:
            advisor = TieredAdvisor(tier=plan)
            tier_info = advisor.get_tier_info()
            logger.info(f"План '{plan}' инициализирован успешно")
            logger.info(f"Доступные функции: {len(tier_info.get('allowed_features', []))} шт.")
        except Exception as e:
            logger.error(f"Ошибка при инициализации плана '{plan}': {e}")

    # Тестируем обновление плана
    try:
        advisor = TieredAdvisor(tier="basic")
        logger.info("Обновление плана с 'basic' до 'professional'")
        result = advisor.upgrade_tier("professional")
        new_tier_info = advisor.get_tier_info()
        logger.info(f"Обновление выполнено успешно: {result}")
        logger.info(
            f"Новые доступные функции: {len(new_tier_info.get('allowed_features', []))} шт."
        )
    except Exception as e:
        logger.error(f"Ошибка при обновлении плана: {e}")


def test_resource_optimizer():
    """Тестирование ResourceOptimizer."""
    logger.info("=== Тестирование ResourceOptimizer ===")

    try:
        # Создаем оптимизатор для разных планов
        optimizers = {
            plan: ResourceOptimizer(tier=TierPlan(plan))
            for plan in ["micro", "basic", "professional", "enterprise"]
        }

        # Проверяем лимиты ресурсов
        for plan, optimizer in optimizers.items():
            limits = optimizer.get_resource_limits()
            logger.info(f"План '{plan}' имеет следующие лимиты ресурсов:")
            for resource_type, limit in limits.items():
                logger.info(f"  - {resource_type}: {limit}")

        # Тестируем оптимизацию конфигурации
        config = {
            "max_parallel_processes": 8,
            "use_cache": False,
            "cache_ttl": 1800,
            "request_timeout": 60,
        }

        for plan, optimizer in optimizers.items():
            optimized_config = optimizer.optimize_config(config)
            logger.info(f"План '{plan}' оптимизировал конфигурацию:")
            for key, value in optimized_config.items():
                if key in config and config[key] != value:
                    logger.info(f"  - {key}: {config[key]} -> {value}")

    except Exception as e:
        logger.error(f"Ошибка при тестировании ResourceOptimizer: {e}")


def test_feature_gating():
    """Тестирование FeatureGating."""
    logger.info("=== Тестирование FeatureGating ===")

    try:
        # Создаем FeatureGating для разных планов
        gating_components = {
            plan: FeatureGating(tier=TierPlan(plan))
            for plan in ["micro", "basic", "professional", "enterprise"]
        }

        # Проверяем доступные функции
        for plan, gating in gating_components.items():
            features = gating.get_allowed_features()
            logger.info(f"План '{plan}' имеет {len(features)} доступных функций")

            # Проверяем несколько конкретных функций
            test_features = [
                "basic_content_analysis",  # Должна быть у всех
                "llm_compatibility",  # Только у professional и enterprise
                "multi_model_analysis",  # Только у enterprise
            ]

            for feature in test_features:
                is_allowed = gating.is_feature_allowed(feature)
                logger.info(
                    f"  - Функция '{feature}': {'Доступна' if is_allowed else 'Недоступна'}"
                )

        # Тестируем фильтрацию аргументов
        test_kwargs = {
            "use_llm": True,
            "analyze_llm_compatibility": True,
            "use_multi_model": True,
            "non_feature_arg": "value",
        }

        for plan, gating in gating_components.items():
            filtered_kwargs = gating.filter_allowed_args(test_kwargs)
            logger.info(f"План '{plan}' отфильтровал аргументы: {filtered_kwargs}")

    except Exception as e:
        logger.error(f"Ошибка при тестировании FeatureGating: {e}")


def test_llm_token_manager():
    """Тестирование LLMTokenManager."""
    logger.info("=== Тестирование LLMTokenManager ===")

    try:
        # Создаем LLMTokenManager для разных планов
        token_managers = {
            plan: LLMTokenManager(tier=TierPlan(plan))
            for plan in ["basic", "professional", "enterprise"]  # Пропускаем micro
        }

        # Проверяем лимиты токенов
        for plan, manager in token_managers.items():
            token_info = manager.get_token_info()
            logger.info(f"План '{plan}' имеет следующие лимиты токенов:")
            for key, value in token_info.items():
                logger.info(f"  - {key}: {value}")

        # Тестируем проверку доступности токенов
        test_content_lengths = [1000, 10000, 100000, 1000000]  # В символах

        for plan, manager in token_managers.items():
            logger.info(f"План '{plan}' - проверка доступности токенов:")
            for length in test_content_lengths:
                is_available = manager.check_tokens_available(length)
                logger.info(
                    f"  - Контент длиной {length} символов: {'Доступно' if is_available else 'Недоступно'}"
                )

        # Тестируем отслеживание использования токенов
        for plan, manager in token_managers.items():
            usage_info = manager.track_token_usage(
                input_tokens=100, output_tokens=50, provider="openai", model="gpt-4"
            )
            logger.info(f"План '{plan}' - отслеживание использования токенов:")
            for key, value in usage_info.items():
                logger.info(f"  - {key}: {value}")

    except Exception as e:
        logger.error(f"Ошибка при тестировании LLMTokenManager: {e}")


if __name__ == "__main__":
    logger.info("Запуск тестирования многоуровневой системы")

    # Запускаем тесты
    test_tiered_advisor()
    test_resource_optimizer()
    test_feature_gating()
    test_llm_token_manager()

    logger.info("Тестирование завершено")
