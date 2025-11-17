"""
Упрощенное тестирование многоуровневой системы TieredAdvisor.

Скрипт выполняет базовое тестирование функциональности
компонентов многоуровневой системы с заглушками вместо реальных компонентов.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Добавляем родительскую директорию в Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Импортируем только необходимое для тестирования
from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan


# Создаем заглушки для необходимых классов
class MockSEOAdvisor:
    def __init__(self, config=None):
        self.config = config or {}

    def analyze_content(self, content, keywords=None, context=None, **kwargs):
        return {
            "basic_metrics": {"word_count": len(content.split())},
            "readability": "Хорошая читаемость",
            "keywords_basic": keywords or [],
            "structure_basic": {"paragraphs": 3, "sentences": 10},
        }


class MockEnhancedSEOAdvisor(MockSEOAdvisor):
    def analyze_content(self, content, keywords=None, context=None, **kwargs):
        result = super().analyze_content(content, keywords, context, **kwargs)
        result.update(
            {
                "enhanced_metrics": {"sentiment": "positive"},
                "keyword_density": 2.5,
                "semantic_analysis": {"coherence": 0.8},
            }
        )
        return result


class MockLLMEnhancedSEOAdvisor(MockEnhancedSEOAdvisor):
    def __init__(self, config=None, api_keys=None):
        super().__init__(config)
        self.api_keys = api_keys or {}

    def analyze_content(self, content, keywords=None, context=None, **kwargs):
        result = super().analyze_content(content, keywords, context, **kwargs)
        result.update(
            {
                "llm_compatibility": 0.85,
                "citability_score": 0.75,
                "llm_eeat_analysis": {
                    "experience": 0.7,
                    "expertise": 0.8,
                    "authority": 0.6,
                    "trust": 0.9,
                },
            }
        )
        return result


def test_resource_optimizer():
    """Тестирование ResourceOptimizer."""
    logger.info("=== Тестирование ResourceOptimizer ===")

    from seo_ai_models.models.tiered_system.core.resource_optimizer import ResourceOptimizer

    try:
        # Создаем оптимизатор для разных планов
        for plan_name in ["micro", "basic", "professional", "enterprise"]:
            plan = TierPlan(plan_name)
            optimizer = ResourceOptimizer(tier=plan)

            limits = optimizer.get_resource_limits()
            logger.info(f"План '{plan_name}' имеет следующие лимиты ресурсов:")
            for resource_type, limit in limits.items():
                logger.info(f"  - {resource_type}: {limit}")

            # Тестируем оптимизацию конфигурации
            config = {
                "max_parallel_processes": 8,
                "use_cache": False,
                "cache_ttl": 1800,
                "request_timeout": 60,
            }

            optimized_config = optimizer.optimize_config(config)
            logger.info(f"План '{plan_name}' оптимизировал конфигурацию:")
            for key, value in optimized_config.items():
                if key in config and config[key] != value:
                    logger.info(f"  - {key}: {config[key]} -> {value}")

    except Exception as e:
        logger.error(f"Ошибка при тестировании ResourceOptimizer: {e}")
        import traceback

        traceback.print_exc()


def test_feature_gating():
    """Тестирование FeatureGating."""
    logger.info("=== Тестирование FeatureGating ===")

    from seo_ai_models.models.tiered_system.core.feature_gating import FeatureGating

    try:
        # Создаем FeatureGating для разных планов
        for plan_name in ["micro", "basic", "professional", "enterprise"]:
            plan = TierPlan(plan_name)
            gating = FeatureGating(tier=plan)

            features = gating.get_allowed_features()
            logger.info(f"План '{plan_name}' имеет {len(features)} доступных функций")

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

        # Тестируем фильтрацию аргументов для одного плана
        plan = TierPlan("professional")
        gating = FeatureGating(tier=plan)

        test_kwargs = {
            "use_llm": True,
            "analyze_llm_compatibility": True,
            "use_multi_model": True,
            "non_feature_arg": "value",
        }

        filtered_kwargs = gating.filter_allowed_args(test_kwargs)
        logger.info(f"План 'professional' отфильтровал аргументы: {filtered_kwargs}")

    except Exception as e:
        logger.error(f"Ошибка при тестировании FeatureGating: {e}")
        import traceback

        traceback.print_exc()


def test_llm_token_manager():
    """Тестирование LLMTokenManager (без использования CostEstimator)."""
    logger.info("=== Тестирование LLMTokenManager ===")

    # Вносим изменения в LLMTokenManager, чтобы он работал без CostEstimator
    from seo_ai_models.models.tiered_system.core.llm_token_manager import LLMTokenManager

    # Переопределяем метод track_token_usage для тестирования
    original_track_token_usage = LLMTokenManager.track_token_usage

    def mock_track_token_usage(self, input_tokens, output_tokens, provider, model):
        self._check_reset_counters()
        total_tokens = input_tokens + output_tokens
        self.tokens_used += total_tokens
        self.requests_used += 1

        usage_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": 0.01,  # Фиктивная стоимость
            "currency": "USD",
            "tokens_remaining": self._get_tokens_remaining(),
            "requests_remaining": self._get_requests_remaining(),
            "reset_time": self.reset_time.isoformat(),
        }

        return usage_info

    try:
        # Временно заменяем метод
        LLMTokenManager.track_token_usage = mock_track_token_usage

        # Создаем LLMTokenManager для разных планов
        for plan_name in ["basic", "professional", "enterprise"]:
            plan = TierPlan(plan_name)
            manager = LLMTokenManager(tier=plan)

            token_info = manager.get_token_info()
            logger.info(f"План '{plan_name}' имеет следующие лимиты токенов:")
            for key, value in token_info.items():
                logger.info(f"  - {key}: {value}")

        # Тестируем одним конкретным менеджером
        plan = TierPlan("professional")
        manager = LLMTokenManager(tier=plan)

        # Тестируем проверку доступности токенов
        test_content_lengths = [1000, 10000, 100000, 1000000]  # В символах

        logger.info(f"План 'professional' - проверка доступности токенов:")
        for length in test_content_lengths:
            is_available = manager.check_tokens_available(length)
            logger.info(
                f"  - Контент длиной {length} символов: {'Доступно' if is_available else 'Недоступно'}"
            )

        # Тестируем отслеживание использования токенов
        usage_info = manager.track_token_usage(
            input_tokens=100, output_tokens=50, provider="openai", model="gpt-4"
        )
        logger.info(f"План 'professional' - отслеживание использования токенов:")
        for key, value in usage_info.items():
            logger.info(f"  - {key}: {value}")

    except Exception as e:
        logger.error(f"Ошибка при тестировании LLMTokenManager: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Восстанавливаем оригинальный метод
        LLMTokenManager.track_token_usage = original_track_token_usage


def test_tiered_advisor():
    """Тестирование TieredAdvisor с заглушками."""
    logger.info("=== Тестирование TieredAdvisor с заглушками ===")

    import importlib.util

    # Проверяем, доступен ли основной класс TieredAdvisor
    spec = importlib.util.find_spec("seo_ai_models.models.tiered_system.core.tiered_advisor")

    if spec is None:
        logger.error("Модуль tiered_advisor не найден")
        return

    # Импортируем модуль
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Сохраняем оригинальные классы
    original_SEOAdvisor = module.SEOAdvisor
    original_EnhancedSEOAdvisor = module.EnhancedSEOAdvisor
    original_LLMEnhancedSEOAdvisor = module.LLMEnhancedSEOAdvisor

    try:
        # Заменяем классы на заглушки
        module.SEOAdvisor = MockSEOAdvisor
        module.EnhancedSEOAdvisor = MockEnhancedSEOAdvisor
        module.LLMEnhancedSEOAdvisor = MockLLMEnhancedSEOAdvisor

        # Теперь используем TieredAdvisor
        TieredAdvisor = module.TieredAdvisor

        # Тестируем создание advisor для разных планов
        for plan_name in ["micro", "basic", "professional", "enterprise"]:
            try:
                advisor = TieredAdvisor(tier=plan_name)
                tier_info = advisor.get_tier_info()
                logger.info(f"План '{plan_name}' инициализирован успешно")
                logger.info(f"Доступные функции: {len(tier_info.get('allowed_features', []))} шт.")

                # Проверяем базовый функционал
                result = advisor.analyze_content(
                    content="Это тестовый контент для анализа.",
                    keywords=["test", "content"],
                    context={"url": "http://example.com"},
                )
                logger.info(f"Анализ выполнен, получено {len(result)} метрик")

            except Exception as e:
                logger.error(f"Ошибка при инициализации плана '{plan_name}': {e}")
                import traceback

                traceback.print_exc()

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
            import traceback

            traceback.print_exc()

    except Exception as e:
        logger.error(f"Ошибка при тестировании TieredAdvisor: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Восстанавливаем оригинальные классы
        module.SEOAdvisor = original_SEOAdvisor
        module.EnhancedSEOAdvisor = original_EnhancedSEOAdvisor
        module.LLMEnhancedSEOAdvisor = original_LLMEnhancedSEOAdvisor


if __name__ == "__main__":
    logger.info("Запуск упрощенного тестирования многоуровневой системы")

    # Запускаем тесты по отдельности
    test_resource_optimizer()
    test_feature_gating()
    test_llm_token_manager()
    test_tiered_advisor()

    logger.info("Тестирование завершено")
