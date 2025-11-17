# -*- coding: utf-8 -*-
"""
ValueDemonstrator - Демонстратор ценности платных функций для Freemium-модели.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import json
import os

from seo_ai_models.models.freemium.core.enums import FreemiumPlan

logger = logging.getLogger(__name__)


class ValueDemonstrator:
    """
    Демонстрирует ценность платных функций для Freemium-модели.

    Помогает пользователям бесплатного плана понять преимущества
    платных планов через демонстрации и примеры.
    """

    # Функции и их доступность в разных планах
    FEATURES = {
        "basic_analysis": {
            FreemiumPlan.FREE: True,
            FreemiumPlan.MICRO: True,
            FreemiumPlan.BASIC: True,
            FreemiumPlan.PROFESSIONAL: True,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Базовый анализ контента для SEO",
        },
        "advanced_analysis": {
            FreemiumPlan.FREE: False,
            FreemiumPlan.MICRO: True,
            FreemiumPlan.BASIC: True,
            FreemiumPlan.PROFESSIONAL: True,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Расширенный анализ контента для SEO",
        },
        "bulk_analysis": {
            FreemiumPlan.FREE: False,
            FreemiumPlan.MICRO: True,
            FreemiumPlan.BASIC: True,
            FreemiumPlan.PROFESSIONAL: True,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Массовый анализ URL",
        },
        "competitor_analysis": {
            FreemiumPlan.FREE: False,
            FreemiumPlan.MICRO: False,
            FreemiumPlan.BASIC: True,
            FreemiumPlan.PROFESSIONAL: True,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Анализ контента конкурентов",
        },
        "llm_integration": {
            FreemiumPlan.FREE: False,
            FreemiumPlan.MICRO: False,
            FreemiumPlan.BASIC: False,
            FreemiumPlan.PROFESSIONAL: True,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Интеграция с LLM для улучшения оптимизации",
        },
        "api_access": {
            FreemiumPlan.FREE: False,
            FreemiumPlan.MICRO: False,
            FreemiumPlan.BASIC: True,
            FreemiumPlan.PROFESSIONAL: True,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Доступ к API для интеграции с другими системами",
        },
        "custom_metrics": {
            FreemiumPlan.FREE: False,
            FreemiumPlan.MICRO: False,
            FreemiumPlan.BASIC: False,
            FreemiumPlan.PROFESSIONAL: True,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Настраиваемые метрики для анализа",
        },
        "dedicated_support": {
            FreemiumPlan.FREE: False,
            FreemiumPlan.MICRO: False,
            FreemiumPlan.BASIC: False,
            FreemiumPlan.PROFESSIONAL: False,
            FreemiumPlan.ENTERPRISE: True,
            "description": "Выделенная поддержка",
        },
    }

    # Примеры для демонстрации функций
    FEATURE_EXAMPLES = {
        "basic_analysis": {
            "example": "Базовый анализ контента определяет основные метрики SEO, такие как длина текста, читаемость, базовые ключевые слова. Пример результата: {'word_count': 500, 'readability': 'средняя', 'keyword_density': {'seo': 2.5}}",
            "image_url": "basic_analysis_example.png",
        },
        "advanced_analysis": {
            "example": "Расширенный анализ включает семантический анализ, структуру контента, соответствие E-E-A-T. Пример результата: {'semantic_depth': 0.85, 'structure_score': 0.9, 'eeat_score': 0.75, 'topic_relevance': 0.92}",
            "image_url": "advanced_analysis_example.png",
        },
        "bulk_analysis": {
            "example": "Массовый анализ позволяет проверить сразу несколько URL. Пример результата: {'analyzed_urls': 50, 'average_score': 85, 'improvement_opportunities': 15}",
            "image_url": "bulk_analysis_example.png",
        },
        "competitor_analysis": {
            "example": "Анализ конкурентов показывает, как ваш контент выглядит по сравнению с топ-10 результатами поиска. Пример результата: {'competitiveness': 0.75, 'content_gap': ['topic A', 'topic B'], 'strength_areas': ['topic C']}",
            "image_url": "competitor_analysis_example.png",
        },
        "llm_integration": {
            "example": "Интеграция с LLM использует передовые языковые модели для создания и оптимизации контента. Пример запроса: 'Оптимизировать для цитируемости в LLM' -> Result: {'llm_citability_score': 0.92, 'enhancement_suggestions': ['A', 'B', 'C']}",
            "image_url": "llm_integration_example.png",
        },
        "api_access": {
            "example": "API доступ позволяет интегрировать анализ в ваши системы. Пример запроса: 'POST /api/analyze' -> Result: JSON с результатами анализа",
            "image_url": "api_access_example.png",
        },
        "custom_metrics": {
            "example": "Настраиваемые метрики позволяют создавать собственные показатели для анализа. Пример: Создание метрики 'brand_consistency' для отслеживания согласованности бренда в контенте",
            "image_url": "custom_metrics_example.png",
        },
        "dedicated_support": {
            "example": "Выделенная поддержка предоставляет персонального менеджера и приоритетное решение проблем. Время ответа: < 2 часа",
            "image_url": "dedicated_support_example.png",
        },
    }

    def __init__(self, current_plan: Union[str, FreemiumPlan], user_id: Optional[str] = None):
        """
        Инициализирует ValueDemonstrator.

        Args:
            current_plan: Текущий план пользователя
            user_id: Идентификатор пользователя
        """
        self.current_plan = (
            current_plan if isinstance(current_plan, FreemiumPlan) else FreemiumPlan(current_plan)
        )
        self.user_id = user_id or "anonymous"

    def demonstrate_feature(self, feature_name: str) -> Dict[str, Any]:
        """
        Демонстрирует функцию.

        Args:
            feature_name: Имя функции

        Returns:
            Информация о функции и ее демонстрация
        """
        if feature_name not in self.FEATURES:
            return {"status": "error", "message": f"Feature {feature_name} not found"}

        feature_info = self.FEATURES[feature_name]
        feature_example = self.FEATURE_EXAMPLES.get(feature_name, {})

        # Проверяем, доступна ли функция в текущем плане
        available_in_current_plan = feature_info.get(self.current_plan, False)

        # Определяем, в каких планах доступна функция
        available_in_plans = [
            plan.value
            for plan in feature_info
            if isinstance(plan, FreemiumPlan) and feature_info[plan]
        ]

        # Если функция недоступна в текущем плане, находим минимальный план, в котором она доступна
        min_required_plan = None
        if not available_in_current_plan:
            for plan in [
                FreemiumPlan.MICRO,
                FreemiumPlan.BASIC,
                FreemiumPlan.PROFESSIONAL,
                FreemiumPlan.ENTERPRISE,
            ]:
                if feature_info.get(plan, False):
                    min_required_plan = plan.value
                    break

        return {
            "feature": feature_name,
            "description": feature_info.get("description", ""),
            "available_in_current_plan": available_in_current_plan,
            "available_in_plan": (
                available_in_plans[0] if available_in_plans else None
            ),  # Для совместимости с тестами
            "available_in_plans": available_in_plans,
            "min_required_plan": min_required_plan,
            "example": feature_example.get("example", ""),
            "image_url": feature_example.get("image_url", ""),
        }

    def demonstrate_all_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Демонстрирует все функции.

        Returns:
            Информация о всех функциях и их демонстрации
        """
        result = {}

        for feature_name in self.FEATURES:
            demo = self.demonstrate_feature(feature_name)
            result[feature_name] = demo

        return result

    def demonstrate_premium_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Демонстрирует премиум-функции, недоступные в текущем плане.

        Returns:
            Информация о премиум-функциях и их демонстрации
        """
        result = {}

        for feature_name, feature_info in self.FEATURES.items():
            # Проверяем, доступна ли функция в текущем плане
            available_in_current_plan = feature_info.get(self.current_plan, False)

            if not available_in_current_plan:
                demo = self.demonstrate_feature(feature_name)
                result[feature_name] = demo

        return result

    def get_comparison_table(self) -> Dict[str, Any]:
        """
        Возвращает таблицу сравнения планов.

        Returns:
            Таблица сравнения планов
        """
        plans = [
            FreemiumPlan.FREE,
            FreemiumPlan.MICRO,
            FreemiumPlan.BASIC,
            FreemiumPlan.PROFESSIONAL,
            FreemiumPlan.ENTERPRISE,
        ]

        # Создаем таблицу сравнения
        table = {"headers": ["Feature"] + [plan.value for plan in plans], "rows": []}

        # Заполняем таблицу
        for feature_name, feature_info in self.FEATURES.items():
            row = [feature_info.get("description", feature_name)]

            for plan in plans:
                row.append("✓" if feature_info.get(plan, False) else "-")

            table["rows"].append(row)

        return table

    def get_roi_calculator(self, plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Возвращает калькулятор ROI для указанного плана.

        Args:
            plan: План для расчета ROI

        Returns:
            Калькулятор ROI
        """
        plan = plan if isinstance(plan, FreemiumPlan) else FreemiumPlan(plan)

        # Примерные значения для расчета ROI
        roi_data = {
            FreemiumPlan.MICRO: {
                "monthly_cost": 1990,
                "estimated_time_saving": 10,  # часов в месяц
                "estimated_ranking_improvement": "10-15%",
                "estimated_traffic_improvement": "15-20%",
                "estimated_roi": "300%",
            },
            FreemiumPlan.BASIC: {
                "monthly_cost": 5990,
                "estimated_time_saving": 30,  # часов в месяц
                "estimated_ranking_improvement": "15-25%",
                "estimated_traffic_improvement": "25-40%",
                "estimated_roi": "400%",
            },
            FreemiumPlan.PROFESSIONAL: {
                "monthly_cost": 15990,
                "estimated_time_saving": 60,  # часов в месяц
                "estimated_ranking_improvement": "25-40%",
                "estimated_traffic_improvement": "40-60%",
                "estimated_roi": "500%",
            },
            FreemiumPlan.ENTERPRISE: {
                "monthly_cost": 49990,
                "estimated_time_saving": 120,  # часов в месяц
                "estimated_ranking_improvement": "40-60%",
                "estimated_traffic_improvement": "60-100%",
                "estimated_roi": "600%",
            },
        }

        if plan not in roi_data:
            return {"status": "error", "message": f"ROI data for plan {plan.value} not found"}

        return {
            "plan": plan.value,
            "roi_data": roi_data[plan],
            "calculator": {
                "hourly_rate": 1000,  # стоимость часа работы, рублей
                "monthly_traffic": 10000,  # месячный трафик
                "conversion_rate": 2,  # процент конверсии
                "average_order_value": 3000,  # средний чек, рублей
            },
        }

    def get_available_features(
        self, plan: Optional[Union[str, FreemiumPlan]] = None
    ) -> Dict[str, bool]:
        """
        Возвращает список доступных функций для указанного плана.

        Args:
            plan: План, для которого нужно получить доступные функции (если не указан, используется текущий план)

        Returns:
            Словарь с доступностью функций
        """
        if plan is None:
            plan = self.current_plan
        elif not isinstance(plan, FreemiumPlan):
            plan = FreemiumPlan(plan)

        available_features = {}

        for feature_name, feature_info in self.FEATURES.items():
            available_features[feature_name] = feature_info.get(plan, False)

        return available_features
