"""
Советник по оптимизации затрат.

Модуль предоставляет функциональность для оптимизации затрат
на использование SEO AI Models.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan
from seo_ai_models.models.tiered_system.scheduling.credit_manager import CreditManager, CreditType


class OptimizationStrategy(Enum):
    """Стратегии оптимизации затрат."""

    MINIMIZE_COST = "minimize_cost"  # Минимизация стоимости
    MAXIMIZE_PERFORMANCE = "maximize_performance"  # Максимизация производительности
    BALANCED = "balanced"  # Сбалансированный подход


class CostOptimizationAdvisor:
    """
    Советник по оптимизации затрат.

    Класс предоставляет рекомендации по оптимизации затрат
    на использование SEO AI Models.
    """

    # Стоимость планов в условных единицах
    PLAN_COSTS = {
        "micro": 10,
        "basic": 50,
        "professional": 200,
        "enterprise": 500,
    }

    # Оценка эффективности планов (0-1)
    PLAN_EFFICIENCY = {
        "micro": 0.3,
        "basic": 0.6,
        "professional": 0.9,
        "enterprise": 1.0,
    }

    def __init__(
        self,
        tier: TierPlan,
        credit_manager: Optional[CreditManager] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        **kwargs,
    ):
        """
        Инициализирует советника по оптимизации затрат.

        Args:
            tier: Текущий план использования
            credit_manager: Менеджер кредитов
            strategy: Стратегия оптимизации
            **kwargs: Дополнительные параметры
        """
        self.logger = logging.getLogger(__name__)
        self.tier = tier
        self.credit_manager = credit_manager
        self.strategy = strategy

        # Коэффициенты для стратегий
        self.strategy_coefficients = {
            OptimizationStrategy.MINIMIZE_COST: {"cost": 0.8, "performance": 0.2},
            OptimizationStrategy.MAXIMIZE_PERFORMANCE: {"cost": 0.2, "performance": 0.8},
            OptimizationStrategy.BALANCED: {"cost": 0.5, "performance": 0.5},
        }

        # История использования
        self.usage_history = []

        # Максимальное количество записей в истории
        self.max_history_records = kwargs.get("max_history_records", 30)

        self.logger.info(
            f"CostOptimizationAdvisor инициализирован для плана {tier.value} "
            f"со стратегией {strategy.value}"
        )

    def add_usage_record(self, record: Dict[str, Any]) -> None:
        """
        Добавляет запись об использовании.

        Args:
            record: Запись об использовании
        """
        # Добавляем временную метку, если её нет
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        # Добавляем запись в историю
        self.usage_history.append(record)

        # Ограничиваем размер истории
        if len(self.usage_history) > self.max_history_records:
            self.usage_history = self.usage_history[-self.max_history_records :]

    def get_optimization_recommendations(
        self, usage_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Возвращает рекомендации по оптимизации затрат.

        Args:
            usage_data: Данные об использовании

        Returns:
            Рекомендации по оптимизации
        """
        # Если данные не переданы, используем историю использования
        if not usage_data and self.credit_manager:
            usage_data = self._get_usage_data_from_credit_manager()

        if not usage_data and not self.usage_history:
            return {
                "message": "Недостаточно данных для рекомендаций",
                "recommendations": [],
            }

        if not usage_data:
            usage_data = self._aggregate_usage_history()

        # Анализируем использование и генерируем рекомендации
        recommendations = []

        # Рекомендации по оптимизации плана
        plan_recommendation = self._recommend_plan(usage_data)
        if plan_recommendation:
            recommendations.append(plan_recommendation)

        # Рекомендации по использованию функций
        feature_recommendations = self._recommend_features(usage_data)
        recommendations.extend(feature_recommendations)

        # Рекомендации по управлению ресурсами
        resource_recommendations = self._recommend_resources(usage_data)
        recommendations.extend(resource_recommendations)

        # Рекомендации по распределению нагрузки
        scheduling_recommendations = self._recommend_scheduling(usage_data)
        recommendations.extend(scheduling_recommendations)

        # Оцениваем потенциальную экономию
        savings_estimate = self._estimate_savings(recommendations)

        return {
            "message": "Рекомендации по оптимизации затрат",
            "recommendations": recommendations,
            "strategy": self.strategy.value,
            "estimated_savings": savings_estimate,
            "current_tier": self.tier.value,
        }

    def _get_usage_data_from_credit_manager(self) -> Dict[str, Any]:
        """
        Получает данные об использовании из менеджера кредитов.

        Returns:
            Данные об использовании
        """
        if not self.credit_manager:
            return {}

        credits_info = self.credit_manager.get_credits_info()

        usage_data = {
            "tier": credits_info.get("tier"),
            "credits": {},
        }

        # Преобразуем данные о кредитах
        for credit_type, credit_info in credits_info.get("credits", {}).items():
            usage_data["credits"][credit_type] = {
                "limit": credit_info.get("limit"),
                "used": credit_info.get("used"),
                "usage_percent": credit_info.get("usage_percent"),
            }

        # Получаем историю операций
        operations_history = self.credit_manager.get_operations_history(
            from_date=datetime.now() - timedelta(days=7)
        )

        # Группируем операции по типам
        operations_by_type = {}
        for record in operations_history:
            operation = record.get("operation")
            if operation not in operations_by_type:
                operations_by_type[operation] = 0

            operations_by_type[operation] += 1

        usage_data["operations"] = operations_by_type

        return usage_data

    def _aggregate_usage_history(self) -> Dict[str, Any]:
        """
        Агрегирует историю использования.

        Returns:
            Агрегированные данные об использовании
        """
        if not self.usage_history:
            return {}

        # Группируем записи по дате
        usage_by_date = {}
        for record in self.usage_history:
            timestamp = record.get("timestamp")
            date = timestamp.split("T")[0] if timestamp else None

            if not date:
                continue

            if date not in usage_by_date:
                usage_by_date[date] = []

            usage_by_date[date].append(record)

        # Агрегируем данные по дням
        daily_usage = {}
        for date, records in usage_by_date.items():
            daily_usage[date] = {
                "operations_count": len(records),
                "operations_by_type": {},
                "credits_used": {},
            }

            for record in records:
                operation = record.get("operation")
                if operation:
                    if operation not in daily_usage[date]["operations_by_type"]:
                        daily_usage[date]["operations_by_type"][operation] = 0

                    daily_usage[date]["operations_by_type"][operation] += 1

                credits = record.get("credits_used", {})
                for credit_type, amount in credits.items():
                    if credit_type not in daily_usage[date]["credits_used"]:
                        daily_usage[date]["credits_used"][credit_type] = 0

                    daily_usage[date]["credits_used"][credit_type] += amount

        return {
            "tier": self.tier.value,
            "daily_usage": daily_usage,
        }

    def _recommend_plan(self, usage_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Генерирует рекомендацию по оптимизации плана.

        Args:
            usage_data: Данные об использовании

        Returns:
            Рекомендация по оптимизации плана
        """
        current_tier = usage_data.get("tier") or self.tier.value

        # Проверяем использование кредитов
        credits = usage_data.get("credits", {})

        # Определяем максимальное использование кредитов в процентах
        max_usage_percent = 0
        for credit_type, credit_info in credits.items():
            usage_percent = credit_info.get("usage_percent")
            if usage_percent is not None and usage_percent > max_usage_percent:
                max_usage_percent = usage_percent

        # Определяем рекомендуемый план на основе использования
        recommended_tier = None

        if max_usage_percent >= 95:
            # Если использование близко к лимиту, рекомендуем более высокий план
            if current_tier == "micro":
                recommended_tier = "basic"
            elif current_tier == "basic":
                recommended_tier = "professional"
            elif current_tier == "professional":
                recommended_tier = "enterprise"

        elif max_usage_percent <= 30:
            # Если использование низкое, рекомендуем более низкий план
            if current_tier == "enterprise":
                recommended_tier = "professional"
            elif current_tier == "professional":
                recommended_tier = "basic"
            elif current_tier == "basic":
                recommended_tier = "micro"

        # Если рекомендуемый план совпадает с текущим, не возвращаем рекомендацию
        if not recommended_tier or recommended_tier == current_tier:
            return None

        # Учитываем стратегию оптимизации
        if self.strategy == OptimizationStrategy.MINIMIZE_COST:
            # Если стратегия - минимизация стоимости, рекомендуем более низкий план
            if recommended_tier in ["professional", "enterprise"] and max_usage_percent < 90:
                if recommended_tier == "enterprise":
                    recommended_tier = "professional"
                elif recommended_tier == "professional":
                    recommended_tier = "basic"

        elif self.strategy == OptimizationStrategy.MAXIMIZE_PERFORMANCE:
            # Если стратегия - максимизация производительности, рекомендуем более высокий план
            if recommended_tier in ["micro", "basic"] and max_usage_percent > 50:
                if recommended_tier == "micro":
                    recommended_tier = "basic"
                elif recommended_tier == "basic":
                    recommended_tier = "professional"

        # Формируем рекомендацию
        cost_diff = self.PLAN_COSTS[recommended_tier] - self.PLAN_COSTS[current_tier]
        efficiency_diff = (
            self.PLAN_EFFICIENCY[recommended_tier] - self.PLAN_EFFICIENCY[current_tier]
        )

        recommendation = {
            "type": "plan",
            "title": f"Перейти на план '{recommended_tier}'",
            "current": current_tier,
            "recommended": recommended_tier,
            "description": "",
            "cost_impact": cost_diff,
            "efficiency_impact": efficiency_diff,
        }

        if cost_diff > 0:
            recommendation["description"] = (
                f"Рекомендуется перейти на более высокий план '{recommended_tier}' "
                f"для повышения производительности и расширения возможностей. "
                f"Текущее использование ресурсов ({max_usage_percent:.1f}%) близко к лимиту."
            )
        else:
            recommendation["description"] = (
                f"Рекомендуется перейти на более низкий план '{recommended_tier}' "
                f"для оптимизации затрат. Текущее использование ресурсов ({max_usage_percent:.1f}%) "
                f"значительно ниже лимита, что указывает на избыточность текущего плана."
            )

        return recommendation

    def _recommend_features(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации по использованию функций.

        Args:
            usage_data: Данные об использовании

        Returns:
            Рекомендации по использованию функций
        """
        recommendations = []

        # Анализируем использование операций
        operations = usage_data.get("operations", {})

        # Проверяем использование LLM-анализа
        llm_operations = ["analyze_llm_compatibility", "analyze_citability"]
        llm_usage = sum(operations.get(op, 0) for op in llm_operations)

        if llm_usage > 50 and self.tier.value != "enterprise":
            recommendations.append(
                {
                    "type": "feature",
                    "title": "Оптимизировать использование LLM-анализа",
                    "description": (
                        f"Высокое использование LLM-анализа ({llm_usage} операций). "
                        f"Рекомендуется оптимизировать использование LLM-функций, "
                        f"выполняя анализ только для важных страниц, или "
                        f"рассмотреть переход на более высокий план."
                    ),
                    "feature": "llm_analysis",
                    "impact": "high",
                }
            )

        # Проверяем использование массового анализа
        bulk_usage = operations.get("bulk_analyze", 0)

        if bulk_usage > 10 and self.tier.value == "basic":
            recommendations.append(
                {
                    "type": "feature",
                    "title": "Оптимизировать использование массового анализа",
                    "description": (
                        f"Высокое использование массового анализа ({bulk_usage} операций). "
                        f"Рекомендуется перейти на план 'professional' для более "
                        f"эффективного выполнения массового анализа."
                    ),
                    "feature": "bulk_analyze",
                    "impact": "medium",
                }
            )

        return recommendations

    def _recommend_resources(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации по управлению ресурсами.

        Args:
            usage_data: Данные об использовании

        Returns:
            Рекомендации по управлению ресурсами
        """
        recommendations = []

        # Анализируем использование кредитов
        credits = usage_data.get("credits", {})

        # Проверяем неравномерное использование кредитов
        if "analysis" in credits and "keyword" in credits:
            analysis_percent = credits["analysis"].get("usage_percent", 0)
            keyword_percent = credits["keyword"].get("usage_percent", 0)

            if abs(analysis_percent - keyword_percent) > 30:
                # Если разница в использовании более 30%, рекомендуем балансировку
                if analysis_percent > keyword_percent:
                    recommendations.append(
                        {
                            "type": "resource",
                            "title": "Сбалансировать использование кредитов",
                            "description": (
                                f"Неравномерное использование кредитов: анализ ({analysis_percent:.1f}%), "
                                f"ключевые слова ({keyword_percent:.1f}%). Рекомендуется более "
                                f"активно использовать анализ ключевых слов для оптимизации контента."
                            ),
                            "impact": "medium",
                        }
                    )
                else:
                    recommendations.append(
                        {
                            "type": "resource",
                            "title": "Сбалансировать использование кредитов",
                            "description": (
                                f"Неравномерное использование кредитов: ключевые слова ({keyword_percent:.1f}%), "
                                f"анализ ({analysis_percent:.1f}%). Рекомендуется более "
                                f"активно использовать базовый анализ для оптимизации контента."
                            ),
                            "impact": "medium",
                        }
                    )

        return recommendations

    def _recommend_scheduling(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации по распределению нагрузки.

        Args:
            usage_data: Данные об использовании

        Returns:
            Рекомендации по распределению нагрузки
        """
        recommendations = []

        # Анализируем ежедневное использование
        daily_usage = usage_data.get("daily_usage", {})

        if not daily_usage:
            return recommendations

        # Вычисляем среднее количество операций в день
        daily_operations = [data.get("operations_count", 0) for data in daily_usage.values()]
        avg_operations = sum(daily_operations) / len(daily_operations) if daily_operations else 0

        # Проверяем наличие дней с пиковой нагрузкой
        peak_days = []
        for date, data in daily_usage.items():
            operations_count = data.get("operations_count", 0)
            if operations_count > avg_operations * 1.5:
                peak_days.append((date, operations_count))

        if peak_days:
            recommendations.append(
                {
                    "type": "scheduling",
                    "title": "Распределить нагрузку равномерно",
                    "description": (
                        f"Обнаружены дни с пиковой нагрузкой: "
                        f"{', '.join(f'{date} ({count} операций)' for date, count in peak_days[:3])}. "
                        f"Рекомендуется распределить нагрузку более равномерно "
                        f"для оптимального использования ресурсов."
                    ),
                    "impact": "medium",
                }
            )

        return recommendations

    def _estimate_savings(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценивает потенциальную экономию от рекомендаций.

        Args:
            recommendations: Рекомендации

        Returns:
            Оценка потенциальной экономии
        """
        cost_savings = 0
        efficiency_impact = 0

        for recommendation in recommendations:
            if recommendation["type"] == "plan":
                cost_savings += -recommendation.get("cost_impact", 0)
                efficiency_impact += recommendation.get("efficiency_impact", 0)
            elif recommendation["type"] == "feature":
                impact_map = {"high": 15, "medium": 10, "low": 5}
                impact = recommendation.get("impact", "low")
                cost_savings += impact_map.get(impact, 0)
            elif recommendation["type"] == "resource":
                impact_map = {"high": 10, "medium": 7, "low": 3}
                impact = recommendation.get("impact", "low")
                cost_savings += impact_map.get(impact, 0)
            elif recommendation["type"] == "scheduling":
                impact_map = {"high": 8, "medium": 5, "low": 2}
                impact = recommendation.get("impact", "low")
                cost_savings += impact_map.get(impact, 0)

        # Применяем коэффициенты стратегии
        coefficients = self.strategy_coefficients.get(
            self.strategy, {"cost": 0.5, "performance": 0.5}
        )

        # Конвертируем экономию в проценты от текущей стоимости
        current_cost = self.PLAN_COSTS.get(self.tier.value, 50)
        cost_percent = (cost_savings / current_cost) * 100 if current_cost > 0 else 0

        return {
            "cost_savings": round(cost_savings, 2),
            "cost_savings_percent": round(cost_percent, 2),
            "efficiency_impact": round(efficiency_impact, 2),
            "overall_score": round(
                coefficients["cost"] * cost_percent
                + coefficients["performance"] * efficiency_impact * 100,
                2,
            ),
        }

    def set_strategy(self, strategy: OptimizationStrategy) -> None:
        """
        Устанавливает стратегию оптимизации.

        Args:
            strategy: Стратегия оптимизации
        """
        self.strategy = strategy
        self.logger.info(f"Установлена стратегия оптимизации: {strategy.value}")

    def update_tier(self, new_tier: TierPlan) -> None:
        """
        Обновляет план использования.

        Args:
            new_tier: Новый план
        """
        self.tier = new_tier
        self.logger.info(f"Обновлен план использования: {new_tier.value}")
