"""
Планировщик бюджета для многоуровневой системы SEO AI Models.

Этот модуль предоставляет функциональность для планирования и управления бюджетом
для разных уровней системы SEO AI Models, помогая оптимально распределять кредиты.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

from seo_ai_models.models.tiered_system.scheduling.credit_manager import CreditManager, CreditType
from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan

logger = logging.getLogger(__name__)


class BudgetPlanner:
    """
    Класс планировщика бюджета, который помогает управлять и оптимизировать
    распределение кредитов для разных уровней системы SEO AI Models.

    Этот класс работает совместно с CreditManager для помощи пользователям
    в планировании использования кредитов и оптимизации распределения бюджета.
    """

    # Коэффициенты стоимости по умолчанию для разных операций
    DEFAULT_COST_COEFFICIENTS = {
        "content_analysis": {
            CreditType.ANALYSIS.value: 1.0,
            CreditType.LLM.value: 0.0,
            CreditType.PREMIUM.value: 0.0,
        },
        "keyword_analysis": {
            CreditType.ANALYSIS.value: 0.5,
            CreditType.KEYWORD.value: 1.0,
            CreditType.LLM.value: 0.0,
            CreditType.PREMIUM.value: 0.0,
        },
        "structure_analysis": {
            CreditType.ANALYSIS.value: 0.5,
            CreditType.LLM.value: 0.0,
            CreditType.PREMIUM.value: 0.0,
        },
        "readability_analysis": {
            CreditType.ANALYSIS.value: 0.3,
            CreditType.LLM.value: 0.0,
            CreditType.PREMIUM.value: 0.0,
        },
        "eeat_analysis": {
            CreditType.ANALYSIS.value: 1.0,
            CreditType.LLM.value: 0.5,
            CreditType.PREMIUM.value: 0.0,
        },
        "llm_analysis": {
            CreditType.ANALYSIS.value: 0.5,
            CreditType.LLM.value: 5.0,
            CreditType.PREMIUM.value: 0.0,
        },
        "serp_analysis": {
            CreditType.ANALYSIS.value: 1.0,
            CreditType.KEYWORD.value: 1.0,
            CreditType.PREMIUM.value: 0.0,
        },
        "competitor_analysis": {
            CreditType.ANALYSIS.value: 1.5,
            CreditType.KEYWORD.value: 1.0,
            CreditType.LLM.value: 1.0,
            CreditType.PREMIUM.value: 0.0,
        },
        "premium_recommendations": {
            CreditType.ANALYSIS.value: 0.5,
            CreditType.LLM.value: 1.0,
            CreditType.PREMIUM.value: 2.0,
        },
    }

    # Распределение бюджета по умолчанию для разных уровней доступа (в процентах)
    DEFAULT_BUDGET_ALLOCATION = {
        TierPlan.MICRO.value: {
            "content_analysis": 40,
            "keyword_analysis": 30,
            "structure_analysis": 15,
            "readability_analysis": 15,
            "eeat_analysis": 0,
            "llm_analysis": 0,
            "serp_analysis": 0,
            "competitor_analysis": 0,
            "premium_recommendations": 0,
        },
        TierPlan.BASIC.value: {
            "content_analysis": 30,
            "keyword_analysis": 25,
            "structure_analysis": 15,
            "readability_analysis": 10,
            "eeat_analysis": 10,
            "llm_analysis": 0,
            "serp_analysis": 5,
            "competitor_analysis": 5,
            "premium_recommendations": 0,
        },
        TierPlan.PROFESSIONAL.value: {
            "content_analysis": 20,
            "keyword_analysis": 20,
            "structure_analysis": 10,
            "readability_analysis": 5,
            "eeat_analysis": 15,
            "llm_analysis": 15,
            "serp_analysis": 10,
            "competitor_analysis": 5,
            "premium_recommendations": 0,
        },
        TierPlan.ENTERPRISE.value: {
            "content_analysis": 15,
            "keyword_analysis": 15,
            "structure_analysis": 10,
            "readability_analysis": 5,
            "eeat_analysis": 15,
            "llm_analysis": 20,
            "serp_analysis": 10,
            "competitor_analysis": 5,
            "premium_recommendations": 5,
        },
    }

    # Лимиты дневного использования по умолчанию (в процентах от общего)
    DEFAULT_DAILY_USAGE_LIMITS = {
        TierPlan.MICRO.value: 10,  # 10% от общего кол-ва кредитов в день
        TierPlan.BASIC.value: 15,  # 15% от общего кол-ва кредитов в день
        TierPlan.PROFESSIONAL.value: 20,  # 20% от общего кол-ва кредитов в день
        TierPlan.ENTERPRISE.value: 100,  # без ограничений
    }

    def __init__(
        self,
        user_id: str,
        tier: Union[TierPlan, str] = TierPlan.MICRO,
        credit_manager: Optional[CreditManager] = None,
        cost_coefficients: Optional[Dict[str, Dict[str, float]]] = None,
        budget_allocation: Optional[Dict[str, Dict[str, int]]] = None,
        daily_usage_limits: Optional[Dict[str, int]] = None,
        data_dir: str = "data/budget_plans",
    ):
        """
        Инициализация планировщика бюджета.

        Args:
            user_id: Уникальный идентификатор пользователя
            tier: Уровень подписки пользователя
            credit_manager: Экземпляр CreditManager для операций с кредитами
            cost_coefficients: Пользовательские коэффициенты стоимости для разных операций
            budget_allocation: Пользовательское распределение бюджета для разных уровней
            daily_usage_limits: Лимиты дневного использования для разных уровней
            data_dir: Директория для хранения планов бюджета
        """
        self.user_id = user_id

        # Преобразование tier в строку, если это enum
        if isinstance(tier, TierPlan):
            self.tier = tier.value
        else:
            self.tier = tier.lower()

        self.credit_manager = credit_manager or CreditManager(user_id)
        self.cost_coefficients = cost_coefficients or self.DEFAULT_COST_COEFFICIENTS
        self.budget_allocation = budget_allocation or self.DEFAULT_BUDGET_ALLOCATION
        self.daily_usage_limits = daily_usage_limits or self.DEFAULT_DAILY_USAGE_LIMITS

        # Создание директории данных, если она не существует
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Файл плана бюджета конкретного пользователя
        self.budget_plan_file = os.path.join(self.data_dir, f"{user_id}_budget_plan.json")

        # Загрузка существующего плана бюджета или создание нового
        self.budget_plan = self._load_budget_plan()

        # Отслеживание дневного использования
        self.daily_usage = self._initialize_daily_usage()

    def _initialize_daily_usage(self) -> Dict[str, float]:
        """Инициализация отслеживания дневного использования."""
        daily_usage = {}
        for credit_type in CreditType:
            daily_usage[credit_type.value] = 0.0
        return daily_usage

    def _load_budget_plan(self) -> Dict:
        """Загрузка существующего плана бюджета или создание нового."""
        if os.path.exists(self.budget_plan_file):
            try:
                with open(self.budget_plan_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Ошибка загрузки плана бюджета: {e}")
                return self._create_default_budget_plan()
        else:
            return self._create_default_budget_plan()

    def _create_default_budget_plan(self) -> Dict:
        """Создание плана бюджета по умолчанию на основе уровня пользователя."""
        # Получение доступных кредитов для каждого типа
        available_credits = {}
        total_credits = 0

        for credit_type in CreditType:
            credits = self.credit_manager.get_credits(credit_type)
            available_credits[credit_type.value] = credits
            total_credits += credits

        # Получение распределения по умолчанию для уровня пользователя
        tier_allocation = self.budget_allocation.get(
            self.tier, self.budget_allocation[TierPlan.MICRO.value]
        )

        # Расчет распределения кредитов для разных операций
        operation_allocations = {}
        for operation, percentage in tier_allocation.items():
            operation_allocations[operation] = {
                credit_type.value: (percentage / 100) * available_credits.get(credit_type.value, 0)
                for credit_type in CreditType
            }

        # Расчет дневного лимита использования
        daily_limit_percentage = self.daily_usage_limits.get(self.tier, 10)
        daily_limits = {
            credit_type.value: (daily_limit_percentage / 100)
            * available_credits.get(credit_type.value, 0)
            for credit_type in CreditType
        }

        current_date = datetime.now().strftime("%Y-%m-%d")

        return {
            "user_id": self.user_id,
            "tier": self.tier,
            "created_at": current_date,
            "updated_at": current_date,
            "available_credits": available_credits,
            "total_credits": total_credits,
            "operation_allocations": operation_allocations,
            "daily_limits": daily_limits,
            "usage_history": {},
            "forecasted_needs": {},
            "optimization_suggestions": [],
        }

    def save_budget_plan(self) -> bool:
        """Сохранение текущего плана бюджета в файл."""
        try:
            with open(self.budget_plan_file, "w") as f:
                json.dump(self.budget_plan, f, indent=4, ensure_ascii=False)
            return True
        except IOError as e:
            logger.error(f"Ошибка сохранения плана бюджета: {e}")
            return False

    def get_allocation_for_operation(
        self, operation: str, credit_type: Union[CreditType, str] = None
    ) -> Dict[str, float]:
        """
        Получение распределения кредитов для конкретной операции.

        Args:
            operation: Операция, для которой запрашивается распределение
            credit_type: Тип кредита (если указан, возвращается только значение для этого типа)

        Returns:
            Dict[str, float] или float: Распределение кредитов для операции
        """
        if operation not in self.budget_plan["operation_allocations"]:
            if credit_type:
                return 0.0
            return {credit_type.value: 0.0 for credit_type in CreditType}

        allocation = self.budget_plan["operation_allocations"][operation]

        if credit_type:
            # Преобразование в строку, если это enum
            if isinstance(credit_type, CreditType):
                credit_type = credit_type.value

            return allocation.get(credit_type, 0.0)

        return allocation

    def update_budget_allocation(self, new_allocations: Dict[str, Dict[str, float]]) -> bool:
        """
        Обновление распределения бюджета для разных операций.

        Args:
            new_allocations: Словарь, связывающий операции с распределением кредитов по типам

        Returns:
            bool: True, если обновление успешно, False в противном случае
        """
        # Проверка, что общая сумма не превышает доступные кредиты
        for credit_type in CreditType:
            type_value = credit_type.value
            total_for_type = sum(
                allocation.get(type_value, 0.0) for allocation in new_allocations.values()
            )
            available = self.budget_plan["available_credits"].get(type_value, 0.0)

            if total_for_type > available:
                logger.error(f"Общее распределение для {type_value} превышает доступные кредиты")
                return False

        # Обновление распределения
        self.budget_plan["operation_allocations"] = new_allocations
        self.budget_plan["updated_at"] = datetime.now().strftime("%Y-%m-%d")

        return self.save_budget_plan()

    def record_operation_usage(self, operation: str, credit_usage: Dict[str, float]) -> bool:
        """
        Запись использования кредитов для операции.

        Args:
            operation: Операция, для которой были использованы кредиты
            credit_usage: Словарь использования кредитов по типам

        Returns:
            bool: True, если запись успешна, False в противном случае
        """
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Инициализация истории использования для текущей даты, если ее нет
        if current_date not in self.budget_plan["usage_history"]:
            self.budget_plan["usage_history"][current_date] = {}

        if operation not in self.budget_plan["usage_history"][current_date]:
            self.budget_plan["usage_history"][current_date][operation] = {
                credit_type.value: 0.0 for credit_type in CreditType
            }

        # Обновление истории использования
        for credit_type, amount in credit_usage.items():
            if credit_type in self.budget_plan["usage_history"][current_date][operation]:
                self.budget_plan["usage_history"][current_date][operation][credit_type] += amount
                # Обновление дневного использования
                self.daily_usage[credit_type] += amount

        # Проверка дневных лимитов
        for credit_type, usage in self.daily_usage.items():
            if credit_type in self.budget_plan["daily_limits"]:
                limit = self.budget_plan["daily_limits"][credit_type]
                if usage > limit:
                    logger.warning(f"Превышен дневной лимит для {credit_type}: {usage}/{limit}")

        # Сохранение обновленного плана
        return self.save_budget_plan()

    def check_operation_availability(
        self, operation: str, required_credits: Dict[str, float]
    ) -> bool:
        """
        Проверка доступности операции на основе требуемых кредитов и текущих лимитов.

        Args:
            operation: Операция для проверки
            required_credits: Требуемые кредиты по типам

        Returns:
            bool: True, если операция доступна, False в противном случае
        """
        # Проверка доступности кредитов для каждого типа
        for credit_type, amount in required_credits.items():
            # Проверка дневного лимита
            if credit_type in self.daily_usage and credit_type in self.budget_plan["daily_limits"]:
                current_usage = self.daily_usage[credit_type]
                daily_limit = self.budget_plan["daily_limits"][credit_type]

                if current_usage + amount > daily_limit:
                    logger.warning(f"Операция {operation} превысит дневной лимит для {credit_type}")
                    return False

            # Проверка общего распределения для операции
            allocation = self.get_allocation_for_operation(operation, credit_type)
            current_usage = self._get_current_operation_usage(operation, credit_type)

            if current_usage + amount > allocation:
                logger.warning(f"Операция {operation} превысит выделенный бюджет для {credit_type}")
                return False

            # Проверка наличия кредитов у пользователя
            if not self.credit_manager.has_enough_credits(CreditType(credit_type), amount):
                logger.warning(f"Недостаточно кредитов типа {credit_type} для операции {operation}")
                return False

        return True

    def _get_current_operation_usage(self, operation: str, credit_type: str) -> float:
        """Получение текущего использования для операции и типа кредита."""
        current_date = datetime.now().strftime("%Y-%m-%d")

        if (
            current_date in self.budget_plan["usage_history"]
            and operation in self.budget_plan["usage_history"][current_date]
            and credit_type in self.budget_plan["usage_history"][current_date][operation]
        ):
            return self.budget_plan["usage_history"][current_date][operation][credit_type]

        return 0.0

    def calculate_operation_cost(self, operation: str, content_size: int = 1) -> Dict[str, float]:
        """
        Расчет стоимости операции на основе размера контента и коэффициентов стоимости.

        Args:
            operation: Операция для расчета стоимости
            content_size: Размер контента (например, количество URL или слов)

        Returns:
            Dict[str, float]: Стоимость операции по типам кредитов
        """
        if operation not in self.cost_coefficients:
            logger.warning(f"Операция {operation} не найдена в коэффициентах стоимости")
            return {credit_type.value: 0.0 for credit_type in CreditType}

        cost = {}
        for credit_type, coefficient in self.cost_coefficients[operation].items():
            cost[credit_type] = coefficient * content_size

        return cost

    def forecast_needs(
        self, days: int = 30, usage_pattern: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Прогнозирование потребности в кредитах на основе текущего использования.

        Args:
            days: Количество дней для прогноза
            usage_pattern: Пользовательский шаблон использования (коэффициенты по операциям)

        Returns:
            Dict[str, Any]: Прогноз потребностей в кредитах
        """
        # Расчет среднего дневного использования за последние 7 дней
        today = datetime.now().strftime("%Y-%m-%d")
        last_7_days = [
            (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)
        ]

        avg_daily_usage = {credit_type.value: 0.0 for credit_type in CreditType}
        days_with_data = 0

        for day in last_7_days:
            if day in self.budget_plan["usage_history"]:
                days_with_data += 1
                for operation_usage in self.budget_plan["usage_history"][day].values():
                    for credit_type, amount in operation_usage.items():
                        avg_daily_usage[credit_type] += amount

        # Если данных за последние дни нет, используем шаблон или возвращаем нули
        if days_with_data == 0:
            if usage_pattern:
                avg_daily_usage = usage_pattern
            else:
                avg_daily_usage = {credit_type.value: 0.0 for credit_type in CreditType}
        else:
            # Расчет среднего
            for credit_type in avg_daily_usage:
                avg_daily_usage[credit_type] /= days_with_data

        # Прогноз на указанное количество дней
        forecast = {
            "days": days,
            "from_date": today,
            "to_date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
            "avg_daily_usage": avg_daily_usage,
            "projected_usage": {
                credit_type: amount * days for credit_type, amount in avg_daily_usage.items()
            },
            "available_credits": self.budget_plan["available_credits"],
            "projected_balance": {
                credit_type: self.budget_plan["available_credits"].get(credit_type, 0)
                - (amount * days)
                for credit_type, amount in avg_daily_usage.items()
            },
        }

        # Сохранение прогноза в план бюджета
        self.budget_plan["forecasted_needs"] = forecast
        self.save_budget_plan()

        return forecast

    def generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Генерация предложений по оптимизации использования кредитов.

        Returns:
            List[Dict[str, Any]]: Список предложений по оптимизации
        """
        suggestions = []

        # Получение прогноза потребностей
        forecast = self.budget_plan.get("forecasted_needs", {})
        if not forecast:
            forecast = self.forecast_needs()

        # Проверка на потенциальное исчерпание кредитов
        projected_balance = forecast.get("projected_balance", {})
        for credit_type, balance in projected_balance.items():
            if balance < 0:
                days_to_zero = forecast["available_credits"].get(credit_type, 0) / forecast[
                    "avg_daily_usage"
                ].get(credit_type, 1)

                suggestions.append(
                    {
                        "type": "credit_depletion",
                        "credit_type": credit_type,
                        "days_to_zero": round(days_to_zero),
                        "current_balance": self.budget_plan["available_credits"].get(
                            credit_type, 0
                        ),
                        "suggestion": f"Кредиты типа {credit_type} будут исчерпаны через {round(days_to_zero)} дней при текущем использовании. Рекомендуется обновить план или перераспределить использование.",
                    }
                )

        # Анализ эффективности использования по операциям
        usage_history = self.budget_plan.get("usage_history", {})
        operations_usage = {}

        # Сбор статистики использования по операциям
        for day, operations in usage_history.items():
            for operation, credits in operations.items():
                if operation not in operations_usage:
                    operations_usage[operation] = {credit_type: 0.0 for credit_type in credits}

                for credit_type, amount in credits.items():
                    operations_usage[operation][credit_type] += amount

        # Поиск неэффективного распределения
        for operation, credits in operations_usage.items():
            allocations = self.get_allocation_for_operation(operation)

            for credit_type, amount in credits.items():
                allocation = allocations.get(credit_type, 0.0)

                # Если использование значительно ниже выделенного бюджета
                if allocation > 0 and amount / allocation < 0.5:
                    suggestions.append(
                        {
                            "type": "underutilization",
                            "operation": operation,
                            "credit_type": credit_type,
                            "allocated": allocation,
                            "used": amount,
                            "utilization_ratio": amount / allocation if allocation > 0 else 0,
                            "suggestion": f"Неэффективное использование кредитов {credit_type} для операции {operation}. Рекомендуется перераспределить бюджет.",
                        }
                    )

                # Если использование близко к лимиту
                if allocation > 0 and amount / allocation > 0.9:
                    suggestions.append(
                        {
                            "type": "near_limit",
                            "operation": operation,
                            "credit_type": credit_type,
                            "allocated": allocation,
                            "used": amount,
                            "utilization_ratio": amount / allocation,
                            "suggestion": f"Использование кредитов {credit_type} для операции {operation} близко к лимиту. Рекомендуется увеличить выделенный бюджет.",
                        }
                    )

        # Сохранение сгенерированных предложений
        self.budget_plan["optimization_suggestions"] = suggestions
        self.save_budget_plan()

        return suggestions

    def reset_daily_usage(self) -> bool:
        """Сброс дневного отслеживания использования."""
        self.daily_usage = self._initialize_daily_usage()
        return True

    def update_daily_limits(self, new_limits: Dict[str, float]) -> bool:
        """
        Обновление дневных лимитов использования.

        Args:
            new_limits: Новые дневные лимиты по типам кредитов

        Returns:
            bool: True, если обновление успешно, False в противном случае
        """
        self.budget_plan["daily_limits"] = new_limits
        return self.save_budget_plan()

    def get_budget_summary(self) -> Dict[str, Any]:
        """
        Получение сводки по бюджету.

        Returns:
            Dict[str, Any]: Сводка по бюджету
        """
        # Расчет общего использования
        total_usage = {credit_type.value: 0.0 for credit_type in CreditType}

        for day, operations in self.budget_plan.get("usage_history", {}).items():
            for operation, credits in operations.items():
                for credit_type, amount in credits.items():
                    if credit_type in total_usage:
                        total_usage[credit_type] += amount

        # Формирование сводки
        summary = {
            "user_id": self.user_id,
            "tier": self.tier,
            "available_credits": self.budget_plan["available_credits"],
            "total_usage": total_usage,
            "utilization_ratio": {
                credit_type: (
                    (
                        total_usage.get(credit_type, 0.0)
                        / self.budget_plan["available_credits"].get(credit_type, 1.0)
                    )
                    if self.budget_plan["available_credits"].get(credit_type, 0.0) > 0
                    else 0.0
                )
                for credit_type in total_usage
            },
            "daily_limits": self.budget_plan["daily_limits"],
            "current_daily_usage": self.daily_usage,
            "operation_allocations": self.budget_plan["operation_allocations"],
            "optimization_suggestions": self.budget_plan.get("optimization_suggestions", []),
            "forecasted_needs": self.budget_plan.get("forecasted_needs", {}),
        }

        return summary
