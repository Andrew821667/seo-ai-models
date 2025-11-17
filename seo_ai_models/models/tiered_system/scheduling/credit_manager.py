"""
Система управления кредитами анализа.

Модуль предоставляет функциональность для управления кредитами анализа,
которые определяют доступный объем анализа для пользователя.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union

from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan


class CreditType(Enum):
    """Типы кредитов для анализа."""

    ANALYSIS = "analysis"  # Кредиты для базового анализа
    KEYWORD = "keyword"  # Кредиты для анализа ключевых слов
    LLM = "llm"  # Кредиты для LLM-анализа
    PREMIUM = "premium"  # Кредиты для премиальных функций


class CreditManager:
    """
    Система управления кредитами анализа.

    Класс отвечает за учет и распределение кредитов анализа,
    которые определяют доступный объем анализа для пользователя.
    """

    # Базовые лимиты кредитов для каждого плана
    CREDIT_LIMITS = {
        "micro": {
            CreditType.ANALYSIS: 10,  # 10 базовых анализов в день
            CreditType.KEYWORD: 5,  # 5 анализов ключевых слов в день
            CreditType.LLM: 0,  # Нет LLM-анализов
            CreditType.PREMIUM: 0,  # Нет премиальных функций
        },
        "basic": {
            CreditType.ANALYSIS: 50,  # 50 базовых анализов в день
            CreditType.KEYWORD: 30,  # 30 анализов ключевых слов в день
            CreditType.LLM: 10,  # 10 LLM-анализов в день
            CreditType.PREMIUM: 0,  # Нет премиальных функций
        },
        "professional": {
            CreditType.ANALYSIS: 200,  # 200 базовых анализов в день
            CreditType.KEYWORD: 100,  # 100 анализов ключевых слов в день
            CreditType.LLM: 50,  # 50 LLM-анализов в день
            CreditType.PREMIUM: 20,  # 20 премиальных функций в день
        },
        "enterprise": {
            CreditType.ANALYSIS: -1,  # Неограниченно
            CreditType.KEYWORD: -1,  # Неограниченно
            CreditType.LLM: 200,  # 200 LLM-анализов в день
            CreditType.PREMIUM: 100,  # 100 премиальных функций в день
        },
    }

    # Стоимость операций в кредитах
    OPERATION_COSTS = {
        "analyze_content": {
            CreditType.ANALYSIS: 1,
        },
        "analyze_keywords": {
            CreditType.KEYWORD: 1,
        },
        "analyze_llm_compatibility": {
            CreditType.LLM: 2,
        },
        "analyze_citability": {
            CreditType.LLM: 3,
        },
        "optimize_content": {
            CreditType.ANALYSIS: 2,
        },
        "generate_recommendations": {
            CreditType.ANALYSIS: 1,
        },
        "bulk_analyze": {
            CreditType.ANALYSIS: 5,
            CreditType.KEYWORD: 3,
        },
    }

    def __init__(
        self,
        tier: TierPlan,
        user_id: Optional[str] = None,
        bonus_credits: Optional[Dict[CreditType, int]] = None,
        **kwargs,
    ):
        """
        Инициализирует менеджер кредитов.

        Args:
            tier: План использования
            user_id: ID пользователя
            bonus_credits: Дополнительные кредиты
            **kwargs: Дополнительные параметры
        """
        self.logger = logging.getLogger(__name__)
        self.tier = tier
        self.user_id = user_id or "anonymous"

        # Устанавливаем лимиты кредитов в зависимости от плана
        self.credit_limits = self.CREDIT_LIMITS[tier.value].copy()

        # Добавляем бонусные кредиты, если они есть
        if bonus_credits:
            for credit_type, bonus in bonus_credits.items():
                if self.credit_limits.get(credit_type, 0) >= 0:  # Не добавляем к безлимитным (-1)
                    self.credit_limits[credit_type] = self.credit_limits.get(credit_type, 0) + bonus

        # Счетчики использованных кредитов
        self.used_credits = {credit_type: 0 for credit_type in CreditType}

        # История операций
        self.operations_history = []

        # Определяем период сброса (0:00 следующего дня)
        self.reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        # Период хранения истории операций (30 дней)
        self.history_retention_period = kwargs.get("history_retention_period", 30)

        self.logger.info(
            f"CreditManager инициализирован для пользователя {self.user_id} с планом {tier.value}"
        )

    def check_credits_available(self, operation: str, count: int = 1) -> bool:
        """
        Проверяет доступность кредитов для операции.

        Args:
            operation: Название операции
            count: Количество операций

        Returns:
            True, если кредитов достаточно
        """
        # Проверяем, нужно ли сбросить счетчики
        self._check_reset_counters()

        # Получаем стоимость операции
        operation_costs = self.OPERATION_COSTS.get(operation, {})

        # Проверяем достаточность кредитов для каждого типа
        for credit_type, cost in operation_costs.items():
            limit = self.credit_limits.get(credit_type, 0)

            # Если лимит -1, значит неограниченно
            if limit == -1:
                continue

            used = self.used_credits.get(credit_type, 0)
            required = cost * count

            if used + required > limit:
                return False

        return True

    def use_credits(
        self, operation: str, count: int = 1, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Использует кредиты для операции.

        Args:
            operation: Название операции
            count: Количество операций
            metadata: Дополнительные данные об операции

        Returns:
            True, если кредиты успешно использованы
        """
        # Проверяем доступность кредитов
        if not self.check_credits_available(operation, count):
            return False

        # Проверяем, нужно ли сбросить счетчики
        self._check_reset_counters()

        # Получаем стоимость операции
        operation_costs = self.OPERATION_COSTS.get(operation, {})

        # Используем кредиты
        for credit_type, cost in operation_costs.items():
            if self.credit_limits.get(credit_type, 0) != -1:  # Не считаем для безлимитных
                self.used_credits[credit_type] = self.used_credits.get(credit_type, 0) + (
                    cost * count
                )

        # Добавляем операцию в историю
        timestamp = datetime.now().isoformat()
        operation_record = {
            "timestamp": timestamp,
            "operation": operation,
            "count": count,
            "costs": operation_costs,
            "metadata": metadata or {},
        }
        self.operations_history.append(operation_record)

        # Очищаем старые записи в истории
        self._cleanup_history()

        return True

    def get_credits_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о кредитах.

        Returns:
            Информация о кредитах
        """
        # Проверяем, нужно ли сбросить счетчики
        self._check_reset_counters()

        credits_info = {
            "tier": self.tier.value,
            "user_id": self.user_id,
            "reset_time": self.reset_time.isoformat(),
            "credits": {},
        }

        # Формируем информацию о кредитах по типам
        for credit_type in CreditType:
            limit = self.credit_limits.get(credit_type, 0)
            used = self.used_credits.get(credit_type, 0)

            # Для безлимитных кредитов
            if limit == -1:
                remaining = None
                usage_percent = None
            else:
                remaining = max(0, limit - used)
                usage_percent = (used / limit) * 100 if limit > 0 else 0

            credits_info["credits"][credit_type.value] = {
                "limit": limit,
                "used": used,
                "remaining": remaining,
                "usage_percent": usage_percent,
            }

        return credits_info

    def get_operations_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        operation_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает историю операций.

        Args:
            from_date: Начальная дата
            to_date: Конечная дата
            operation_type: Тип операции

        Returns:
            История операций
        """
        filtered_history = []

        for record in self.operations_history:
            # Конвертируем строку timestamp в datetime
            record_time = datetime.fromisoformat(record["timestamp"])

            # Фильтрация по дате
            if from_date and record_time < from_date:
                continue
            if to_date and record_time > to_date:
                continue

            # Фильтрация по типу операции
            if operation_type and record["operation"] != operation_type:
                continue

            filtered_history.append(record)

        return filtered_history

    def add_bonus_credits(
        self, credit_type: CreditType, amount: int, reason: Optional[str] = None
    ) -> bool:
        """
        Добавляет бонусные кредиты.

        Args:
            credit_type: Тип кредитов
            amount: Количество кредитов
            reason: Причина добавления бонусных кредитов

        Returns:
            True, если бонусные кредиты успешно добавлены
        """
        # Проверяем, что тип кредитов существует и лимит не безлимитный
        if credit_type not in self.credit_limits or self.credit_limits[credit_type] == -1:
            return False

        # Добавляем бонусные кредиты
        self.credit_limits[credit_type] += amount

        # Добавляем запись в историю
        timestamp = datetime.now().isoformat()
        operation_record = {
            "timestamp": timestamp,
            "operation": "add_bonus_credits",
            "credit_type": credit_type.value,
            "amount": amount,
            "reason": reason or "Бонусные кредиты",
        }
        self.operations_history.append(operation_record)

        self.logger.info(
            f"Добавлены бонусные кредиты для пользователя {self.user_id}: "
            f"{amount} кредитов типа {credit_type.value}. Причина: {reason}"
        )

        return True

    def _check_reset_counters(self) -> None:
        """Проверяет, нужно ли сбросить счетчики использования."""
        now = datetime.now()
        if now >= self.reset_time:
            # Сбрасываем счетчики
            for credit_type in CreditType:
                self.used_credits[credit_type] = 0

            # Обновляем время сброса
            self.reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                days=1
            )

            # Добавляем запись в историю
            timestamp = datetime.now().isoformat()
            operation_record = {
                "timestamp": timestamp,
                "operation": "reset_counters",
                "message": "Счетчики использования кредитов сброшены",
            }
            self.operations_history.append(operation_record)

            self.logger.info(
                f"Счетчики использования кредитов сброшены для пользователя {self.user_id}"
            )

    def _cleanup_history(self) -> None:
        """Очищает старые записи в истории операций."""
        if not self.operations_history:
            return

        retention_date = datetime.now() - timedelta(days=self.history_retention_period)
        new_history = []

        for record in self.operations_history:
            record_time = datetime.fromisoformat(record["timestamp"])
            if record_time >= retention_date:
                new_history.append(record)

        removed_count = len(self.operations_history) - len(new_history)
        if removed_count > 0:
            self.operations_history = new_history
            self.logger.info(f"Удалено {removed_count} устаревших записей из истории операций")

    def update_tier(self, new_tier: TierPlan) -> None:
        """
        Обновляет план использования.

        Args:
            new_tier: Новый план
        """
        # Сохраняем текущие лимиты и использование
        old_tier = self.tier
        old_limits = self.credit_limits.copy()
        old_used = self.used_credits.copy()

        # Устанавливаем новые лимиты
        self.tier = new_tier
        self.credit_limits = self.CREDIT_LIMITS[new_tier.value].copy()

        # Пересчитываем использованные кредиты
        # (сохраняем процент использования для непустых лимитов)
        for credit_type in CreditType:
            old_limit = old_limits.get(credit_type, 0)
            old_usage = old_used.get(credit_type, 0)
            new_limit = self.credit_limits.get(credit_type, 0)

            # Если старый лимит был безлимитным или равен 0, сбрасываем использование
            if old_limit <= 0:
                self.used_credits[credit_type] = 0
                continue

            # Если новый лимит безлимитный, сбрасываем использование
            if new_limit == -1:
                self.used_credits[credit_type] = 0
                continue

            # Пересчитываем использование с сохранением процента
            usage_percent = min(1.0, old_usage / old_limit)
            self.used_credits[credit_type] = int(usage_percent * new_limit)

        # Добавляем запись в историю
        timestamp = datetime.now().isoformat()
        operation_record = {
            "timestamp": timestamp,
            "operation": "update_tier",
            "old_tier": old_tier.value,
            "new_tier": new_tier.value,
            "message": f"План обновлен с {old_tier.value} до {new_tier.value}",
        }
        self.operations_history.append(operation_record)

        self.logger.info(
            f"Обновлен план использования для пользователя {self.user_id}: "
            f"{old_tier.value} -> {new_tier.value}"
        )

    def get_credits(self, credit_type: CreditType) -> float:
        """
        Получение доступных кредитов указанного типа.
        Этот метод добавлен для обратной совместимости с BudgetPlanner.

        Args:
            credit_type: Тип кредитов

        Returns:
            float: Количество доступных кредитов
        """
        # Преобразование credit_type в строку, если это enum
        if isinstance(credit_type, CreditType):
            type_value = credit_type.value
        else:
            type_value = credit_type

        credits_info = self.get_credits_info()
        return credits_info.get(type_value, {}).get("available", 0.0)
