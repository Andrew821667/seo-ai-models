# -*- coding: utf-8 -*-
"""
UpgradePath - Компонент для обновления плана Freemium-модели.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import os

from seo_ai_models.models.freemium.core.enums import FreemiumPlan

logger = logging.getLogger(__name__)


class UpgradePath:
    """
    Управляет путями обновления для Freemium-модели.

    Предоставляет функциональность для обновления плана пользователя
    и отображения доступных путей обновления.
    """

    # Пути обновления (от какого плана к какому можно обновиться)
    UPGRADE_PATHS = {
        FreemiumPlan.FREE: [
            FreemiumPlan.MICRO,
            FreemiumPlan.BASIC,
            FreemiumPlan.PROFESSIONAL,
            FreemiumPlan.ENTERPRISE,
        ],
        FreemiumPlan.MICRO: [
            FreemiumPlan.BASIC,
            FreemiumPlan.PROFESSIONAL,
            FreemiumPlan.ENTERPRISE,
        ],
        FreemiumPlan.BASIC: [FreemiumPlan.PROFESSIONAL, FreemiumPlan.ENTERPRISE],
        FreemiumPlan.PROFESSIONAL: [FreemiumPlan.ENTERPRISE],
        FreemiumPlan.ENTERPRISE: [],
    }

    # Преимущества обновления
    UPGRADE_BENEFITS = {
        FreemiumPlan.MICRO: [
            "До 100 URL в месяц",
            "Полный набор рекомендаций",
            "Базовые функции оптимизации",
        ],
        FreemiumPlan.BASIC: ["До 1 000 URL в месяц", "Расширенная оптимизация", "Доступ к API"],
        FreemiumPlan.PROFESSIONAL: [
            "До 10 000 URL в месяц",
            "Полная интеграция с LLM",
            "Расширенная аналитика",
        ],
        FreemiumPlan.ENTERPRISE: [
            "Неограниченное количество URL",
            "Индивидуальные настройки",
            "Выделенная поддержка",
        ],
    }

    # Цены планов
    PLAN_PRICES = {
        FreemiumPlan.MICRO: "1 990 ₽/месяц",
        FreemiumPlan.BASIC: "5 990 ₽/месяц",
        FreemiumPlan.PROFESSIONAL: "15 990 ₽/месяц",
        FreemiumPlan.ENTERPRISE: "От 49 990 ₽/месяц",
    }

    def __init__(
        self,
        current_plan: Union[str, FreemiumPlan],
        user_id: Optional[str] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Инициализирует UpgradePath.

        Args:
            current_plan: Текущий план пользователя
            user_id: Идентификатор пользователя
            storage_path: Путь для хранения данных о обновлениях
        """
        self.current_plan = (
            current_plan if isinstance(current_plan, FreemiumPlan) else FreemiumPlan(current_plan)
        )
        self.user_id = user_id or "anonymous"
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".seo_ai_models", "upgrades"
        )

        # Создаем директорию для хранения данных о обновлениях, если она не существует
        os.makedirs(self.storage_path, exist_ok=True)

        # История обновлений
        self.upgrade_history = self._load_upgrade_history()

    def _get_history_file_path(self) -> str:
        """
        Возвращает путь к файлу с историей обновлений.

        Returns:
            Путь к файлу
        """
        return os.path.join(self.storage_path, f"{self.user_id}_upgrades.json")

    def _load_upgrade_history(self) -> List[Dict[str, Any]]:
        """
        Загружает историю обновлений из файла.

        Returns:
            История обновлений
        """
        history_file = self._get_history_file_path()

        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading upgrade history for user {self.user_id}: {e}")
                return []

        return []

    def _save_upgrade_history(self):
        """Сохраняет историю обновлений в файл."""
        history_file = self._get_history_file_path()

        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.upgrade_history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving upgrade history for user {self.user_id}: {e}")

    def get_upgrade_options(self) -> Dict[FreemiumPlan, Dict[str, Any]]:
        """
        Возвращает доступные опции обновления.

        Returns:
            Словарь с доступными опциями обновления
        """
        options = {}

        for target_plan in self.UPGRADE_PATHS.get(self.current_plan, []):
            options[target_plan] = {
                "benefits": self.UPGRADE_BENEFITS.get(target_plan, []),
                "price": self.PLAN_PRICES.get(target_plan, ""),
                "upgrade_path": [self.current_plan, target_plan],
            }

        return options

    def initiate_upgrade(self, target_plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Инициирует процесс обновления.

        Args:
            target_plan: Целевой план

        Returns:
            Результат инициализации обновления
        """
        target_plan = (
            target_plan if isinstance(target_plan, FreemiumPlan) else FreemiumPlan(target_plan)
        )

        # Проверяем, можно ли обновиться до указанного плана
        if target_plan not in self.UPGRADE_PATHS.get(self.current_plan, []):
            return {
                "status": "error",
                "message": f"Cannot upgrade from {self.current_plan.value} to {target_plan.value}",
            }

        # Записываем обновление в историю
        upgrade_info = {
            "from_plan": self.current_plan.value,
            "to_plan": target_plan.value,
            "initiated_at": datetime.now().isoformat(),
            "status": "initiated",
        }

        self.upgrade_history.append(upgrade_info)
        self._save_upgrade_history()

        return {
            "status": "success",
            "message": f"Upgrade from {self.current_plan.value} to {target_plan.value} initiated",
            "current_plan": self.current_plan.value,
            "target_plan": target_plan.value,
            "benefits": self.UPGRADE_BENEFITS.get(target_plan, []),
            "price": self.PLAN_PRICES.get(target_plan, ""),
        }

    def get_upgrade_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает историю обновлений.

        Returns:
            История обновлений
        """
        return self.upgrade_history

    def complete_upgrade(self, target_plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Завершает процесс обновления.

        Args:
            target_plan: Целевой план

        Returns:
            Результат завершения обновления
        """
        target_plan = (
            target_plan if isinstance(target_plan, FreemiumPlan) else FreemiumPlan(target_plan)
        )

        # Проверяем, есть ли инициированное обновление
        initiated_upgrades = [
            upgrade
            for upgrade in self.upgrade_history
            if upgrade["to_plan"] == target_plan.value and upgrade["status"] == "initiated"
        ]

        if not initiated_upgrades:
            return {
                "status": "error",
                "message": f"No initiated upgrade to {target_plan.value} found",
            }

        # Обновляем статус последнего инициированного обновления
        initiated_upgrades[-1]["status"] = "completed"
        initiated_upgrades[-1]["completed_at"] = datetime.now().isoformat()

        # Обновляем текущий план
        self.current_plan = target_plan

        # Сохраняем изменения
        self._save_upgrade_history()

        return {
            "status": "success",
            "message": f"Upgrade to {target_plan.value} completed",
            "current_plan": self.current_plan.value,
        }

    def cancel_upgrade(self, target_plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Отменяет процесс обновления.

        Args:
            target_plan: Целевой план

        Returns:
            Результат отмены обновления
        """
        target_plan = (
            target_plan if isinstance(target_plan, FreemiumPlan) else FreemiumPlan(target_plan)
        )

        # Проверяем, есть ли инициированное обновление
        initiated_upgrades = [
            upgrade
            for upgrade in self.upgrade_history
            if upgrade["to_plan"] == target_plan.value and upgrade["status"] == "initiated"
        ]

        if not initiated_upgrades:
            return {
                "status": "error",
                "message": f"No initiated upgrade to {target_plan.value} found",
            }

        # Обновляем статус последнего инициированного обновления
        initiated_upgrades[-1]["status"] = "cancelled"
        initiated_upgrades[-1]["cancelled_at"] = datetime.now().isoformat()

        # Сохраняем изменения
        self._save_upgrade_history()

        return {"status": "success", "message": f"Upgrade to {target_plan.value} cancelled"}

    def get_personalized_recommendations(self):
        """Возвращает персонализированные рекомендации."""
        return [{"plan": "basic", "reason": "Расширьте возможности"}]

    def get_trial_offers(self):
        """Возвращает пробные предложения."""
        return [{"offer_type": "7_day_trial", "plan": "professional"}]
