"""
Менеджер токенов для LLM-сервисов.

Модуль предоставляет функциональность для управления использованием токенов
при работе с LLM API, оптимизации затрат и контроля лимитов в зависимости от плана.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union

# Импорты из tiered_system
from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan

# Импорты из llm_integration
try:
    from seo_ai_models.models.llm_integration.service.cost_estimator import CostEstimator
    from seo_ai_models.models.llm_integration.common.constants import LLM_PROVIDERS, LLM_MODELS
except ImportError:
    # Создаем заглушки, если модули не найдены
    CostEstimator = object
    LLM_PROVIDERS = {}
    LLM_MODELS = {}


class LLMTokenManager:
    """
    Менеджер токенов для LLM-сервисов.
    
    Класс отвечает за управление использованием токенов при работе с LLM API,
    оптимизацию затрат и контроль лимитов в зависимости от плана использования.
    """
    
    # Лимиты токенов для каждого плана (токенов в день)
    TOKEN_LIMITS = {
        'micro': 5000,       # 5K токенов в день
        'basic': 50000,      # 50K токенов в день
        'professional': 500000,  # 500K токенов в день
        'enterprise': -1,    # Неограниченно
    }
    
    # Лимиты запросов для каждого плана (запросов в день)
    REQUEST_LIMITS = {
        'micro': 10,         # 10 запросов в день
        'basic': 100,        # 100 запросов в день
        'professional': 1000,  # 1000 запросов в день
        'enterprise': -1,    # Неограниченно
    }
    
    def __init__(
        self, 
        tier: TierPlan,
        api_keys: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Инициализирует менеджер токенов LLM.
        
        Args:
            tier: План использования
            api_keys: API ключи для различных сервисов
            **kwargs: Дополнительные параметры
        """
        self.logger = logging.getLogger(__name__)
        self.tier = tier
        self.api_keys = api_keys or {}
        
        # Инициализируем оценщик стоимости
        self.cost_estimator = None
        try:
            self.cost_estimator = CostEstimator(api_keys=self.api_keys)
        except Exception as e:
            self.logger.warning(f"Не удалось инициализировать CostEstimator: {e}")
        
        # Устанавливаем лимиты в зависимости от плана
        self.token_limit = self.TOKEN_LIMITS[tier.value]
        self.request_limit = self.REQUEST_LIMITS[tier.value]
        
        # Счетчики использования
        self.tokens_used = 0
        self.requests_used = 0
        
        # Определяем период сброса (0:00 следующего дня)
        self.reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        
        # Применяем дополнительные параметры
        self.custom_token_limit = kwargs.get('custom_token_limit')
        if self.custom_token_limit and self.tier != TierPlan.MICRO:
            self.token_limit = self.custom_token_limit
            self.logger.info(f"Установлен пользовательский лимит токенов: {self.token_limit}")
            
        self.logger.info(f"LLMTokenManager инициализирован для плана {tier.value}")
    
    def check_tokens_available(self, content_length: int) -> bool:
        """
        Проверяет доступность токенов для анализа контента указанной длины.
        
        Args:
            content_length: Длина контента в символах
            
        Returns:
            True, если токенов достаточно
        """
        # Проверяем, нужно ли сбросить счетчики
        self._check_reset_counters()
        
        # Для Enterprise плана всегда возвращаем True
        if self.token_limit == -1:
            return True
        
        # Примерная оценка количества токенов (приблизительно 4 символа на токен)
        estimated_tokens = content_length // 4
        
        # Добавляем запас 20% для учета промптов и метаданных
        total_estimated_tokens = int(estimated_tokens * 1.2)
        
        # Проверяем, достаточно ли токенов
        return (self.tokens_used + total_estimated_tokens) <= self.token_limit
    
    def track_token_usage(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        provider: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Отслеживает использование токенов.
        
        Args:
            input_tokens: Количество входных токенов
            output_tokens: Количество выходных токенов
            provider: Провайдер LLM
            model: Модель LLM
            
        Returns:
            Информация об использовании и стоимости
        """
        # Проверяем, нужно ли сбросить счетчики
        self._check_reset_counters()
        
        # Обновляем счетчики
        total_tokens = input_tokens + output_tokens
        self.tokens_used += total_tokens
        self.requests_used += 1
        
        # Оцениваем стоимость, если доступен cost_estimator
        cost_info = {"cost": None, "currency": "USD"}
        if self.cost_estimator:
            try:
                cost_info = self.cost_estimator.estimate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider=provider,
                    model=model
                )
            except Exception as e:
                self.logger.warning(f"Не удалось оценить стоимость: {e}")
        
        # Формируем информацию об использовании
        usage_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost_info.get("cost"),
            "currency": cost_info.get("currency"),
            "tokens_remaining": self._get_tokens_remaining(),
            "requests_remaining": self._get_requests_remaining(),
            "reset_time": self.reset_time.isoformat(),
        }
        
        # Логируем, если приближаемся к лимиту
        if self.token_limit != -1:  # Если не безлимитный
            usage_percent = (self.tokens_used / self.token_limit) * 100
            if usage_percent >= 80:
                self.logger.warning(
                    f"Использовано {usage_percent:.1f}% лимита токенов"
                )
        
        return usage_info
    
    def _check_reset_counters(self) -> None:
        """Проверяет, нужно ли сбросить счетчики использования."""
        now = datetime.now()
        if now >= self.reset_time:
            # Сбрасываем счетчики
            self.tokens_used = 0
            self.requests_used = 0
            
            # Обновляем время сброса
            self.reset_time = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            
            self.logger.info("Счетчики использования токенов сброшены")
    
    def _get_tokens_remaining(self) -> Optional[int]:
        """
        Возвращает оставшееся количество токенов.
        
        Returns:
            Количество оставшихся токенов или None для безлимитного плана
        """
        if self.token_limit == -1:
            return None
        return max(0, self.token_limit - self.tokens_used)
    
    def _get_requests_remaining(self) -> Optional[int]:
        """
        Возвращает оставшееся количество запросов.
        
        Returns:
            Количество оставшихся запросов или None для безлимитного плана
        """
        if self.request_limit == -1:
            return None
        return max(0, self.request_limit - self.requests_used)
    
    def get_token_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о токенах и использовании.
        
        Returns:
            Информация о токенах
        """
        return {
            "token_limit": self.token_limit,
            "tokens_used": self.tokens_used,
            "tokens_remaining": self._get_tokens_remaining(),
            "request_limit": self.request_limit,
            "requests_used": self.requests_used,
            "requests_remaining": self._get_requests_remaining(),
            "reset_time": self.reset_time.isoformat(),
        }
    
    def update_tier(self, new_tier: TierPlan) -> None:
        """
        Обновляет план использования.
        
        Args:
            new_tier: Новый план
        """
        self.tier = new_tier
        
        # Обновляем лимиты
        self.token_limit = self.TOKEN_LIMITS[new_tier.value]
        self.request_limit = self.REQUEST_LIMITS[new_tier.value]
        
        # Если установлен пользовательский лимит, используем его (кроме микро-плана)
        if self.custom_token_limit and new_tier != TierPlan.MICRO:
            self.token_limit = self.custom_token_limit
            
        self.logger.info(f"LLMTokenManager обновлен до плана {new_tier.value}")
