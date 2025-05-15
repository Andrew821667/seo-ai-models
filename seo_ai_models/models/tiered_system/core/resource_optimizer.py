"""
Оптимизатор ресурсов для многоуровневой системы.

Модуль предоставляет функциональность для оптимизации использования 
ресурсов (CPU, память, запросы к API) в зависимости от выбранного плана.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Union

from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan


class ResourceType(Enum):
    """Типы ресурсов, которые могут быть ограничены."""
    CPU = "cpu"
    MEMORY = "memory"
    API_REQUESTS = "api_requests"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"


class ResourceOptimizer:
    """
    Оптимизатор ресурсов для многоуровневой системы.
    
    Класс отвечает за управление и оптимизацию использования ресурсов
    в зависимости от выбранного плана использования.
    """
    
    # Базовые лимиты ресурсов для каждого плана
    RESOURCE_LIMITS = {
        'micro': {
            ResourceType.CPU: 1,           # 1 ядро
            ResourceType.MEMORY: 512,      # 512 МБ
            ResourceType.API_REQUESTS: 10,  # 10 запросов в день
            ResourceType.STORAGE: 50,      # 50 МБ
            ResourceType.BANDWIDTH: 1,     # 1 ГБ в месяц
        },
        'basic': {
            ResourceType.CPU: 2,           # 2 ядра
            ResourceType.MEMORY: 1024,     # 1 ГБ
            ResourceType.API_REQUESTS: 100, # 100 запросов в день
            ResourceType.STORAGE: 200,     # 200 МБ
            ResourceType.BANDWIDTH: 5,     # 5 ГБ в месяц
        },
        'professional': {
            ResourceType.CPU: 4,           # 4 ядра
            ResourceType.MEMORY: 4096,     # 4 ГБ
            ResourceType.API_REQUESTS: 1000, # 1000 запросов в день
            ResourceType.STORAGE: 1024,    # 1 ГБ
            ResourceType.BANDWIDTH: 20,    # 20 ГБ в месяц
        },
        'enterprise': {
            ResourceType.CPU: 8,           # 8 ядер
            ResourceType.MEMORY: 8192,     # 8 ГБ
            ResourceType.API_REQUESTS: -1,  # Неограниченно
            ResourceType.STORAGE: 10240,   # 10 ГБ
            ResourceType.BANDWIDTH: 100,   # 100 ГБ в месяц
        },
    }
    
    def __init__(self, tier: TierPlan, **kwargs):
        """
        Инициализирует оптимизатор ресурсов.
        
        Args:
            tier: План использования
            **kwargs: Дополнительные параметры
        """
        self.logger = logging.getLogger(__name__)
        self.tier = tier
        
        # Устанавливаем лимиты ресурсов в зависимости от плана
        # Используем tier.value вместо самого tier в качестве ключа
        self.resource_limits = self.RESOURCE_LIMITS[tier.value]
        
        # Счетчики использования ресурсов
        self.resource_usage = {
            ResourceType.API_REQUESTS: 0,
            ResourceType.BANDWIDTH: 0,
        }
        
        self.logger.info(f"ResourceOptimizer инициализирован для плана {tier.value}")
    
    def optimize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизирует конфигурацию с учетом ограничений ресурсов.
        
        Args:
            config: Исходная конфигурация
            
        Returns:
            Оптимизированная конфигурация
        """
        optimized_config = config.copy()
        
        # Применяем оптимизации в зависимости от плана
        if self.tier == TierPlan.MICRO:
            # Для микро-плана ограничиваем параллельные процессы и кэшируем результаты
            optimized_config['max_parallel_processes'] = min(
                optimized_config.get('max_parallel_processes', 4), 1
            )
            optimized_config['use_cache'] = True
            optimized_config['cache_ttl'] = optimized_config.get('cache_ttl', 86400) # 1 день
            
        elif self.tier == TierPlan.BASIC:
            # Для базового плана ограничиваем параллельные процессы
            optimized_config['max_parallel_processes'] = min(
                optimized_config.get('max_parallel_processes', 4), 2
            )
            
        # Для всех планов устанавливаем таймауты в зависимости от плана
        # Исправление: используем строковые ключи вместо объектов TierPlan
        timeouts = {
            'micro': 10,    # 10 секунд
            'basic': 30,    # 30 секунд
            'professional': 60,  # 60 секунд
            'enterprise': 120,   # 120 секунд
        }
        optimized_config['request_timeout'] = timeouts[self.tier.value]
        
        return optimized_config
    
    def check_resource_availability(
        self, 
        resource_type: ResourceType,
        required_amount: float
    ) -> bool:
        """
        Проверяет доступность ресурса.
        
        Args:
            resource_type: Тип ресурса
            required_amount: Требуемое количество
            
        Returns:
            True, если ресурс доступен
        """
        limit = self.resource_limits.get(resource_type)
        
        # Неограниченный ресурс (для Enterprise)
        if limit == -1:
            return True
            
        current_usage = self.resource_usage.get(resource_type, 0)
        
        return (current_usage + required_amount) <= limit
    
    def track_resource_usage(
        self, 
        resource_type: ResourceType,
        used_amount: float
    ) -> None:
        """
        Отслеживает использование ресурса.
        
        Args:
            resource_type: Тип ресурса
            used_amount: Использованное количество
        """
        if resource_type in self.resource_usage:
            self.resource_usage[resource_type] += used_amount
            
            # Логируем, если приближаемся к лимиту
            limit = self.resource_limits.get(resource_type)
            if limit != -1:  # Если не безлимитный
                usage_percent = (self.resource_usage[resource_type] / limit) * 100
                if usage_percent >= 80:
                    self.logger.warning(
                        f"Использовано {usage_percent:.1f}% лимита {resource_type.value}"
                    )
    
    def reset_usage_counters(self, resource_type: Optional[ResourceType] = None) -> None:
        """
        Сбрасывает счетчики использования ресурсов.
        
        Args:
            resource_type: Конкретный тип ресурса для сброса или None для всех
        """
        if resource_type:
            if resource_type in self.resource_usage:
                self.resource_usage[resource_type] = 0
        else:
            for res_type in self.resource_usage:
                self.resource_usage[res_type] = 0
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """
        Возвращает текущие лимиты ресурсов.
        
        Returns:
            Словарь с лимитами ресурсов
        """
        # Преобразуем Enum-ключи в строки для JSON-сериализации
        return {res_type.value: limit for res_type, limit in self.resource_limits.items()}
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Возвращает текущее использование ресурсов.
        
        Returns:
            Словарь с использованием ресурсов
        """
        # Преобразуем Enum-ключи в строки для JSON-сериализации
        return {res_type.value: usage for res_type, usage in self.resource_usage.items()}
    
    def update_tier(self, new_tier: TierPlan) -> None:
        """
        Обновляет план использования.
        
        Args:
            new_tier: Новый план
        """
        self.tier = new_tier
        self.resource_limits = self.RESOURCE_LIMITS[new_tier.value]
        self.logger.info(f"ResourceOptimizer обновлен до плана {new_tier.value}")
