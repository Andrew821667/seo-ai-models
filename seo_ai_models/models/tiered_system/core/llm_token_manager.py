"""
Менеджер токенов LLM для многоуровневой системы SEO AI Models.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import logging
import json
import os
from datetime import datetime, timedelta

from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Провайдеры LLM."""
    OPENAI = "openai"       # OpenAI (GPT)
    ANTHROPIC = "anthropic" # Anthropic (Claude)
    COHERE = "cohere"       # Cohere
    YANDEX = "yandex"       # YandexGPT
    SBER = "sber"           # SberAI (GigaChat)
    LLAMA = "llama"         # Llama (Meta)
    LOCAL = "local"         # Локальные модели

class LLMTokenManager:
    """
    Менеджер токенов LLM для многоуровневой системы SEO AI Models.
    
    Этот класс отвечает за управление использованием токенов LLM, 
    отслеживание лимитов и оптимизацию затрат.
    """
    
    # Лимиты токенов по умолчанию для разных уровней
    DEFAULT_TOKEN_LIMITS = {
        TierPlan.MICRO.value: {
            LLMProvider.OPENAI.value: 0,          # Нет доступа
            LLMProvider.ANTHROPIC.value: 0,       # Нет доступа
            LLMProvider.COHERE.value: 0,          # Нет доступа
            LLMProvider.YANDEX.value: 5000,       # 5K токенов
            LLMProvider.SBER.value: 5000,         # 5K токенов
            LLMProvider.LLAMA.value: 10000,       # 10K токенов
            LLMProvider.LOCAL.value: float('inf')  # Неограниченно
        },
        TierPlan.BASIC.value: {
            LLMProvider.OPENAI.value: 0,          # Нет доступа
            LLMProvider.ANTHROPIC.value: 0,       # Нет доступа
            LLMProvider.COHERE.value: 10000,      # 10K токенов
            LLMProvider.YANDEX.value: 20000,      # 20K токенов
            LLMProvider.SBER.value: 20000,        # 20K токенов
            LLMProvider.LLAMA.value: 50000,       # 50K токенов
            LLMProvider.LOCAL.value: float('inf')  # Неограниченно
        },
        TierPlan.PROFESSIONAL.value: {
            LLMProvider.OPENAI.value: 20000,      # 20K токенов
            LLMProvider.ANTHROPIC.value: 20000,   # 20K токенов
            LLMProvider.COHERE.value: 50000,      # 50K токенов
            LLMProvider.YANDEX.value: 100000,     # 100K токенов
            LLMProvider.SBER.value: 100000,       # 100K токенов
            LLMProvider.LLAMA.value: 200000,      # 200K токенов
            LLMProvider.LOCAL.value: float('inf')  # Неограниченно
        },
        TierPlan.ENTERPRISE.value: {
            LLMProvider.OPENAI.value: 100000,     # 100K токенов
            LLMProvider.ANTHROPIC.value: 100000,  # 100K токенов
            LLMProvider.COHERE.value: 200000,     # 200K токенов
            LLMProvider.YANDEX.value: 500000,     # 500K токенов
            LLMProvider.SBER.value: 500000,       # 500K токенов
            LLMProvider.LLAMA.value: 1000000,     # 1M токенов
            LLMProvider.LOCAL.value: float('inf')  # Неограниченно
        }
    }
    
    # Соотношение стоимости токенов для разных провайдеров (относительно OpenAI)
    TOKEN_COST_RATIO = {
        LLMProvider.OPENAI.value: 1.0,      # Базовый коэффициент
        LLMProvider.ANTHROPIC.value: 1.2,   # На 20% дороже
        LLMProvider.COHERE.value: 0.8,      # На 20% дешевле
        LLMProvider.YANDEX.value: 0.5,      # На 50% дешевле
        LLMProvider.SBER.value: 0.5,        # На 50% дешевле
        LLMProvider.LLAMA.value: 0.3,       # На 70% дешевле
        LLMProvider.LOCAL.value: 0.1        # На 90% дешевле
    }
    
    def __init__(self, 
                 user_id: str, 
                 tier: Union[TierPlan, str] = TierPlan.MICRO,
                 token_limits: Optional[Dict[str, Dict[str, int]]] = None,
                 data_dir: str = "data/llm_tokens"):
        """
        Инициализация менеджера токенов LLM.
        
        Args:
            user_id: ID пользователя
            tier: Уровень подписки пользователя
            token_limits: Пользовательские лимиты токенов
            data_dir: Директория для хранения данных
        """
        self.user_id = user_id
        
        # Преобразование tier в строку, если это enum
        if isinstance(tier, TierPlan):
            self.tier = tier.value
        else:
            self.tier = tier.lower()
        
        # Лимиты токенов
        self.token_limits = token_limits or self.DEFAULT_TOKEN_LIMITS
        
        # Директория данных
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Файл для хранения использования токенов
        self.token_usage_file = os.path.join(self.data_dir, f"{user_id}_token_usage.json")
        
        # Загрузка или создание записи использования токенов
        self.token_usage = self._load_token_usage()
    
    def _load_token_usage(self) -> Dict:
        """Загрузка или создание записи использования токенов."""
        if os.path.exists(self.token_usage_file):
            try:
                with open(self.token_usage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Ошибка загрузки использования токенов: {e}")
                return self._create_default_token_usage()
        else:
            return self._create_default_token_usage()
    
    def _create_default_token_usage(self) -> Dict:
        """Создание записи использования токенов по умолчанию."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        return {
            "user_id": self.user_id,
            "tier": self.tier,
            "created_at": today,
            "updated_at": today,
            "total_usage": {provider.value: 0 for provider in LLMProvider},
            "daily_usage": {
                today: {provider.value: 0 for provider in LLMProvider}
            },
            "history": []
        }
    
    def _save_token_usage(self) -> bool:
        """Сохранение записи использования токенов."""
        try:
            with open(self.token_usage_file, 'w') as f:
                json.dump(self.token_usage, f, indent=4)
            return True
        except IOError as e:
            logger.error(f"Ошибка сохранения использования токенов: {e}")
            return False
    
    def get_token_limit(self, provider: Union[LLMProvider, str]) -> int:
        """
        Получение лимита токенов для указанного провайдера.
        
        Args:
            provider: Провайдер LLM
            
        Returns:
            int: Лимит токенов
        """
        # Преобразование provider в строку, если это enum
        if isinstance(provider, LLMProvider):
            provider = provider.value
            
        return self.token_limits.get(self.tier, {}).get(provider, 0)
    
    def get_token_usage(self, provider: Union[LLMProvider, str]) -> int:
        """
        Получение текущего использования токенов для указанного провайдера.
        
        Args:
            provider: Провайдер LLM
            
        Returns:
            int: Текущее использование токенов
        """
        # Преобразование provider в строку, если это enum
        if isinstance(provider, LLMProvider):
            provider = provider.value
            
        return self.token_usage["total_usage"].get(provider, 0)
    
    def get_daily_token_usage(self, provider: Union[LLMProvider, str], date: Optional[str] = None) -> int:
        """
        Получение дневного использования токенов для указанного провайдера.
        
        Args:
            provider: Провайдер LLM
            date: Дата в формате 'YYYY-MM-DD' (по умолчанию - сегодня)
            
        Returns:
            int: Дневное использование токенов
        """
        # Преобразование provider в строку, если это enum
        if isinstance(provider, LLMProvider):
            provider = provider.value
            
        # Если дата не указана, используем сегодняшнюю
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        # Если записи для указанной даты нет, возвращаем 0
        if date not in self.token_usage["daily_usage"]:
            return 0
            
        return self.token_usage["daily_usage"][date].get(provider, 0)
    
    def record_token_usage(self, 
                          provider: Union[LLMProvider, str], 
                          tokens: int,
                          operation: str = "analysis") -> bool:
        """
        Запись использования токенов.
        
        Args:
            provider: Провайдер LLM
            tokens: Количество использованных токенов
            operation: Операция, для которой использовались токены
            
        Returns:
            bool: True, если запись успешна, False в противном случае
        """
        # Преобразование provider в строку, если это enum
        if isinstance(provider, LLMProvider):
            provider = provider.value
            
        # Получение текущей даты
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Обновление общего использования
        self.token_usage["total_usage"][provider] = self.token_usage["total_usage"].get(provider, 0) + tokens
        
        # Обновление дневного использования
        if today not in self.token_usage["daily_usage"]:
            self.token_usage["daily_usage"][today] = {p.value: 0 for p in LLMProvider}
            
        self.token_usage["daily_usage"][today][provider] = self.token_usage["daily_usage"][today].get(provider, 0) + tokens
        
        # Добавление записи в историю
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.token_usage["history"].append({
            "timestamp": timestamp,
            "provider": provider,
            "tokens": tokens,
            "operation": operation
        })
        
        # Обновление времени обновления
        self.token_usage["updated_at"] = today
        
        # Сохранение изменений
        return self._save_token_usage()
    
    def check_token_availability(self, provider: Union[LLMProvider, str], tokens: int) -> bool:
        """
        Проверка доступности токенов для использования.
        
        Args:
            provider: Провайдер LLM
            tokens: Количество требуемых токенов
            
        Returns:
            bool: True, если токены доступны, False в противном случае
        """
        # Преобразование provider в строку, если это enum
        if isinstance(provider, LLMProvider):
            provider = provider.value
            
        # Получение лимита токенов
        limit = self.get_token_limit(provider)
        
        # Если лимит неограничен (infinity), возвращаем True
        if limit == float('inf'):
            return True
            
        # Получение текущего использования
        usage = self.get_token_usage(provider)
        
        # Проверка, что использование с учетом новых токенов не превысит лимит
        return usage + tokens <= limit
    
    def get_recommended_provider(self, tokens: int) -> Tuple[str, float]:
        """
        Получение рекомендуемого провайдера для использования токенов.
        
        Args:
            tokens: Количество требуемых токенов
            
        Returns:
            Tuple[str, float]: Провайдер и относительная стоимость
        """
        available_providers = []
        
        # Перебор всех провайдеров
        for provider in LLMProvider:
            # Проверка доступности токенов
            if self.check_token_availability(provider, tokens):
                # Расчет относительной стоимости
                cost = tokens * self.TOKEN_COST_RATIO.get(provider.value, 1.0)
                available_providers.append((provider.value, cost))
        
        # Если нет доступных провайдеров, возвращаем LOCAL с максимальной стоимостью
        if not available_providers:
            return (LLMProvider.LOCAL.value, tokens * self.TOKEN_COST_RATIO.get(LLMProvider.LOCAL.value, 0.1))
            
        # Сортировка по стоимости (возрастание)
        available_providers.sort(key=lambda x: x[1])
        
        # Возврат провайдера с минимальной стоимостью
        return available_providers[0]
    
    def reset_daily_usage(self) -> bool:
        """Сброс дневного использования токенов."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.token_usage["daily_usage"][today] = {provider.value: 0 for provider in LLMProvider}
        return self._save_token_usage()
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики использования токенов.
        
        Returns:
            Dict[str, Any]: Статистика использования токенов
        """
        # Получение текущей даты
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Расчет статистики за последние 7 дней
        last_7_days = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
        last_7_days_usage = {}
        
        for provider in LLMProvider:
            provider_value = provider.value
            last_7_days_usage[provider_value] = sum(
                self.token_usage["daily_usage"].get(day, {}).get(provider_value, 0)
                for day in last_7_days
            )
        
        # Формирование статистики
        statistics = {
            "user_id": self.user_id,
            "tier": self.tier,
            "total_usage": self.token_usage["total_usage"],
            "today_usage": self.token_usage["daily_usage"].get(today, {}),
            "last_7_days_usage": last_7_days_usage,
            "token_limits": self.token_limits.get(self.tier, {}),
            "utilization_percentage": {
                provider.value: (self.token_usage["total_usage"].get(provider.value, 0) / self.token_limits.get(self.tier, {}).get(provider.value, 1)) * 100
                if self.token_limits.get(self.tier, {}).get(provider.value, 0) > 0 else 0
                for provider in LLMProvider
            }
        }
        
        return statistics
