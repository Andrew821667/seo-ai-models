"""
Базовая архитектура многоуровневой системы SEO Advisor.

Модуль предоставляет основной класс TieredAdvisor, который поддерживает
различные планы использования (Micro, Basic, Professional) с разными
возможностями и уровнями потребления ресурсов.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union

# Импорты из SEO Advisor
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor
from seo_ai_models.models.seo_advisor.enhanced_advisor import EnhancedSEOAdvisor
from seo_ai_models.models.seo_advisor.llm_integration_adapter import LLMEnhancedSEOAdvisor

# Импорты из Tiered System
from seo_ai_models.models.tiered_system.core.resource_optimizer import ResourceOptimizer
from seo_ai_models.models.tiered_system.core.feature_gating import FeatureGating
from seo_ai_models.models.tiered_system.core.llm_token_manager import LLMTokenManager


class TierPlan(Enum):
    """Планы использования многоуровневой системы."""
    MICRO = "micro"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TieredAdvisor:
    """
    Многоуровневая система анализа SEO с разными планами использования.
    
    Класс обеспечивает доступ к различным уровням функциональности SEO Advisor
    в зависимости от выбранного плана, оптимизируя использование ресурсов
    и обеспечивая соответствующие ограничения.
    """
    
    def __init__(
        self,
        tier: Union[str, TierPlan] = TierPlan.BASIC,
        config: Optional[Dict[str, Any]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        optimize_resources: bool = True,
        **kwargs
    ):
        """
        Инициализирует TieredAdvisor с указанным планом.
        
        Args:
            tier: План использования (micro, basic, professional, enterprise)
            config: Дополнительные параметры конфигурации
            api_keys: API ключи для различных сервисов
            optimize_resources: Флаг оптимизации ресурсов
            **kwargs: Дополнительные аргументы для advisor
        """
        self.logger = logging.getLogger(__name__)
        
        # Нормализуем tier
        if isinstance(tier, str):
            try:
                self.tier = TierPlan(tier.lower())
            except ValueError:
                self.logger.warning(f"Неизвестный план '{tier}', используется BASIC")
                self.tier = TierPlan.BASIC
        else:
            self.tier = tier
            
        self.config = config or {}
        self.api_keys = api_keys or {}
        
        # Инициализируем компоненты управления ресурсами
        self.resource_optimizer = ResourceOptimizer(tier=self.tier, **kwargs)
        self.feature_gating = FeatureGating(tier=self.tier, **kwargs)
        
        # Инициализируем токен-менеджер для LLM только если не микро-план
        self.token_manager = None
        if self.tier != TierPlan.MICRO:
            self.token_manager = LLMTokenManager(
                tier=self.tier,
                api_keys=self.api_keys,
                **kwargs
            )
        
        # Выбираем соответствующий advisor в зависимости от плана
        self._initialize_advisor()
        
        self.logger.info(f"TieredAdvisor инициализирован с планом {self.tier.value}")
    
    def _initialize_advisor(self):
        """Инициализирует соответствующий advisor в зависимости от плана."""
        if self.tier == TierPlan.MICRO:
            # Для микро-плана используем базовый SEOAdvisor без LLM
            self.advisor = SEOAdvisor(
                config=self.resource_optimizer.optimize_config(self.config)
            )
        elif self.tier == TierPlan.BASIC:
            # Для базового плана используем EnhancedSEOAdvisor с ограниченным LLM
            self.advisor = EnhancedSEOAdvisor(
                config=self.resource_optimizer.optimize_config(self.config)
            )
        else:
            # Для профессионального и enterprise планов используем LLM-интеграцию
            self.advisor = LLMEnhancedSEOAdvisor(
                config=self.resource_optimizer.optimize_config(self.config),
                api_keys=self.api_keys
            )
    
    def analyze_content(
        self, 
        content: str, 
        keywords: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Анализирует контент с учетом выбранного плана и доступных функций.
        
        Args:
            content: Текст для анализа
            keywords: Ключевые слова
            context: Контекстная информация
            **kwargs: Дополнительные параметры
            
        Returns:
            Результаты анализа с учетом ограничений плана
        """
        # Применяем ограничения feature gating
        allowed_features = self.feature_gating.get_allowed_features()
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in allowed_features}
        
        # Для планов с LLM управляем использованием токенов
        if self.token_manager and self.tier != TierPlan.MICRO:
            # Проверяем доступность токенов и оптимизируем запрос
            content_length = len(content)
            if not self.token_manager.check_tokens_available(content_length):
                self.logger.warning(
                    f"Недостаточно токенов для полного анализа. "
                    f"Будет выполнен ограниченный анализ."
                )
                # Здесь можно применить дополнительные оптимизации или ограничения
        
        # Выполняем анализ с помощью соответствующего advisor
        results = self.advisor.analyze_content(
            content=content,
            keywords=keywords,
            context=context,
            **filtered_kwargs
        )
        
        # Применяем постобработку результатов в соответствии с планом
        processed_results = self._process_results_by_tier(results)
        
        return processed_results
    
    def _process_results_by_tier(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает результаты анализа в соответствии с выбранным планом.
        
        Args:
            results: Исходные результаты анализа
            
        Returns:
            Обработанные результаты с учетом ограничений плана
        """
        # Ограничиваем результаты в зависимости от плана
        if self.tier == TierPlan.MICRO:
            # Для микро-плана оставляем только базовые метрики
            allowed_keys = [
                'basic_metrics', 'readability', 'keywords_basic',
                'structure_basic', 'core_recommendations'
            ]
            filtered_results = {k: v for k, v in results.items() if k in allowed_keys}
            return filtered_results
            
        elif self.tier == TierPlan.BASIC:
            # Для базового плана исключаем расширенные LLM-анализы
            excluded_keys = [
                'llm_compatibility', 'citability_score', 
                'llm_eeat_analysis', 'feature_importance'
            ]
            filtered_results = {k: v for k, v in results.items() if k not in excluded_keys}
            return filtered_results
            
        # Для профессионального и enterprise планов возвращаем полные результаты
        return results
    
    def get_tier_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущем плане и его ограничениях.
        
        Returns:
            Информация о плане
        """
        allowed_features = self.feature_gating.get_allowed_features()
        
        tier_info = {
            'plan': self.tier.value,
            'allowed_features': allowed_features,
            'resource_limits': self.resource_optimizer.get_resource_limits(),
        }
        
        # Добавляем информацию о токенах, если доступно
        if self.token_manager:
            tier_info['token_info'] = self.token_manager.get_token_info()
            
        return tier_info
    
    def upgrade_tier(self, new_tier: Union[str, TierPlan]) -> bool:
        """
        Обновляет план использования.
        
        Args:
            new_tier: Новый план
            
        Returns:
            True, если обновление выполнено успешно
        """
        # Нормализуем new_tier
        if isinstance(new_tier, str):
            try:
                new_tier = TierPlan(new_tier.lower())
            except ValueError:
                self.logger.error(f"Неизвестный план '{new_tier}'")
                return False
        
        # Проверяем, является ли это апгрейдом
        current_tier_index = list(TierPlan).index(self.tier)
        new_tier_index = list(TierPlan).index(new_tier)
        
        if new_tier_index < current_tier_index:
            self.logger.warning(
                f"Даунгрейд с {self.tier.value} до {new_tier.value} не рекомендуется"
            )
        
        # Обновляем план
        self.tier = new_tier
        
        # Обновляем компоненты
        self.resource_optimizer.update_tier(new_tier)
        self.feature_gating.update_tier(new_tier)
        
        if new_tier == TierPlan.MICRO:
            self.token_manager = None
        elif self.token_manager:
            self.token_manager.update_tier(new_tier)
        else:
            self.token_manager = LLMTokenManager(
                tier=new_tier,
                api_keys=self.api_keys
            )
        
        # Реинициализируем advisor
        self._initialize_advisor()
        
        self.logger.info(f"План обновлен до {new_tier.value}")
        return True
