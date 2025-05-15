"""
Система управления доступом к функциям в зависимости от уровня подписки.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union

class TierPlan(Enum):
    """Уровни подписки в системе."""
    MICRO = "micro"           # Микро-бизнес (базовые возможности)
    BASIC = "basic"           # Базовый (стандартные возможности)
    PROFESSIONAL = "professional"  # Профессиональный (расширенные возможности)
    ENTERPRISE = "enterprise"      # Корпоративный (полный доступ)

class FeatureGating:
    """
    Система управления доступом к функциям в зависимости от уровня подписки.
    
    Этот класс отвечает за определение доступных функций для каждого уровня
    подписки и контроль доступа к защищенным функциям.
    """
    
    # Определение доступных функций для каждого уровня
    DEFAULT_FEATURES = {
        TierPlan.MICRO.value: {
            "max_urls": 100,
            "max_daily_analysis": 10,
            "content_analysis": True,
            "keyword_analysis": True,
            "structure_analysis": True,
            "readability_analysis": True,
            "eeat_analysis": False,
            "llm_analysis": False,
            "serp_analysis": False,
            "competitor_analysis": False,
            "premium_recommendations": False,
            "api_access": False,
            "team_collaboration": False,
            "white_label": False,
            "priority_support": False
        },
        TierPlan.BASIC.value: {
            "max_urls": 1000,
            "max_daily_analysis": 50,
            "content_analysis": True,
            "keyword_analysis": True,
            "structure_analysis": True,
            "readability_analysis": True,
            "eeat_analysis": True,
            "llm_analysis": False,
            "serp_analysis": True,
            "competitor_analysis": True,
            "premium_recommendations": False,
            "api_access": False,
            "team_collaboration": False,
            "white_label": False,
            "priority_support": False
        },
        TierPlan.PROFESSIONAL.value: {
            "max_urls": 10000,
            "max_daily_analysis": 200,
            "content_analysis": True,
            "keyword_analysis": True,
            "structure_analysis": True,
            "readability_analysis": True,
            "eeat_analysis": True,
            "llm_analysis": True,
            "serp_analysis": True,
            "competitor_analysis": True,
            "premium_recommendations": False,
            "api_access": True,
            "team_collaboration": True,
            "white_label": False,
            "priority_support": True
        },
        TierPlan.ENTERPRISE.value: {
            "max_urls": float('inf'),  # Неограниченно
            "max_daily_analysis": float('inf'),  # Неограниченно
            "content_analysis": True,
            "keyword_analysis": True,
            "structure_analysis": True,
            "readability_analysis": True,
            "eeat_analysis": True,
            "llm_analysis": True,
            "serp_analysis": True,
            "competitor_analysis": True,
            "premium_recommendations": True,
            "api_access": True,
            "team_collaboration": True,
            "white_label": True,
            "priority_support": True
        }
    }
    
    def __init__(self, user_id: str, tier: str = TierPlan.MICRO.value):
        """
        Инициализация системы управления доступом.
        
        Args:
            user_id: ID пользователя
            tier: Уровень подписки пользователя
        """
        self.user_id = user_id
        
        # Преобразование tier в строку, если это enum
        if isinstance(tier, TierPlan):
            self.tier = tier.value
        else:
            self.tier = tier.lower()
        
        # Получение доступных функций для уровня
        self.available_features = self.DEFAULT_FEATURES.get(
            self.tier, 
            self.DEFAULT_FEATURES[TierPlan.MICRO.value]
        )
    
    def is_feature_available(self, feature_name: str) -> bool:
        """
        Проверка доступности функции для текущего уровня подписки.
        
        Args:
            feature_name: Название функции
            
        Returns:
            bool: True, если функция доступна, False в противном случае
        """
        return self.available_features.get(feature_name, False)
    
    def get_limit(self, limit_name: str) -> Any:
        """
        Получение значения лимита для текущего уровня подписки.
        
        Args:
            limit_name: Название лимита
            
        Returns:
            Any: Значение лимита или None, если лимит не найден
        """
        return self.available_features.get(limit_name, None)
    
    def get_all_features(self) -> Dict[str, Any]:
        """
        Получение всех доступных функций и лимитов для текущего уровня подписки.
        
        Returns:
            Dict[str, Any]: Словарь доступных функций и лимитов
        """
        return self.available_features
    
    def upgrade_tier(self, new_tier: Union[TierPlan, str]) -> bool:
        """
        Обновление уровня подписки пользователя.
        
        Args:
            new_tier: Новый уровень подписки
            
        Returns:
            bool: True, если обновление успешно, False в противном случае
        """
        # Преобразование new_tier в строку, если это enum
        if isinstance(new_tier, TierPlan):
            new_tier = new_tier.value
        else:
            new_tier = new_tier.lower()
        
        # Проверка существования уровня
        if new_tier not in self.DEFAULT_FEATURES:
            return False
        
        # Обновление уровня и доступных функций
        self.tier = new_tier
        self.available_features = self.DEFAULT_FEATURES[new_tier]
        
        return True
