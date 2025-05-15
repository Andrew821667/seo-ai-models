"""
Управление доступом к функциям в зависимости от плана.

Модуль предоставляет функциональность для ограничения доступа к определенным
возможностям и функциям в зависимости от выбранного плана использования.
"""

import logging
from enum import Enum
from typing import Dict, List, Set, Any, Optional


class TierPlan(Enum):
    """Планы использования многоуровневой системы."""
    MICRO = "micro"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class FeatureGating:
    """
    Управление доступом к функциям в зависимости от плана.
    
    Класс отвечает за определение доступных функций и возможностей
    для каждого плана использования.
    """
    
    # Определяем доступные функции для каждого плана
    FEATURE_SETS = {
        'micro': {
            # Базовые функции для микро-плана
            'basic_content_analysis',
            'basic_keyword_analysis',
            'readability_analysis',
            'basic_structure_analysis',
            'basic_recommendations',
        },
        'basic': {
            # Функции для базового плана (включая все функции микро-плана)
            'basic_content_analysis',
            'basic_keyword_analysis',
            'readability_analysis',
            'basic_structure_analysis',
            'basic_recommendations',
            'advanced_content_analysis',
            'enhanced_keyword_analysis',
            'semantic_analysis',
            'eeat_analysis',
            'enhanced_structure_analysis',
            'comprehensive_recommendations',
            'limited_llm_analysis',  # Ограниченный LLM-анализ
        },
        'professional': {
            # Функции для профессионального плана (включая все функции базового плана)
            'basic_content_analysis',
            'basic_keyword_analysis',
            'readability_analysis',
            'basic_structure_analysis',
            'basic_recommendations',
            'advanced_content_analysis',
            'enhanced_keyword_analysis',
            'semantic_analysis',
            'eeat_analysis',
            'enhanced_structure_analysis',
            'comprehensive_recommendations',
            'full_llm_analysis',      # Полный LLM-анализ
            'llm_compatibility',      # Совместимость с LLM
            'citability_scoring',     # Оценка цитируемости
            'content_structure_enhancement',  # Улучшение структуры
            'llm_eeat_analysis',     # E-E-A-T для LLM
            'competitive_analysis',  # Анализ конкурентов
            'content_optimization',  # Оптимизация контента
        },
        'enterprise': {
            # Все доступные функции (включая все функции профессионального плана)
            'basic_content_analysis',
            'basic_keyword_analysis',
            'readability_analysis',
            'basic_structure_analysis',
            'basic_recommendations',
            'advanced_content_analysis',
            'enhanced_keyword_analysis',
            'semantic_analysis',
            'eeat_analysis',
            'enhanced_structure_analysis',
            'comprehensive_recommendations',
            'full_llm_analysis',
            'llm_compatibility',
            'citability_scoring',
            'content_structure_enhancement',
            'llm_eeat_analysis',
            'competitive_analysis',
            'content_optimization',
            'multi_model_analysis',   # Анализ с несколькими моделями
            'feature_importance_analysis',  # Анализ важности факторов
            'custom_llm_parameters',  # Настройка параметров LLM
            'custom_reports',         # Настраиваемые отчеты
            'bulk_analysis',          # Массовый анализ
            'api_access',             # Доступ через API
            'white_labeling',         # White labeling
        },
    }
    
    # Соответствие между аргументами функций и функциями
    FEATURE_ARGS_MAPPING = {
        'use_llm': {'limited_llm_analysis', 'full_llm_analysis'},
        'detailed_analysis': {'advanced_content_analysis'},
        'analyze_competitors': {'competitive_analysis'},
        'optimize_content': {'content_optimization'},
        'analyze_llm_compatibility': {'llm_compatibility'},
        'score_citability': {'citability_scoring'},
        'enhance_structure': {'content_structure_enhancement'},
        'analyze_llm_eeat': {'llm_eeat_analysis'},
        'use_multi_model': {'multi_model_analysis'},
        'analyze_feature_importance': {'feature_importance_analysis'},
        'customize_reports': {'custom_reports'},
        'bulk_mode': {'bulk_analysis'},
    }
    
    def __init__(self, tier: TierPlan, **kwargs):
        """
        Инициализирует управление доступом к функциям.
        
        Args:
            tier: План использования
            **kwargs: Дополнительные параметры
        """
        self.logger = logging.getLogger(__name__)
        self.tier = tier
        
        # Инициализируем набор доступных функций, используя tier.value вместо tier
        self.allowed_features = self.FEATURE_SETS[tier.value]
        
        # Добавляем дополнительные функции для beta-тестеров, если указано
        self.is_beta_tester = kwargs.get('is_beta_tester', False)
        if self.is_beta_tester:
            self._add_beta_features()
            
        self.logger.info(f"FeatureGating инициализирован для плана {tier.value}")
    
    def _add_beta_features(self) -> None:
        """Добавляет дополнительные функции для beta-тестеров."""
        # Определяем дополнительные бета-функции для каждого плана
        beta_features = {
            'micro': {'limited_llm_analysis'},
            'basic': {'llm_compatibility', 'citability_scoring'},
            'professional': {'multi_model_analysis', 'feature_importance_analysis'},
            'enterprise': {'experimental_features', 'early_access'},
        }
        
        # Добавляем бета-функции для текущего плана
        if self.tier.value in beta_features:
            self.allowed_features.update(beta_features[self.tier.value])
            self.logger.info(f"Добавлены бета-функции для плана {self.tier.value}")
    
    def is_feature_allowed(self, feature_name: str) -> bool:
        """
        Проверяет, доступна ли указанная функция для текущего плана.
        
        Args:
            feature_name: Название функции
            
        Returns:
            True, если функция доступна
        """
        return feature_name in self.allowed_features
    
    def get_allowed_features(self) -> Set[str]:
        """
        Возвращает набор доступных функций для текущего плана.
        
        Returns:
            Набор доступных функций
        """
        return self.allowed_features
    
    def filter_allowed_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Фильтрует аргументы, оставляя только те, что соответствуют доступным функциям.
        
        Args:
            kwargs: Исходные аргументы
            
        Returns:
            Отфильтрованные аргументы
        """
        filtered_kwargs = {}
        
        for arg_name, arg_value in kwargs.items():
            # Проверяем, связан ли аргумент с какой-либо функцией
            if arg_name in self.FEATURE_ARGS_MAPPING:
                # Получаем связанные функции
                related_features = self.FEATURE_ARGS_MAPPING[arg_name]
                
                # Проверяем, есть ли хотя бы одна доступная функция
                if any(feature in self.allowed_features for feature in related_features):
                    filtered_kwargs[arg_name] = arg_value
            else:
                # Если аргумент не связан с функциями, оставляем его
                filtered_kwargs[arg_name] = arg_value
                
        return filtered_kwargs
    
    def update_tier(self, new_tier: TierPlan) -> None:
        """
        Обновляет план использования.
        
        Args:
            new_tier: Новый план
        """
        self.tier = new_tier
        self.allowed_features = self.FEATURE_SETS[new_tier.value].copy()
        
        if self.is_beta_tester:
            self._add_beta_features()
            
        self.logger.info(f"FeatureGating обновлен до плана {new_tier.value}")
