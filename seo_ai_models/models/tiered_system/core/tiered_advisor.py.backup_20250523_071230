# -*- coding: utf-8 -*-
"""
TieredAdvisor - Многоуровневый советник для SEO.
Обеспечивает различные уровни функциональности в зависимости от плана.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class TierPlan:
    """Планы многоуровневой системы."""
    MICRO = "micro"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class ResourceOptimizer:
    """
    Оптимизирует использование ресурсов в зависимости от выбранного уровня.
    
    Управляет использованием дорогостоящих ресурсов (LLM API, вычислительные
    ресурсы и т.д.) для разных уровней плана.
    """
    
    # Коэффициенты оптимизации для разных уровней
    OPTIMIZATION_FACTORS = {
        TierPlan.MICRO: {
            "llm_token_limit": 10000,
            "llm_model": "basic",
            "analysis_depth": 0.4,
            "request_throttling": 5,  # запросов в минуту
            "max_content_size": 50000  # символов
        },
        TierPlan.BASIC: {
            "llm_token_limit": 50000,
            "llm_model": "standard",
            "analysis_depth": 0.7,
            "request_throttling": 15,  # запросов в минуту
            "max_content_size": 150000  # символов
        },
        TierPlan.PROFESSIONAL: {
            "llm_token_limit": 200000,
            "llm_model": "advanced",
            "analysis_depth": 0.9,
            "request_throttling": 40,  # запросов в минуту
            "max_content_size": 500000  # символов
        },
        TierPlan.ENTERPRISE: {
            "llm_token_limit": None,  # Без ограничений
            "llm_model": "premium",
            "analysis_depth": 1.0,
            "request_throttling": 100,  # запросов в минуту
            "max_content_size": None  # Без ограничений
        }
    }
    
    def __init__(
        self,
        tier: Union[str, TierPlan],
        user_id: str,
        **kwargs
    ):
        """
        Инициализирует ResourceOptimizer.
        
        Args:
            tier: Уровень плана
            user_id: Идентификатор пользователя
            **kwargs: Дополнительные аргументы
        """
        self.tier = tier
        self.user_id = user_id
        self.optimization_factors = self._get_optimization_factors(tier)
        
        # Статистика использования ресурсов
        self.usage_stats = {
            "llm_tokens_used": 0,
            "requests_processed": 0,
            "content_bytes_processed": 0,
            "last_reset": datetime.now().isoformat()
        }
    
    def _get_optimization_factors(self, tier: Union[str, TierPlan]) -> Dict[str, Any]:
        """
        Возвращает коэффициенты оптимизации для указанного уровня.
        
        Args:
            tier: Уровень плана
            
        Returns:
            Коэффициенты оптимизации
        """
        if isinstance(tier, str):
            tier_str = tier
        else:
            tier_str = tier
        
        return self.OPTIMIZATION_FACTORS.get(tier_str, self.OPTIMIZATION_FACTORS[TierPlan.MICRO])
    
    def optimize_request(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизирует параметры запроса в зависимости от уровня.
        
        Args:
            request_params: Параметры запроса
            
        Returns:
            Оптимизированные параметры запроса
        """
        optimized_params = copy.deepcopy(request_params)
        
        # Оптимизация модели LLM
        if "llm_model" in optimized_params:
            tier_model = self.optimization_factors["llm_model"]
            # Проверяем, соответствует ли запрошенная модель ограничениям уровня
            requested_model = optimized_params["llm_model"]
            model_ranking = {"basic": 1, "standard": 2, "advanced": 3, "premium": 4}
            
            if model_ranking.get(requested_model, 0) > model_ranking.get(tier_model, 0):
                # Понижаем модель до разрешенной
                optimized_params["llm_model"] = tier_model
                logger.info(f"Model downgraded from {requested_model} to {tier_model} for tier {self.tier}")
        
        # Оптимизация глубины анализа
        if "analysis_depth" in optimized_params:
            tier_depth = self.optimization_factors["analysis_depth"]
            requested_depth = float(optimized_params["analysis_depth"])
            
            if requested_depth > tier_depth:
                # Понижаем глубину анализа до разрешенной
                optimized_params["analysis_depth"] = tier_depth
                logger.info(f"Analysis depth reduced from {requested_depth} to {tier_depth} for tier {self.tier}")
        
        # Оптимизация максимального размера контента
        if "content" in optimized_params:
            max_size = self.optimization_factors["max_content_size"]
            
            if max_size is not None and len(optimized_params["content"]) > max_size:
                # Обрезаем контент до максимального размера
                optimized_params["content"] = optimized_params["content"][:max_size]
                logger.info(f"Content truncated to {max_size} characters for tier {self.tier}")
        
        return optimized_params
    
    def check_token_limit(self, token_count: int) -> bool:
        """
        Проверяет, не превышен ли лимит токенов.
        
        Args:
            token_count: Количество токенов
            
        Returns:
            True, если лимит не превышен, иначе False
        """
        token_limit = self.optimization_factors["llm_token_limit"]
        
        if token_limit is None:
            return True
        
        return self.usage_stats["llm_tokens_used"] + token_count <= token_limit
    
    def update_usage_stats(self, tokens_used: int, content_size: int = 0):
        """
        Обновляет статистику использования ресурсов.
        
        Args:
            tokens_used: Количество использованных токенов
            content_size: Размер обработанного контента
        """
        self.usage_stats["llm_tokens_used"] += tokens_used
        self.usage_stats["requests_processed"] += 1
        self.usage_stats["content_bytes_processed"] += content_size
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования ресурсов.
        
        Returns:
            Статистика использования ресурсов
        """
        return self.usage_stats
    
    def reset_usage_stats(self):
        """Сбрасывает статистику использования ресурсов."""
        self.usage_stats = {
            "llm_tokens_used": 0,
            "requests_processed": 0,
            "content_bytes_processed": 0,
            "last_reset": datetime.now().isoformat()
        }
    
    def get_tier_limits(self) -> Dict[str, Any]:
        """
        Возвращает ограничения уровня.
        
        Returns:
            Ограничения уровня
        """
        return self.optimization_factors

class FeatureGating:
    """
    Управляет доступом к функциям в зависимости от уровня.
    
    Определяет, какие функции доступны для каждого уровня плана.
    """
    
    # Доступность функций для разных уровней
    FEATURE_AVAILABILITY = {
        "basic_analysis": {
            TierPlan.MICRO: True,
            TierPlan.BASIC: True,
            TierPlan.PROFESSIONAL: True,
            TierPlan.ENTERPRISE: True
        },
        "advanced_analysis": {
            TierPlan.MICRO: False,
            TierPlan.BASIC: True,
            TierPlan.PROFESSIONAL: True,
            TierPlan.ENTERPRISE: True
        },
        "bulk_analysis": {
            TierPlan.MICRO: False,
            TierPlan.BASIC: True,
            TierPlan.PROFESSIONAL: True,
            TierPlan.ENTERPRISE: True
        },
        "competitor_analysis": {
            TierPlan.MICRO: False,
            TierPlan.BASIC: True,
            TierPlan.PROFESSIONAL: True,
            TierPlan.ENTERPRISE: True
        },
        "llm_integration": {
            TierPlan.MICRO: False,
            TierPlan.BASIC: False,
            TierPlan.PROFESSIONAL: True,
            TierPlan.ENTERPRISE: True
        },
        "api_access": {
            TierPlan.MICRO: False,
            TierPlan.BASIC: False,
            TierPlan.PROFESSIONAL: True,
            TierPlan.ENTERPRISE: True
        },
        "custom_metrics": {
            TierPlan.MICRO: False,
            TierPlan.BASIC: False,
            TierPlan.PROFESSIONAL: True,
            TierPlan.ENTERPRISE: True
        }
    }
    
    def __init__(self, tier: Union[str, TierPlan]):
        """
        Инициализирует FeatureGating.
        
        Args:
            tier: Уровень плана
        """
        self.tier = tier
    
    def is_feature_available(self, feature_name: str) -> bool:
        """
        Проверяет, доступна ли функция для текущего уровня.
        
        Args:
            feature_name: Имя функции
            
        Returns:
            True, если функция доступна, иначе False
        """
        if feature_name not in self.FEATURE_AVAILABILITY:
            # Неизвестная функция считается недоступной
            return False
        
        return self.FEATURE_AVAILABILITY[feature_name].get(self.tier, False)
    
    def get_available_features(self) -> Dict[str, bool]:
        """
        Возвращает список доступных функций для текущего уровня.
        
        Returns:
            Словарь с доступностью функций
        """
        available_features = {}
        
        for feature_name, availability in self.FEATURE_AVAILABILITY.items():
            available_features[feature_name] = availability.get(self.tier, False)
        
        return available_features
    
    def get_feature_tiers(self, feature_name: str) -> List[str]:
        """
        Возвращает список уровней, для которых доступна указанная функция.
        
        Args:
            feature_name: Имя функции
            
        Returns:
            Список уровней
        """
        if feature_name not in self.FEATURE_AVAILABILITY:
            return []
        
        return [
            tier for tier, available in self.FEATURE_AVAILABILITY[feature_name].items()
            if available
        ]

class TieredAdvisor:
    """
    Многоуровневый советник для SEO.
    
    Обеспечивает различные уровни функциональности в зависимости от плана,
    оптимизируя использование ресурсов и предоставляя доступ к функциям
    в соответствии с выбранным уровнем.
    """
    
    def __init__(
        self,
        tier: Union[str, TierPlan],
        config: Optional[Dict[str, Any]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        optimize_resources: bool = True,
        **kwargs
    ):
        """
        Инициализирует TieredAdvisor.
        
        Args:
            tier: Уровень плана
            config: Конфигурация
            api_keys: API-ключи для внешних сервисов
            optimize_resources: Флаг оптимизации ресурсов
            **kwargs: Дополнительные аргументы
        """
        self.tier = tier
        self.config = config or {}
        self.api_keys = api_keys or {}
        self.optimize_resources = optimize_resources
        
        # Получаем идентификатор пользователя из kwargs или используем значение по умолчанию
        self.user_id = kwargs.get("user_id", "anonymous")
        
        # Создаем ResourceOptimizer, если включена оптимизация ресурсов
        self.resource_optimizer = ResourceOptimizer(
            tier=self.tier,
            user_id=self.user_id,
            **kwargs
        ) if self.optimize_resources else None
        
        # Создаем FeatureGating
        self.feature_gating = FeatureGating(tier=self.tier)
        
        # Настройки анализа
        self.analysis_settings = self._init_analysis_settings()
        
        # Настройки отчетов
        self.report_settings = self._init_report_settings()
    
    def _init_analysis_settings(self) -> Dict[str, Any]:
        """
        Инициализирует настройки анализа в зависимости от уровня.
        
        Returns:
            Настройки анализа
        """
        # Базовые настройки
        base_settings = {
            "word_count": True,
            "readability": True,
            "keyword_density": True,
            "heading_analysis": True,
            "sentiment_analysis": False,
            "semantic_analysis": False,
            "entity_recognition": False,
            "competitor_analysis": False,
            "llm_optimization": False,
            "custom_metrics": False
        }
        
        # Настройки для разных уровней
        tier_settings = {
            TierPlan.MICRO: {
                "sentiment_analysis": False,
                "semantic_analysis": False,
                "entity_recognition": False,
                "competitor_analysis": False,
                "llm_optimization": False,
                "custom_metrics": False
            },
            TierPlan.BASIC: {
                "sentiment_analysis": True,
                "semantic_analysis": True,
                "entity_recognition": False,
                "competitor_analysis": True,
                "llm_optimization": False,
                "custom_metrics": False
            },
            TierPlan.PROFESSIONAL: {
                "sentiment_analysis": True,
                "semantic_analysis": True,
                "entity_recognition": True,
                "competitor_analysis": True,
                "llm_optimization": True,
                "custom_metrics": False
            },
            TierPlan.ENTERPRISE: {
                "sentiment_analysis": True,
                "semantic_analysis": True,
                "entity_recognition": True,
                "competitor_analysis": True,
                "llm_optimization": True,
                "custom_metrics": True
            }
        }
        
        # Обновляем базовые настройки настройками уровня
        if self.tier in tier_settings:
            base_settings.update(tier_settings[self.tier])
        
        return base_settings
    
    def _init_report_settings(self) -> Dict[str, Any]:
        """
        Инициализирует настройки отчетов в зависимости от уровня.
        
        Returns:
            Настройки отчетов
        """
        # Базовые настройки
        base_settings = {
            "include_summary": True,
            "include_recommendations": True,
            "include_metrics": True,
            "include_charts": False,
            "include_competitor_analysis": False,
            "include_llm_insights": False,
            "include_custom_metrics": False,
            "max_recommendations": 5
        }
        
        # Настройки для разных уровней
        tier_settings = {
            TierPlan.MICRO: {
                "include_charts": False,
                "include_competitor_analysis": False,
                "include_llm_insights": False,
                "include_custom_metrics": False,
                "max_recommendations": 5
            },
            TierPlan.BASIC: {
                "include_charts": True,
                "include_competitor_analysis": True,
                "include_llm_insights": False,
                "include_custom_metrics": False,
                "max_recommendations": 10
            },
            TierPlan.PROFESSIONAL: {
                "include_charts": True,
                "include_competitor_analysis": True,
                "include_llm_insights": True,
                "include_custom_metrics": False,
                "max_recommendations": 20
            },
            TierPlan.ENTERPRISE: {
                "include_charts": True,
                "include_competitor_analysis": True,
                "include_llm_insights": True,
                "include_custom_metrics": True,
                "max_recommendations": -1  # Без ограничений
            }
        }
        
        # Обновляем базовые настройки настройками уровня
        if self.tier in tier_settings:
            base_settings.update(tier_settings[self.tier])
        
        return base_settings
    
    def analyze_content(self, content: str, **kwargs) -> Dict[str, Any]:
        """
        Анализирует контент с учетом уровня плана.
        
        Args:
            content: Текст для анализа
            **kwargs: Дополнительные параметры
            
        Returns:
            Результаты анализа
        """
        # Проверяем, доступна ли функция анализа контента для текущего уровня
        if not self.feature_gating.is_feature_available("basic_analysis"):
            return {
                "error": "Feature not available",
                "message": "Content analysis is not available for your plan"
            }
        
        # Оптимизируем параметры запроса, если включена оптимизация ресурсов
        request_params = {"content": content, **kwargs}
        
        if self.optimize_resources:
            request_params = self.resource_optimizer.optimize_request(request_params)
        
        # Выполняем анализ
        result = self._perform_analysis(request_params)
        
        # Обновляем статистику использования ресурсов
        if self.optimize_resources:
            self.resource_optimizer.update_usage_stats(
                tokens_used=len(content.split()),
                content_size=len(content)
            )
        
        return result
    
    def _perform_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет анализ контента.
        
        Args:
            params: Параметры запроса
            
        Returns:
            Результаты анализа
        """
        # Извлекаем контент из параметров
        content = params.get("content", "")
        
        # Производим базовый анализ
        basic_metrics = {
            "word_count": len(content.split()),
            "character_count": len(content),
            "read_time": len(content.split()) / 200  # Примерно 200 слов в минуту
        }
        
        # Добавляем анализ читабельности, если он включен
        if self.analysis_settings["readability"]:
            basic_metrics["readability"] = self._calculate_readability(content)
        
        # Добавляем анализ ключевых слов, если он включен
        if self.analysis_settings["keyword_density"]:
            basic_metrics["keyword_density"] = self._calculate_keyword_density(content)
        
        # Добавляем анализ заголовков, если он включен
        if self.analysis_settings["heading_analysis"]:
            basic_metrics["heading_analysis"] = self._analyze_headings(content)
        
        # Создаем результат
        result = {
            "basic_metrics": basic_metrics
        }
        
        # Добавляем расширенный анализ, если он доступен для текущего уровня
        if self.analysis_settings["sentiment_analysis"] and self.feature_gating.is_feature_available("advanced_analysis"):
            result["sentiment_analysis"] = self._analyze_sentiment(content)
        
        if self.analysis_settings["semantic_analysis"] and self.feature_gating.is_feature_available("advanced_analysis"):
            result["semantic_analysis"] = self._analyze_semantics(content)
        
        if self.analysis_settings["entity_recognition"] and self.feature_gating.is_feature_available("advanced_analysis"):
            result["entity_recognition"] = self._recognize_entities(content)
        
        # Добавляем анализ конкурентов, если он доступен для текущего уровня
        if self.analysis_settings["competitor_analysis"] and self.feature_gating.is_feature_available("competitor_analysis"):
            keyword = params.get("keyword", "")
            
            if keyword:
                result["competitor_analysis"] = self._analyze_competitors(content, keyword)
        
        # Добавляем оптимизацию для LLM, если она доступна для текущего уровня
        if self.analysis_settings["llm_optimization"] and self.feature_gating.is_feature_available("llm_integration"):
            result["llm_optimization"] = self._optimize_for_llm(content)
        
        # Добавляем пользовательские метрики, если они доступны для текущего уровня
        if self.analysis_settings["custom_metrics"] and self.feature_gating.is_feature_available("custom_metrics"):
            custom_metrics = params.get("custom_metrics", [])
            
            if custom_metrics:
                result["custom_metrics"] = self._calculate_custom_metrics(content, custom_metrics)
        
        # Добавляем рекомендации
        result["recommendations"] = self._generate_recommendations(result)
        
        return result
    
    def _calculate_readability(self, content: str) -> Dict[str, Any]:
        """
        Рассчитывает метрики читабельности.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Метрики читабельности
        """
        # Простая заглушка для демонстрации
        words = content.split()
        sentences = content.split('.')
        avg_words_per_sentence = len(words) / max(1, len(sentences))
        
        # Flesch Reading Ease (примерно)
        flesch_ease = max(0, min(100, 206.835 - 1.015 * avg_words_per_sentence - 84.6 * (sum(len(word) for word in words) / max(1, len(words)) / 100)))
        
        return {
            "avg_words_per_sentence": avg_words_per_sentence,
            "flesch_reading_ease": flesch_ease,
            "readability_level": "Easy" if flesch_ease > 70 else "Medium" if flesch_ease > 50 else "Difficult"
        }
    
    def _calculate_keyword_density(self, content: str) -> Dict[str, float]:
        """
        Рассчитывает плотность ключевых слов.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Плотность ключевых слов
        """
        # Простая заглушка для демонстрации
        words = [word.lower() for word in content.split()]
        total_words = len(words)
        
        # Считаем частоту слов
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Игнорируем короткие слова
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Рассчитываем плотность ключевых слов
        keyword_density = {}
        
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            keyword_density[word] = round(freq / total_words * 100, 2)
        
        return keyword_density
    
    def _analyze_headings(self, content: str) -> Dict[str, Any]:
        """
        Анализирует заголовки в тексте.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Результаты анализа заголовков
        """
        # Простая заглушка для демонстрации
        lines = content.split('\n')
        headings = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('# '):
                headings["h1"] = headings.get("h1", 0) + 1
            elif line.startswith('## '):
                headings["h2"] = headings.get("h2", 0) + 1
            elif line.startswith('### '):
                headings["h3"] = headings.get("h3", 0) + 1
            elif line.startswith('#### '):
                headings["h4"] = headings.get("h4", 0) + 1
            elif line.startswith('##### '):
                headings["h5"] = headings.get("h5", 0) + 1
            elif line.startswith('###### '):
                headings["h6"] = headings.get("h6", 0) + 1
        
        # Анализируем структуру заголовков
        analysis = {
            "headings_count": sum(headings.values()),
            "headings_distribution": headings,
            "has_h1": headings.get("h1", 0) > 0,
            "has_proper_structure": True  # Заглушка
        }
        
        return analysis
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """
        Анализирует тональность текста.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Результаты анализа тональности
        """
        # Простая заглушка для демонстрации
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.2,
            "positive_phrases": 5,
            "negative_phrases": 3,
            "neutral_phrases": 12
        }
    
    def _analyze_semantics(self, content: str) -> Dict[str, Any]:
        """
        Анализирует семантику текста.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Результаты семантического анализа
        """
        # Простая заглушка для демонстрации
        return {
            "semantic_depth": 0.75,
            "topic_relevance": 0.8,
            "topic_coverage": 0.7,
            "main_topics": ["topic1", "topic2", "topic3"]
        }
    
    def _recognize_entities(self, content: str) -> Dict[str, Any]:
        """
        Распознает именованные сущности в тексте.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Результаты распознавания сущностей
        """
        # Простая заглушка для демонстрации
        return {
            "entities": {
                "persons": ["John Doe", "Jane Smith"],
                "organizations": ["Acme Corp", "Example Inc"],
                "locations": ["New York", "London"],
                "dates": ["2023-01-01", "2023-12-31"]
            },
            "entity_count": 8
        }
    
    def _analyze_competitors(self, content: str, keyword: str) -> Dict[str, Any]:
        """
        Анализирует контент конкурентов.
        
        Args:
            content: Текст для анализа
            keyword: Ключевое слово для поиска конкурентов
            
        Returns:
            Результаты анализа конкурентов
        """
        # Простая заглушка для демонстрации
        return {
            "competitors": ["competitor1.com", "competitor2.com", "competitor3.com"],
            "competition_level": "high",
            "word_count_average": 1500,
            "content_gap": ["topic1", "topic2"],
            "content_advantages": ["topic3", "topic4"],
            "recommendations": [
                "Add more content about topic1",
                "Cover topic2 in more detail"
            ]
        }
    
    def _optimize_for_llm(self, content: str) -> Dict[str, Any]:
        """
        Оптимизирует контент для LLM.
        
        Args:
            content: Текст для оптимизации
            
        Returns:
            Результаты оптимизации
        """
        # Простая заглушка для демонстрации
        return {
            "llm_citability_score": 0.85,
            "llm_optimization_score": 0.75,
            "enhancements": [
                "Add more structured data",
                "Include more specific examples",
                "Improve factual accuracy"
            ]
        }
    
    def _calculate_custom_metrics(self, content: str, custom_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Рассчитывает пользовательские метрики.
        
        Args:
            content: Текст для анализа
            custom_metrics: Список пользовательских метрик
            
        Returns:
            Результаты расчета пользовательских метрик
        """
        # Простая заглушка для демонстрации
        return {
            "custom_metric_1": 0.75,
            "custom_metric_2": 0.85,
            "custom_metric_3": 0.65
        }
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации на основе результатов анализа.
        
        Args:
            analysis_results: Результаты анализа
            
        Returns:
            Список рекомендаций
        """
        # Простая заглушка для демонстрации
        recommendations = []
        basic_metrics = analysis_results.get("basic_metrics", {})
        
        # Рекомендации по длине контента
        word_count = basic_metrics.get("word_count", 0)
        
        if word_count < 300:
            recommendations.append({
                "type": "content_length",
                "severity": "high",
                "title": "Increase content length",
                "description": "Your content is too short. Consider adding more information to reach at least 300 words."
            })
        
        # Рекомендации по читабельности
        if "readability" in basic_metrics:
            readability = basic_metrics["readability"]
            if readability.get("readability_level") == "Difficult":
                recommendations.append({
                    "type": "readability",
                    "severity": "medium",
                    "title": "Improve readability",
                    "description": "Your content is difficult to read. Consider using shorter sentences and simpler words."
                })
        
        # Рекомендации по заголовкам
        if "heading_analysis" in basic_metrics:
            heading_analysis = basic_metrics["heading_analysis"]
            if not heading_analysis.get("has_h1", False):
                recommendations.append({
                    "type": "headings",
                    "severity": "high",
                    "title": "Add H1 heading",
                    "description": "Your content doesn't have an H1 heading. Add one to improve structure and SEO."
                })
        
        # Добавляем рекомендации из анализа конкурентов, если он доступен
        if "competitor_analysis" in analysis_results:
            competitor_recommendations = analysis_results["competitor_analysis"].get("recommendations", [])
            
            for rec in competitor_recommendations:
                recommendations.append({
                    "type": "competitor",
                    "severity": "medium",
                    "title": rec,
                    "description": f"Based on competitor analysis: {rec}"
                })
        
        # Добавляем рекомендации из оптимизации для LLM, если она доступна
        if "llm_optimization" in analysis_results:
            llm_enhancements = analysis_results["llm_optimization"].get("enhancements", [])
            
            for enh in llm_enhancements:
                recommendations.append({
                    "type": "llm",
                    "severity": "medium",
                    "title": enh,
                    "description": f"For better LLM optimization: {enh}"
                })
        
        # Ограничиваем количество рекомендаций в соответствии с настройками отчетов
        max_recommendations = self.report_settings["max_recommendations"]
        
        if max_recommendations > 0:
            recommendations = recommendations[:max_recommendations]
        
        return recommendations
    
    def get_available_features(self) -> Dict[str, bool]:
        """
        Возвращает список доступных функций для текущего уровня.
        
        Returns:
            Словарь с доступностью функций
        """
        return self.feature_gating.get_available_features()
    
    def get_tier_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущем уровне.
        
        Returns:
            Информация о текущем уровне
        """
        return {
            "tier": self.tier,
            "available_features": self.get_available_features(),
            "analysis_settings": self.analysis_settings,
            "report_settings": self.report_settings
        }
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Возвращает информацию об использовании ресурсов.
        
        Returns:
            Информация об использовании ресурсов
        """
        if self.optimize_resources:
            return {
                "tier": self.tier,
                "usage_stats": self.resource_optimizer.get_usage_stats(),
                "tier_limits": self.resource_optimizer.get_tier_limits()
            }
        
        return {
            "tier": self.tier,
            "usage_stats": None,
            "tier_limits": None
        }
