import torch
import torch.nn as nn
from typing import Dict, Union, List
import logging

from ...config.advisor_config import AdvisorConfig
from .content_analyzer import ContentAnalyzer
from .rank_predictor import RankPredictor
from .suggester import OptimizationSuggester

logger = logging.getLogger(__name__)

class SEOAdvisor(nn.Module):
    """Основная модель SEO-советника"""
    def __init__(self, config: AdvisorConfig):
        super().__init__()
        self.config = config
        self.content_analyzer = ContentAnalyzer(config)
        
        # Слой объединения признаков
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.content_dim + config.metrics_dim, config.content_dim),
            nn.LayerNorm(config.content_dim),
            nn.ReLU()
        )
        
        # Компоненты анализа и рекомендаций
        self.rank_predictor = RankPredictor(config.content_dim)
        self.optimization_suggester = OptimizationSuggester(
            config.content_dim,
            config.num_suggestions
        )
        
        # Механизм внимания
        self.attention = nn.MultiheadAttention(config.content_dim, num_heads=8)
        
        if config.use_cache:
            from ...utils.caching import setup_cache
            self.cache = setup_cache(config.redis_url)
            
    def forward(self, content: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Прямой проход модели
        Args:
            content: входной контент
        Returns:
            результаты анализа и рекомендации
        """
        # Проверка кэша
        if self.config.use_cache:
            from ...utils.caching import get_cached_result
            cached_result = get_cached_result(self.cache, content)
            if cached_result:
                return cached_result
        
        # Анализ контента
        content_features = self.content_analyzer(content)
        
        # Объединение признаков
        combined_features = self.feature_fusion(content_features)
        
        # Применение механизма внимания
        attended_features, attention_weights = self.attention(
            combined_features,
            combined_features,
            combined_features
        )
        
        # Получение предсказаний и рекомендаций
        ranking_score = self.rank_predictor(attended_features)
        optimization_outputs = self.optimization_suggester(attended_features)
        
        result = {
            'ranking_score': ranking_score,
            'suggestions': optimization_outputs['suggestions'],
            'suggestion_importance': optimization_outputs['importance_scores'],
            'attention_weights': attention_weights
        }
        
        # Сохранение в кэш
        if self.config.use_cache:
            from ...utils.caching import cache_result
            cache_result(self.cache, content, result)
        
        return result
