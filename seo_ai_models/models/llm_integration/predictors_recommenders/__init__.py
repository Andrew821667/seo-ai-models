"""
Пакет для улучшенных предикторов и рекомендаций.

Пакет предоставляет функционал для предсказания ранжирования в LLM-поисковиках,
генерации рекомендаций, оптимизированных для обоих типов поиска, расчета ROI
от внедрения рекомендаций и создания плана действий с приоритизацией.
"""

from .llm_rank_predictor import LLMRankPredictor
from .hybrid_recommender import HybridRecommender
from .roi_calculator import ROICalculator
from .prioritized_action_plan import PrioritizedActionPlan

__all__ = [
    'LLMRankPredictor',
    'HybridRecommender',
    'ROICalculator',
    'PrioritizedActionPlan'
]
