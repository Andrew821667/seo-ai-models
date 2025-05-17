"""
Тестовый скрипт для компонентов улучшенных предикторов и рекомендаций.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import logging

# Отключаем логирование во время тестов
logging.disable(logging.CRITICAL)

class TestPredictorsRecommenders(unittest.TestCase):
    """Тесты для компонентов улучшенных предикторов и рекомендаций."""
    
    def test_imports(self):
        """Проверка корректности импорта компонентов."""
        # Проверяем импорт всех компонентов
        from seo_ai_models.models.llm_integration.predictors_recommenders import (
            LLMRankPredictor,
            HybridRecommender,
            ROICalculator,
            PrioritizedActionPlan
        )
        
        # Если импорт прошел успешно, тест пройден
        self.assertTrue(True)
    
    def test_llm_rank_predictor_init(self):
        """Проверка инициализации LLMRankPredictor."""
        from seo_ai_models.models.llm_integration.predictors_recommenders import LLMRankPredictor
        from seo_ai_models.models.llm_integration.service.llm_service import LLMService
        from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
        
        # Создаем моки
        llm_service = MagicMock(spec=LLMService)
        prompt_generator = MagicMock(spec=PromptGenerator)
        
        # Создаем экземпляр LLMRankPredictor
        rank_predictor = LLMRankPredictor(llm_service, prompt_generator)
        
        # Проверяем атрибуты
        self.assertEqual(rank_predictor.llm_service, llm_service)
        self.assertEqual(rank_predictor.prompt_generator, prompt_generator)
        self.assertTrue(hasattr(rank_predictor, 'default_ranking_weights'))
    
    def test_hybrid_recommender_init(self):
        """Проверка инициализации HybridRecommender."""
        from seo_ai_models.models.llm_integration.predictors_recommenders import HybridRecommender
        from seo_ai_models.models.llm_integration.service.llm_service import LLMService
        from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
        
        # Создаем моки
        llm_service = MagicMock(spec=LLMService)
        prompt_generator = MagicMock(spec=PromptGenerator)
        
        # Создаем экземпляр HybridRecommender
        hybrid_recommender = HybridRecommender(llm_service, prompt_generator)
        
        # Проверяем атрибуты
        self.assertEqual(hybrid_recommender.llm_service, llm_service)
        self.assertEqual(hybrid_recommender.prompt_generator, prompt_generator)
        self.assertTrue(hasattr(hybrid_recommender, 'recommendation_types'))
    
    def test_roi_calculator_init(self):
        """Проверка инициализации ROICalculator."""
        from seo_ai_models.models.llm_integration.predictors_recommenders import ROICalculator, LLMRankPredictor
        from seo_ai_models.models.llm_integration.service.llm_service import LLMService
        from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
        
        # Создаем моки
        llm_service = MagicMock(spec=LLMService)
        prompt_generator = MagicMock(spec=PromptGenerator)
        rank_predictor = MagicMock(spec=LLMRankPredictor)
        
        # Создаем экземпляр ROICalculator
        roi_calculator = ROICalculator(llm_service, prompt_generator, rank_predictor)
        
        # Проверяем атрибуты
        self.assertEqual(roi_calculator.llm_service, llm_service)
        self.assertEqual(roi_calculator.prompt_generator, prompt_generator)
        self.assertEqual(roi_calculator.rank_predictor, rank_predictor)
        self.assertTrue(hasattr(roi_calculator, 'default_traffic_increase_rates'))
    
    def test_prioritized_action_plan_init(self):
        """Проверка инициализации PrioritizedActionPlan."""
        from seo_ai_models.models.llm_integration.predictors_recommenders import (
            PrioritizedActionPlan,
            ROICalculator,
            HybridRecommender
        )
        from seo_ai_models.models.llm_integration.service.llm_service import LLMService
        from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
        
        # Создаем моки
        llm_service = MagicMock(spec=LLMService)
        prompt_generator = MagicMock(spec=PromptGenerator)
        roi_calculator = MagicMock(spec=ROICalculator)
        hybrid_recommender = MagicMock(spec=HybridRecommender)
        
        # Создаем экземпляр PrioritizedActionPlan
        prioritized_action_plan = PrioritizedActionPlan(llm_service, prompt_generator, roi_calculator, hybrid_recommender)
        
        # Проверяем атрибуты
        self.assertEqual(prioritized_action_plan.llm_service, llm_service)
        self.assertEqual(prioritized_action_plan.prompt_generator, prompt_generator)
        self.assertEqual(prioritized_action_plan.roi_calculator, roi_calculator)
        self.assertEqual(prioritized_action_plan.hybrid_recommender, hybrid_recommender)
        self.assertTrue(hasattr(prioritized_action_plan, 'plan_phases'))

if __name__ == '__main__':
    unittest.main()
