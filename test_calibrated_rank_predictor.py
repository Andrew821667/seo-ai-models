import unittest
import sys
import os
sys.path.append('/content/seo-ai-models')

from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor

class TestCalibratedRankPredictor(unittest.TestCase):
    
    def setUp(self):
        self.predictor = CalibratedRankPredictor()
        self.test_features = {
            'keyword_density': 0.05,
            'content_length': 0.7,
            'readability_score': 0.8,
            'meta_tags_score': 0.6,
            'header_structure_score': 0.9,
            'multimedia_score': 0.5,
            'internal_linking_score': 0.7,
            'topic_relevance': 0.8,
            'semantic_depth': 0.6,
            'engagement_potential': 0.7,
            'expertise_score': 0.5,
            'authority_score': 0.4,
            'trust_score': 0.6
        }
        # Отдельно задаем industry как строковую информацию для контекста
        self.industry = 'e-commerce'
    
    def test_predict_position(self):
        # Проверка предсказания позиции
        # Передаем только числовые характеристики, без строковых значений
        result = self.predictor.predict_position(self.test_features)
        
        # Проверяем, что результат - словарь
        self.assertIsInstance(result, dict)
        
        # Проверяем наличие ключевых полей
        self.assertIn('position', result)
        self.assertIn('feature_scores', result)
        
        # Проверяем, что позиция в разумном диапазоне (1-100)
        self.assertTrue(1 <= result['position'] <= 100, 
                       f"Позиция {result['position']} вне допустимого диапазона")
        
        # Проверяем, что feature_scores содержит нормированные оценки
        for feature, score in result['feature_scores'].items():
            self.assertTrue(0 <= score <= 1, 
                           f"Оценка для {feature} должна быть в диапазоне [0,1]")
    
    def test_predict_with_context(self):
        # Проверка предсказания с контекстом
        test_text = "Это тестовый текст о e-commerce с ключевыми словами."
        test_keywords = ["e-commerce", "ключевые слова"]
        
        # Передаем текст и ключевые слова как дополнительный контекст
        result = self.predictor.predict_position(self.test_features, text=test_text, keywords=test_keywords)
        
        # Проверяем, что результат получен
        self.assertIsInstance(result, dict)
        self.assertIn('position', result)
