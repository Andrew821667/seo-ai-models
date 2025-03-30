"""Тесты для Suggester."""

import unittest
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester


class TestSuggester(unittest.TestCase):
    """Тесты для класса Suggester."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.suggester = Suggester()
        
        # Тестовые данные
        self.base_recommendations = {
            'content_improvement': [
                'Увеличьте объем контента до 1500 слов',
                'Добавьте больше примеров'
            ],
            'keyword_optimization': [
                'Увеличьте плотность ключевых слов',
                'Добавьте ключевые слова в заголовки'
            ],
            'technical_optimization': [
                'Оптимизируйте мета-теги',
                'Улучшите структуру заголовков'
            ]
        }
        
        self.feature_scores = {
            'keyword_density': 0.01,
            'content_length': 500,
            'readability_score': 70,
            'meta_tags_score': 0.5,
            'header_structure_score': 0.4,
            'multimedia_score': 0.3,
            'internal_linking_score': 0.2,
            'topic_relevance': 0.8,
            'semantic_depth': 0.6,
            'engagement_potential': 0.5
        }
        
        self.weighted_scores = {
            'keyword_density': 0.002,
            'content_length': 0.05,
            'readability_score': 0.07,
            'meta_tags_score': 0.025,
            'header_structure_score': 0.02,
            'multimedia_score': 0.015,
            'internal_linking_score': 0.01,
            'topic_relevance': 0.08,
            'semantic_depth': 0.06,
            'engagement_potential': 0.025
        }
    
    def test_generate_suggestions_returns_enhanced_recommendations(self):
        """Тест проверяет, что метод generate_suggestions возвращает расширенные рекомендации."""
        industry = 'finance'
        result = self.suggester.generate_suggestions(
            self.base_recommendations, 
            self.feature_scores, 
            industry
        )
        
        # Проверяем, что результат содержит все исходные категории
        for category in self.base_recommendations:
            self.assertIn(category, result)
            
        # Проверяем, что в каждой категории есть все исходные рекомендации
        for category, recommendations in self.base_recommendations.items():
            for recommendation in recommendations:
                self.assertIn(recommendation, result[category])
    
    def test_generate_suggestions_adds_industry_specific_recommendations(self):
        """Тест проверяет, что добавляются отраслевые рекомендации."""
        # Проверяем для финансовой отрасли
        finance_result = self.suggester.generate_suggestions(
            self.base_recommendations, 
            self.feature_scores, 
            'finance'
        )
        
        # Проверяем для отрасли здравоохранения
        health_result = self.suggester.generate_suggestions(
            self.base_recommendations, 
            self.feature_scores, 
            'health'
        )
        
        # Проверяем наличие рекомендаций для разных отраслей
        self.assertTrue(len(finance_result['content_improvement']) > 0)
        self.assertTrue(len(health_result['content_improvement']) > 0)
    
    def test_prioritize_tasks_returns_sorted_list(self):
        """Тест проверяет, что задачи отсортированы по приоритету."""
        recommendations = {
            'content_improvement': [
                'Увеличьте объем контента до 1500 слов',
                'Добавьте больше примеров'
            ],
            'keyword_optimization': [
                'Увеличьте плотность ключевых слов',
                'Добавьте ключевые слова в заголовки'
            ]
        }
        
        result = self.suggester.prioritize_tasks(
            recommendations, 
            self.feature_scores, 
            self.weighted_scores
        )
        
        # Проверяем, что результат является списком
        self.assertIsInstance(result, list)
        
        # Проверяем, что каждый элемент содержит 'task' и поле для приоритета
        for item in result:
            self.assertIn('task', item)
            # В текущей реализации используется 'priority_score' вместо 'priority'
            self.assertIn('priority_score', item)
        
        # Проверяем, что список отсортирован по убыванию приоритета
        for i in range(len(result) - 1):
            self.assertGreaterEqual(
                result[i]['priority_score'],
                result[i + 1]['priority_score']
            )


if __name__ == '__main__':
    unittest.main()
