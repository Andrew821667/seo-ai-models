"""Тесты для SemanticAnalyzer."""

import unittest
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from seo_ai_models.models.seo_advisor.analyzers.semantic_analyzer import SemanticAnalyzer


class TestSemanticAnalyzer(unittest.TestCase):
    """Тесты для класса SemanticAnalyzer."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.analyzer = SemanticAnalyzer()
        self.test_content = """
        # Инвестиции в недвижимость
        
        Инвестирование в недвижимость является одним из самых популярных способов 
        сохранения и приумножения капитала. Рынок недвижимости предлагает различные
        варианты для инвесторов: от покупки жилья для сдачи в аренду до вложений
        в коммерческую недвижимость.
        
        ## Преимущества инвестиций в недвижимость
        
        * Стабильный пассивный доход
        * Защита от инфляции
        * Потенциальный рост стоимости актива
        
        ## Риски инвестирования
        
        При инвестировании в недвижимость необходимо учитывать риски: изменение
        рыночной конъюнктуры, законодательные изменения, затраты на обслуживание.
        """
        self.test_keywords = ["инвестиции", "недвижимость", "доход", "капитал"]
    
    def test_analyze_text_returns_required_metrics(self):
        """Тест проверяет, что метод analyze_text возвращает все необходимые метрики."""
        metrics = self.analyzer.analyze_text(self.test_content, self.test_keywords)
        
        # Проверяем наличие обязательных метрик
        required_metrics = [
            'semantic_density', 'semantic_coverage', 'topical_coherence', 'contextual_relevance'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Метрика {metric} отсутствует в результатах")
            
        # Проверяем, что значения в допустимом диапазоне
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            elif isinstance(value, dict):
                # Для словарей просто проверяем их наличие
                continue
    
    def test_generate_recommendations(self):
        """Тест проверяет генерацию рекомендаций."""
        analysis = {
            'semantic_density': 0.3,
            'semantic_coverage': 0.5,
            'topical_coherence': 0.7,
            'contextual_relevance': 0.6
        }
        
        recommendations = self.analyzer.generate_recommendations(analysis)
        
        # Проверяем, что рекомендации являются списком и не пусты
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)
        
        # Проверяем, что каждая рекомендация - строка
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)
    
    def test_analysis_with_empty_content(self):
        """Тест проверяет обработку пустого контента."""
        metrics = self.analyzer.analyze_text("", self.test_keywords)
        
        # Проверяем, что все метрики определены и не являются None
        for metric, value in metrics.items():
            self.assertIsNotNone(value)
    
    def test_analysis_with_empty_keywords(self):
        """Тест проверяет обработку пустого списка ключевых слов."""
        metrics = self.analyzer.analyze_text(self.test_content, [])
        
        # Проверяем, что все метрики определены и не являются None
        for metric, value in metrics.items():
            self.assertIsNotNone(value)
            
    def test_analysis_gives_higher_score_for_relevant_content(self):
        """Тест проверяет, что релевантный контент получает более высокую оценку."""
        # Релевантный контент
        relevant_content = """
        Инвестиции в недвижимость приносят стабильный доход. Капитал, вложенный в
        недвижимость, обычно хорошо защищен от инфляции. Инвестирование в недвижимость
        считается одним из наиболее надежных способов сохранения капитала.
        """
        
        # Нерелевантный контент
        irrelevant_content = """
        Кулинарные рецепты очень популярны в интернете. Многие люди ищут
        способы приготовления вкусных блюд. Готовка может быть увлекательным хобби.
        """
        
        relevant_metrics = self.analyzer.analyze_text(relevant_content, self.test_keywords)
        irrelevant_metrics = self.analyzer.analyze_text(irrelevant_content, self.test_keywords)
        
        # Проверяем, что релевантный контент получает более высокую оценку семантической релевантности
        self.assertGreater(
            relevant_metrics['contextual_relevance'],
            irrelevant_metrics['contextual_relevance']
        )


if __name__ == '__main__':
    unittest.main()
