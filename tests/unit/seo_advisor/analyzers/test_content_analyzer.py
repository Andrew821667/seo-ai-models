"""Тесты для ContentAnalyzer."""

import unittest
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer


class TestContentAnalyzer(unittest.TestCase):
    """Тесты для класса ContentAnalyzer."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.analyzer = ContentAnalyzer()
        self.test_content = """
        # Заголовок статьи о SEO
        
        Это тестовый контент для анализа. Он содержит ключевые слова для SEO оптимизации.
        
        ## Подзаголовок раздела
        
        Контент должен быть качественным и полезным для пользователей.
        
        * Пункт 1
        * Пункт 2
        * Пункт 3
        
        ## Заключение
        
        В заключение, SEO оптимизация очень важна для продвижения сайта.
        """
        self.test_keywords = ["SEO", "оптимизация", "контент", "ключевые слова"]
    
    def test_analyze_text_returns_required_metrics(self):
        """Тест проверяет, что метод analyze_text возвращает все необходимые метрики."""
        metrics = self.analyzer.analyze_text(self.test_content)
        
        # Проверяем наличие обязательных метрик
        required_metrics = [
            'word_count', 'readability', 'meta_score', 'header_score',
            'multimedia_score', 'linking_score', 'topic_relevance',
            'semantic_depth', 'engagement_potential'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Метрика {metric} отсутствует в результатах")
    
    def test_analyze_text_word_count(self):
        """Тест проверяет корректность подсчета слов."""
        metrics = self.analyzer.analyze_text(self.test_content)
        
        # Проверяем, что количество слов положительное и соответствует ожидаемому диапазону
        self.assertGreater(metrics['word_count'], 0)
        self.assertLess(metrics['word_count'], 100)  # Примерная оценка для тестового контента
    
    def test_extract_keywords_returns_expected_format(self):
        """Тест проверяет формат результата extract_keywords."""
        result = self.analyzer.extract_keywords(self.test_content, self.test_keywords)
        
        # Проверяем наличие обязательных полей
        self.assertIn('density', result)
        self.assertIn('frequency', result)
        self.assertIn('distribution', result)
        
        # Проверяем тип density
        self.assertIsInstance(result['density'], float)
        
        # Проверяем, что frequency содержит все ключевые слова
        for keyword in self.test_keywords:
            self.assertIn(keyword, result['frequency'])
    
    def test_extract_keywords_empty_content(self):
        """Тест проверяет обработку пустого контента."""
        result = self.analyzer.extract_keywords("", self.test_keywords)
        
        # Плотность ключевых слов должна быть нулевой для пустого контента
        self.assertEqual(result['density'], 0.0)
    
    def test_extract_keywords_empty_keywords(self):
        """Тест проверяет обработку пустого списка ключевых слов."""
        result = self.analyzer.extract_keywords(self.test_content, [])
        
        # Плотность ключевых слов должна быть нулевой для пустого списка ключевых слов
        self.assertEqual(result['density'], 0.0)
        # Частота должна быть пустым словарем
        self.assertEqual(result['frequency'], {})


if __name__ == '__main__':
    unittest.main()
