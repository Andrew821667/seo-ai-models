import unittest
import sys
import os
sys.path.append('/content/seo-ai-models')

from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer

class TestContentAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = ContentAnalyzer()
        self.test_text = """
        # Тестовая статья о SEO
        
        Это тестовая статья для проверки функциональности ContentAnalyzer.
        В данной статье мы рассмотрим основные принципы SEO-оптимизации.
        
        ## Заголовок второго уровня
        
        Контент второго уровня с информацией о ключевых словах и их важности.
        
        ### Заголовок третьего уровня
        
        Более специфичная информация о структуре текста и SEO.
        """
        self.test_keywords = ['SEO', 'оптимизация', 'контент', 'ключевые слова']
    
    def test_analyze_text(self):
        # Проверка основного метода анализа текста
        metrics = self.analyzer.analyze_text(self.test_text)
        self.assertIsInstance(metrics, dict)
        
        # Проверяем наличие ключевых метрик
        expected_metrics = [
            'word_count', 'sentence_count', 'avg_sentence_length', 
            'readability', 'header_score', 'structure_score',
            'semantic_depth', 'topic_relevance', 'engagement_potential'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Проверяем, что метрики имеют правильные значения
        self.assertTrue(metrics['word_count'] > 0)
        self.assertTrue(metrics['sentence_count'] > 0)
        self.assertTrue(0 <= metrics['readability'] <= 1)
        self.assertTrue(0 <= metrics['header_score'] <= 1)
    
    def test_extract_keywords(self):
        # Проверка извлечения ключевых слов
        keyword_metrics = self.analyzer.extract_keywords(self.test_text, self.test_keywords)
        self.assertIsInstance(keyword_metrics, dict)
        
        # Проверяем наличие ключевых метрик
        expected_keys = ['density', 'prominence', 'coverage', 'frequency']
        for key in expected_keys:
            self.assertIn(key, keyword_metrics)
        
        # Проверяем, что плотность в разумных пределах
        self.assertTrue(0 <= keyword_metrics['density'] <= 1)
        # Проверяем, что частота имеет значения для каждого ключевого слова
        self.assertEqual(len(keyword_metrics['frequency']), len(self.test_keywords))
    
    def test_analyze_empty_text(self):
        # Проверка обработки пустого текста
        metrics = self.analyzer.analyze_text('')
        # Даже для пустого текста должны быть метрики, но некоторые должны быть 0
        self.assertEqual(metrics['word_count'], 0)
        self.assertEqual(metrics['sentence_count'], 0)
