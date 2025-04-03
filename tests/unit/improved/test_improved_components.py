"""Тесты для улучшенных компонентов SEO AI Models."""

import unittest
import sys
import os
import logging
from pathlib import Path

# Отключаем логирование для тестов
logging.disable(logging.CRITICAL)

# Добавляем корень проекта в путь импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from seo_ai_models.common.utils.text_processing import TextProcessor
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer


class TestImprovedComponents(unittest.TestCase):
    """Тесты для улучшенных компонентов проекта."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.test_text = """
        # Тестовый контент для анализа
        
        Это пример контента, который будет использоваться для тестирования
        улучшенных компонентов SEO AI Models.
        
        ## Раздел с E-E-A-T сигналами
        
        Данный текст содержит ссылки на исследования и экспертные мнения.
        По данным исследования, проведенного в 2024 году, этот подход является эффективным.
        
        Автор: Эксперт с 10-летним опытом
        Источник: https://example.com/research
        
        ## Структурированные данные
        
        * Пункт 1
        * Пункт 2
        * Пункт 3
        
        ### Заголовок третьего уровня
        
        Ещё немного текста для анализа различных компонентов.
        """
        self.test_keywords = ["SEO", "анализ", "контент", "исследование"]
    
    def test_text_processor(self):
        """Тест улучшенного TextProcessor."""
        processor = TextProcessor()
        
        # Тест определения языка
        language = processor.detect_language(self.test_text)
        self.assertIn(language, ['ru', 'en', 'unknown'])
        
        # Тест токенизации
        tokens = processor.tokenize(self.test_text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Тест разбиения на предложения
        sentences = processor.split_sentences(self.test_text)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)
        
        # Тест извлечения заголовков
        headers = processor.extract_headers(self.test_text)
        self.assertIsInstance(headers, list)
        self.assertEqual(len(headers), 4)  # В тестовом тексте 4 заголовка
        
        # Тест анализа структуры текста
        structure = processor.analyze_text_structure(self.test_text)
        self.assertIsInstance(structure, dict)
        self.assertIn('headers_count', structure)
        self.assertIn('paragraphs_count', structure)
        self.assertIn('lists_count', structure)
        
        # Тест расчета читабельности
        readability = processor.calculate_readability(self.test_text)
        self.assertIsInstance(readability, dict)
        self.assertIn('flesch_reading_ease', readability)
    
    def test_eeat_analyzer(self):
        """Тест улучшенного EEATAnalyzer."""
        analyzer = EEATAnalyzer()
        
        # Тест базового анализа
        results = analyzer.analyze(self.test_text)
        
        # Проверяем наличие всех необходимых полей
        self.assertIn('experience_score', results)
        self.assertIn('expertise_score', results)
        self.assertIn('authority_score', results)
        self.assertIn('trust_score', results)
        self.assertIn('structural_score', results)
        self.assertIn('overall_eeat_score', results)
        self.assertIn('recommendations', results)
        self.assertIn('component_details', results)
        
        # Проверяем диапазоны оценок
        self.assertGreaterEqual(results['experience_score'], 0)
        self.assertLessEqual(results['experience_score'], 1)
        self.assertGreaterEqual(results['expertise_score'], 0)
        self.assertLessEqual(results['expertise_score'], 1)
        self.assertGreaterEqual(results['overall_eeat_score'], 0)
        self.assertLessEqual(results['overall_eeat_score'], 1)
        
        # Проверяем наличие рекомендаций
        self.assertIsInstance(results['recommendations'], list)
        
        # Тест для разных отраслей
        finance_results = analyzer.analyze(self.test_text, industry='finance')
        blog_results = analyzer.analyze(self.test_text, industry='blog')
        
        # Проверяем, что YMYL статус правильно устанавливается
        self.assertEqual(finance_results['ymyl_status'], 1)
        self.assertEqual(blog_results['ymyl_status'], 0)
    
    def test_enhanced_eeat_analyzer(self):
        """Тест EnhancedEEATAnalyzer."""
        analyzer = EnhancedEEATAnalyzer()  # Без модели
        
        # Тест базового анализа
        results = analyzer.analyze(self.test_text)
        
        # Проверяем, что все базовые поля присутствуют
        self.assertIn('experience_score', results)
        self.assertIn('expertise_score', results)
        self.assertIn('authority_score', results)
        self.assertIn('trust_score', results)
        self.assertIn('structural_score', results)
        self.assertIn('overall_eeat_score', results)
        
        # Проверяем ML-флаг
        self.assertIn('ml_model_used', results)
        
        # Пробуем с другой отраслью
        health_results = analyzer.analyze(self.test_text, industry='health')
        self.assertEqual(health_results['ymyl_status'], 1)
    
    def test_content_analyzer(self):
        """Тест улучшенного ContentAnalyzer."""
        analyzer = ContentAnalyzer()
        
        # Тест анализа текста
        metrics = analyzer.analyze_text(self.test_text)
        
        # Проверяем наличие всех необходимых метрик
        self.assertIn('word_count', metrics)
        self.assertIn('sentence_count', metrics)
        self.assertIn('readability', metrics)
        self.assertIn('header_score', metrics)
        self.assertIn('structure_score', metrics)
        self.assertIn('semantic_depth', metrics)
        self.assertIn('topic_relevance', metrics)
        self.assertIn('engagement_potential', metrics)
        
        # Проверяем диапазоны оценок
        self.assertGreaterEqual(metrics['readability'], 0)
        self.assertLessEqual(metrics['readability'], 1)
        self.assertGreaterEqual(metrics['header_score'], 0)
        self.assertLessEqual(metrics['header_score'], 1)
        
        # Тест анализа ключевых слов
        keyword_metrics = analyzer.extract_keywords(self.test_text, self.test_keywords)
        
        # Проверяем наличие всех необходимых метрик ключевых слов
        self.assertIn('density', keyword_metrics)
        self.assertIn('distribution', keyword_metrics)
        self.assertIn('frequency', keyword_metrics)
        self.assertIn('coverage', keyword_metrics)
        self.assertIn('prominence', keyword_metrics)
        
        # Проверяем диапазоны оценок
        self.assertGreaterEqual(keyword_metrics['density'], 0)
        self.assertLessEqual(keyword_metrics['density'], 1)
        self.assertGreaterEqual(keyword_metrics['coverage'], 0)
        self.assertLessEqual(keyword_metrics['coverage'], 1)


if __name__ == '__main__':
    unittest.main()
