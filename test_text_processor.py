import unittest
import sys
import os
sys.path.append('/content/seo-ai-models')

from seo_ai_models.common.utils.text_processing import TextProcessor

class TestTextProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = TextProcessor()
        self.test_text = """
        # Тестовая статья о SEO
        
        Это тестовая статья для проверки функциональности TextProcessor.
        В данной статье мы рассмотрим основные принципы SEO-оптимизации.
        
        ## Заголовок второго уровня
        
        Контент второго уровня с информацией о ключевых словах и их важности.
        
        ### Заголовок третьего уровня
        
        Более специфичная информация о структуре текста и SEO.
        """
    
    def test_detect_language(self):
        # Проверка определения языка
        lang = self.processor.detect_language(self.test_text)
        self.assertEqual(lang, 'ru')
    
    def test_tokenize(self):
        # Проверка токенизации
        tokens = self.processor.tokenize(self.test_text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        # Проверяем, что важные слова присутствуют в токенах
        important_words = ['статья', 'seo', 'тестовая']
        for word in important_words:
            self.assertTrue(any(word.lower() in token.lower() for token in tokens))
    
    def test_extract_headers(self):
        # Проверка извлечения заголовков
        headers = self.processor.extract_headers(self.test_text)
        self.assertIsInstance(headers, list)
        self.assertEqual(len(headers), 3)  # должно быть 3 заголовка
        # Проверяем уровни заголовков
        header_levels = [h['level'] for h in headers]
        self.assertEqual(sorted(header_levels), [1, 2, 3])
    
    def test_calculate_readability(self):
        # Проверка расчета читабельности
        readability = self.processor.calculate_readability(self.test_text)
        self.assertIsInstance(readability, dict)
        self.assertIn('flesch_reading_ease', readability)
        # Оценка должна быть в диапазоне 0-100
        self.assertTrue(0 <= readability['flesch_reading_ease'] <= 100)
    
    def test_analyze_structure(self):
        # Проверка анализа структуры
        structure = self.processor.analyze_text_structure(self.test_text)
        self.assertIsInstance(structure, dict)
        # Проверяем наличие ключевых метрик
        expected_keys = ['headers_count', 'paragraphs_count', 'avg_paragraph_length']
        for key in expected_keys:
            self.assertIn(key, structure)
        # Количество заголовков должно соответствовать
        self.assertEqual(structure['headers_count'], 3)
