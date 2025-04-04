import unittest
import sys
import os
sys.path.append('/content/seo-ai-models')

from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor

class TestSEOAdvisor(unittest.TestCase):
    
    def setUp(self):
        self.advisor = SEOAdvisor(industry='blog')
        self.test_text = """
        # Тестовая статья о SEO
        
        Это тестовая статья для проверки функциональности SEOAdvisor.
        В данной статье мы рассмотрим основные принципы SEO-оптимизации.
        
        ## Заголовок второго уровня
        
        Контент второго уровня с информацией о ключевых словах и их важности.
        
        ### Заголовок третьего уровня
        
        Более специфичная информация о структуре текста и SEO.
        """
        self.test_keywords = ['SEO', 'оптимизация', 'контент', 'ключевые слова']
    
    def test_analyze_content(self):
        # Проверка комплексного анализа контента
        report = self.advisor.analyze_content(self.test_text, self.test_keywords)
        
        # Проверяем, что отчет содержит основные разделы
        self.assertTrue(hasattr(report, 'content_metrics'))
        self.assertTrue(hasattr(report, 'keyword_analysis'))
        self.assertTrue(hasattr(report, 'predicted_position'))
        self.assertTrue(hasattr(report, 'content_quality'))
        
        # Проверяем, что метрики контента содержат основные данные
        self.assertIn('word_count', report.content_metrics)
        self.assertIn('readability', report.content_metrics)
        
        # Проверяем, что анализ ключевых слов содержит основные метрики
        self.assertIn('density', report.keyword_analysis)
        
        # Проверяем, что предсказанная позиция в разумных пределах
        self.assertTrue(1 <= report.predicted_position <= 100)
        
        # Проверяем, что есть рекомендации по улучшению
        self.assertTrue(len(report.content_quality.strengths) >= 0)
        self.assertTrue(len(report.content_quality.weaknesses) >= 0)
    
    def test_analyze_with_different_industry(self):
        # Проверка анализа для разных отраслей
        advisor_finance = SEOAdvisor(industry='finance')
        advisor_health = SEOAdvisor(industry='health')
        
        report_blog = self.advisor.analyze_content(self.test_text, self.test_keywords)
        report_finance = advisor_finance.analyze_content(self.test_text, self.test_keywords)
        report_health = advisor_health.analyze_content(self.test_text, self.test_keywords)
        
        # Проверяем, что отрасль влияет на предсказанную позицию
        positions = [report_blog.predicted_position, report_finance.predicted_position, report_health.predicted_position]
        # В разных отраслях должны быть разные оценки
        self.assertTrue(len(set(positions)) > 1, "Разные отрасли должны давать разные предсказания")
