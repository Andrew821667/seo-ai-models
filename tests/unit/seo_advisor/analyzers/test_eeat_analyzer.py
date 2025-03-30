"""Тесты для анализатора E-E-A-T."""

import unittest
import sys
import os
# Добавляем корень проекта в путь импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer


class TestEEATAnalyzer(unittest.TestCase):
    """Тесты для E-E-A-T анализатора."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.analyzer = EEATAnalyzer()
        self.test_content = """
        Это тестовый контент для анализа E-E-A-T.
        Содержит некоторые элементы, такие как ссылки на исследования и экспертные мнения.
        
        По данным исследований, проведенных в 2024 году, этот подход является наиболее эффективным.
        
        Источник: https://example.com/research
        """
    
    def test_analyze_returns_required_fields(self):
        """Тест наличия всех необходимых полей в результате анализа."""
        result = self.analyzer.analyze(self.test_content)
        
        # Проверяем наличие обязательных полей
        self.assertIn('expertise_score', result)
        self.assertIn('authority_score', result)
        self.assertIn('trust_score', result)
        self.assertIn('overall_eeat_score', result)
        self.assertIn('recommendations', result)
    
    def test_industry_impacts_ymyl_status(self):
        """Тест влияния отрасли на статус YMYL."""
        # Проверка YMYL отрасли
        finance_result = self.analyzer.analyze(self.test_content, industry='finance')
        self.assertEqual(finance_result['ymyl_status'], 1)
        
        # Проверка не-YMYL отрасли
        blog_result = self.analyzer.analyze(self.test_content, industry='blog')
        self.assertEqual(blog_result['ymyl_status'], 0)


class TestEnhancedEEATAnalyzer(unittest.TestCase):
    """Тесты для улучшенного E-E-A-T анализатора."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.analyzer = EnhancedEEATAnalyzer()  # Без модели для упрощения тестирования
        self.test_content = """
        # Руководство по инвестированию
        
        По данным исследования Morgan Stanley, акции технологических компаний
        показали рост на 15% в первом квартале 2024 года.
        
        Эксперт по финансовым рынкам Александр Иванов отмечает важность
        диверсификации инвестиционного портфеля.
        
        Источник: Финансовый вестник, март 2024
        """
    
    def test_enhanced_analyzer_inherits_base_functionality(self):
        """Тест наследования базовой функциональности."""
        result = self.analyzer.analyze(self.test_content)
        
        # Проверяем наличие всех полей из базового анализатора
        self.assertIn('expertise_score', result)
        self.assertIn('authority_score', result)
        self.assertIn('trust_score', result)
        self.assertIn('overall_eeat_score', result)
    
    def test_ml_model_used_flag(self):
        """Тест флага использования модели машинного обучения."""
        # Без указания пути модели флаг должен быть False
        self.assertFalse(self.analyzer.ml_model_used)


if __name__ == '__main__':
    unittest.main()
