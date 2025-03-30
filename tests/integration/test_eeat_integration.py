"""Интеграционные тесты для E-E-A-T анализатора."""

import unittest
import sys
import os
import joblib
import numpy as np
from pathlib import Path

# Добавляем корень проекта в путь импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from seo_ai_models.models.seo_advisor.analyzers.eeat import EEATAnalyzer, EnhancedEEATAnalyzer


class TestEEATIntegration(unittest.TestCase):
    """Интеграционные тесты для проверки взаимодействия компонентов E-E-A-T."""
    
    def setUp(self):
        """Подготовка к тестам."""
        self.base_analyzer = EEATAnalyzer()
        
        # Ищем модель для enhanced анализатора
        model_path = Path('seo_ai_models/data/models/eeat/eeat_best_model.joblib')
        if model_path.exists():
            self.enhanced_analyzer = EnhancedEEATAnalyzer(model_path=str(model_path))
            self.model_available = True
        else:
            self.enhanced_analyzer = EnhancedEEATAnalyzer()  # Без модели
            self.model_available = False
        
        self.test_content = """
        # Руководство по инвестированию для начинающих
        
        Инвестирование - это способ обеспечить свое финансовое будущее. 
        В этой статье мы рассмотрим основные подходы к инвестированию для тех, 
        кто только начинает свой путь.
        
        ## Что такое инвестиции?
        
        Инвестиции - это вложение денег с целью получения дохода или увеличения 
        их стоимости с течением времени. Важно понимать разницу между сбережениями и инвестициями.
        
        ## Почему стоит инвестировать?
        
        По данным исследования Morgan Stanley, средняя годовая доходность 
        индекса S&P 500 за последние 30 лет составила около 10%.
        
        ## Основные виды инвестиций
        
        1. Акции - доли в компаниях
        2. Облигации - долговые ценные бумаги
        3. Недвижимость - материальные активы
        4. ETF и взаимные фонды - диверсифицированные инвестиции
        
        ## Советы от экспертов
        
        Профессор экономики Джереми Сигел в своей книге "Stocks for the Long Run" рекомендует...
        
        Источники:
        - Investopedia.com
        - SEC.gov
        - "The Intelligent Investor" by Benjamin Graham
        """
    
    def test_base_and_enhanced_compatibility(self):
        """Тест совместимости базового и улучшенного анализаторов."""
        # Получаем результаты анализа
        base_results = self.base_analyzer.analyze(self.test_content, industry='finance')
        enhanced_results = self.enhanced_analyzer.analyze(self.test_content, industry='finance')
        
        # Проверяем наличие одинаковых ключей в результатах
        for key in base_results.keys():
            self.assertIn(key, enhanced_results)
        
        # Проверяем, что улучшенный анализатор наследует логику базового
        self.assertEqual(base_results['ymyl_status'], enhanced_results['ymyl_status'])
        
        # Дополнительная проверка, если модель доступна
        if self.model_available:
            self.assertTrue(enhanced_results.get('ml_model_used', False))
    
    def test_finance_ymyl_strictness(self):
        """Тест строгости YMYL для финансовой отрасли."""
        # Анализ для финансовой отрасли
        finance_results = self.base_analyzer.analyze(self.test_content, industry='finance')
        
        # Анализ для общей отрасли
        general_results = self.base_analyzer.analyze(self.test_content, industry='default')
        
        # Для YMYL-отраслей требования должны быть строже
        self.assertEqual(finance_results['ymyl_status'], 1)
        self.assertEqual(general_results['ymyl_status'], 0)
        
        # Проверка наличия рекомендаций
        self.assertTrue(len(finance_results['recommendations']) > 0)


if __name__ == '__main__':
    unittest.main()
