import unittest
import sys
import os
sys.path.append('/content/seo-ai-models')

from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer

class TestEEATAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = EEATAnalyzer()
        self.test_text = "# Руководство по здоровому питанию\n\n" + \
                        "По данным исследований, правильное питание является основой здоровья.\n" + \
                        "Профессор Джон Смит из Медицинского университета рекомендует употреблять\n" + \
                        "больше овощей и фруктов.\n\n" + \
                        "## Принципы здорового питания\n\n" + \
                        "* Употребляйте разнообразную пищу\n" + \
                        "* Ограничьте потребление соли и сахара\n" + \
                        "* Пейте достаточно воды\n\n" + \
                        "Я лично следую этим принципам уже 5 лет и могу подтвердить их эффективность.\n\n" + \
                        "Источники:\n" + \
                        "1. Всемирная организация здравоохранения\n" + \
                        "2. Национальный институт здоровья\n\n" + \
                        "Последнее обновление: 15 марта 2023 года\n" + \
                        "Автор: Диетолог с 10-летним опытом работы"
    
    def test_analyze_basic(self):
        # Базовый анализ E-E-A-T
        results = self.analyzer.analyze(self.test_text)
        self.assertIsInstance(results, dict)
        
        # Проверяем наличие ключевых метрик
        expected_keys = [
            'experience_score', 'expertise_score', 'authority_score', 
            'trust_score', 'overall_eeat_score', 'recommendations'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Проверяем, что оценки в диапазоне от 0 до 1
        score_keys = ['experience_score', 'expertise_score', 'authority_score', 'trust_score', 'overall_eeat_score']
        for key in score_keys:
            self.assertTrue(0 <= results[key] <= 1, f"{key} должен быть в диапазоне [0,1]")
        
        # Проверяем, что есть рекомендации
        self.assertTrue(len(results['recommendations']) > 0)
    
    def test_analyze_ymyl_status(self):
        # Проверяем, что отрасли имеют разный YMYL статус
        # Используем параметр industry для определения YMYL статуса
        results_finance = self.analyzer.analyze(self.test_text, industry='finance')
        results_health = self.analyzer.analyze(self.test_text, industry='health')
        results_blog = self.analyzer.analyze(self.test_text, industry='blog')
        
        # Проверяем, что результаты содержат YMYL статус
        self.assertIn('ymyl_status', results_finance)
        self.assertIn('ymyl_status', results_health)
        self.assertIn('ymyl_status', results_blog)
        
        # Проверяем, что YMYL-отрасли имеют статус 1, а не YMYL (блог) - статус 0
        self.assertEqual(results_finance['ymyl_status'], 1, "Финансы должны иметь YMYL статус 1")
        self.assertEqual(results_health['ymyl_status'], 1, "Здоровье должно иметь YMYL статус 1")
        self.assertEqual(results_blog['ymyl_status'], 0, "Блог должен иметь YMYL статус 0")
    
    def test_analyze_industry_impact(self):
        # Проверяем, что для YMYL отраслей рекомендации могут отличаться
        results_finance = self.analyzer.analyze(self.test_text, industry='finance')
        results_health = self.analyzer.analyze(self.test_text, industry='health')
        
        # Проверяем наличие рекомендаций
        finance_recs = results_finance['recommendations']
        health_recs = results_health['recommendations']
        
        # Должны быть рекомендации
        self.assertGreater(len(finance_recs), 0, "Должны быть рекомендации для финансовой отрасли")
        self.assertGreater(len(health_recs), 0, "Должны быть рекомендации для отрасли здоровья")
        
        # Проверяем наличие ключевых оценок
        self.assertIn('expertise_score', results_finance)
        self.assertIn('expertise_score', results_health)
        self.assertIn('overall_eeat_score', results_finance)
        self.assertIn('overall_eeat_score', results_health)
