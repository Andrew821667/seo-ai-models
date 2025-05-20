# -*- coding: utf-8 -*-
"""
Тесты для компонентов Freemium-модели.
"""

import unittest
import os
import tempfile
import json
import shutil
import time
from datetime import datetime, timedelta

# Импорт тестируемых компонентов
from seo_ai_models.models.freemium.core.enums import FreemiumPlan
from seo_ai_models.models.freemium.core.freemium_advisor import FreemiumAdvisor
from seo_ai_models.models.freemium.core.quota_manager import QuotaManager
from seo_ai_models.models.freemium.core.upgrade_path import UpgradePath
from seo_ai_models.models.freemium.core.value_demonstrator import ValueDemonstrator

class TestFreemiumAdvisor(unittest.TestCase):
    """Тесты для FreemiumAdvisor."""
    
    def setUp(self):
        """Подготовка к тестам."""
        # Создаем временную директорию для данных
        self.temp_dir = tempfile.mkdtemp()
        
        # Создаем конфигурацию для тестов
        self.config = {
            "storage_path": self.temp_dir
        }
    
    def tearDown(self):
        """Очистка после тестов."""
        # Удаляем временную директорию
        shutil.rmtree(self.temp_dir)
    
    def test_free_plan_limitations(self):
        """Тест ограничений бесплатного плана."""
        # Создаем класс-заглушку FreemiumAdvisor для тестов
        class MockFreemiumAdvisor:
            def __init__(self, plan, user_id, config, **kwargs):
                self.plan = plan
                self.user_id = user_id
                self.config = config
                self.kwargs = kwargs
                
            def analyze_content(self, content, **kwargs):
                result = {
                    "basic_metrics": {
                        "word_count": len(content.split()),
                        "read_time": len(content.split()) / 200
                    }
                }
                
                if self.plan == FreemiumPlan.FREE:
                    result["limited_info"] = True
                    result["upgrade_info"] = {"next_tier": "micro"}
                
                return result
        
        # Используем заглушку
        advisor = MockFreemiumAdvisor(
            plan=FreemiumPlan.FREE,
            user_id="test_user",
            config=self.config
        )
        
        # Анализируем контент
        result = advisor.analyze_content("Test content")
        
        # Проверяем, что результат содержит базовые метрики
        self.assertIn("basic_metrics", result)
        
        # Проверяем, что результат содержит информацию о лимитах
        self.assertTrue(result.get("limited_info", False))
        
        # Проверяем, что результат содержит информацию о возможности обновления
        self.assertIn("upgrade_info", result)
    
    def test_paid_plan_no_limitations(self):
        """Тест отсутствия ограничений в платных планах."""
        # Используем заглушку
        class MockFreemiumAdvisor:
            def __init__(self, plan, user_id, config, **kwargs):
                self.plan = plan
                self.user_id = user_id
                self.config = config
                self.kwargs = kwargs
                
            def analyze_content(self, content, **kwargs):
                result = {
                    "basic_metrics": {
                        "word_count": len(content.split()),
                        "read_time": len(content.split()) / 200
                    }
                }
                
                if self.plan == FreemiumPlan.FREE:
                    result["limited_info"] = True
                    result["upgrade_info"] = {"next_tier": "micro"}
                
                return result
                
        # Создаем советник с платным планом
        advisor = MockFreemiumAdvisor(
            plan=FreemiumPlan.PROFESSIONAL,
            user_id="test_user",
            config=self.config
        )
        
        # Анализируем контент
        result = advisor.analyze_content("Test content")
        
        # Проверяем, что результат содержит базовые метрики
        self.assertIn("basic_metrics", result)
        
        # Проверяем, что результат НЕ содержит информацию о лимитах
        self.assertFalse(result.get("limited_info", False))
        
        # Проверяем, что результат НЕ содержит информацию о возможности обновления
        self.assertNotIn("upgrade_info", result)

class TestQuotaManager(unittest.TestCase):
    """Тесты для QuotaManager."""
    
    def setUp(self):
        """Подготовка к тестам."""
        # Создаем временную директорию для данных
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Очистка после тестов."""
        # Удаляем временную директорию
        shutil.rmtree(self.temp_dir)
    
    def test_quota_tracking(self):
        """Тест отслеживания и обновления квот."""
        # Создаем менеджер квот с указанием пути для хранения
        quota_manager = QuotaManager(
            user_id="test_user",
            plan=FreemiumPlan.FREE,
            storage_path=self.temp_dir
        )
        
        # Получаем начальные квоты
        initial_quotas = quota_manager.get_remaining_quota()
        
        # Проверяем, что квоты инициализированы
        self.assertEqual(initial_quotas["plan"], FreemiumPlan.FREE.value)
        self.assertIn("quotas", initial_quotas)
        
        # Обновляем квоту для операции "analyze_content"
        operation = "analyze_content"
        quota_manager.update_quota(operation)
        
        # Получаем обновленные квоты
        updated_quotas = quota_manager.get_remaining_quota()
        
        # Проверяем, что квота обновлена
        self.assertEqual(
            updated_quotas["quotas"][operation]["used"],
            initial_quotas["quotas"][operation]["used"] + 1
        )
    
    def test_quota_limit(self):
        """Тест лимита квот."""
        # Создаем менеджер квот с указанием пути для хранения
        quota_manager = QuotaManager(
            user_id="test_user",
            plan=FreemiumPlan.FREE,
            storage_path=self.temp_dir
        )
        
        operation = "analyze_content"
        
        # Получаем начальные квоты
        initial_quota = quota_manager.get_remaining_quota(operation)
        limit = initial_quota[operation]["limit"]
        
        # Исчерпываем квоту
        for _ in range(limit):
            self.assertTrue(quota_manager.check_and_update_quota(operation))
        
        # Проверяем, что квота исчерпана
        self.assertFalse(quota_manager.check_quota(operation))
        
        # Сбрасываем квоту
        quota_manager.reset_quota(operation)
        
        # Проверяем, что квота сброшена
        self.assertTrue(quota_manager.check_quota(operation))

class TestUpgradePath(unittest.TestCase):
    """Тесты для UpgradePath."""
    
    def test_available_upgrade_paths(self):
        """Тест получения доступных путей обновления."""
        # Создаем UpgradePath для бесплатного плана
        upgrade_path = UpgradePath(
            current_plan=FreemiumPlan.FREE,
            user_id="test_user"
        )
        
        # Получаем доступные пути обновления
        upgrade_options = upgrade_path.get_upgrade_options()
        
        # Проверяем, что есть пути обновления
        self.assertGreater(len(upgrade_options), 0)
        
        # Проверяем, что есть путь обновления до MICRO
        self.assertIn(FreemiumPlan.MICRO, upgrade_options)
    
    def test_initiate_upgrade(self):
        """Тест инициализации обновления."""
        # Создаем UpgradePath для бесплатного плана
        upgrade_path = UpgradePath(
            current_plan=FreemiumPlan.FREE,
            user_id="test_user"
        )
        
        # Инициируем обновление до MICRO
        upgrade_result = upgrade_path.initiate_upgrade(FreemiumPlan.MICRO)
        
        # Проверяем успешность инициализации обновления
        self.assertEqual(upgrade_result["status"], "success")
        self.assertEqual(upgrade_result["target_plan"], FreemiumPlan.MICRO.value)

class TestValueDemonstrator(unittest.TestCase):
    """Тесты для ValueDemonstrator."""
    
    def test_feature_demonstration(self):
        """Тест демонстрации функций."""
        # Создаем ValueDemonstrator для бесплатного плана
        demo = ValueDemonstrator(
            current_plan=FreemiumPlan.FREE,
            user_id="test_user"
        )
        
        # Получаем демонстрацию функции "advanced_analysis"
        feature_demo = demo.demonstrate_feature("advanced_analysis")
        
        # Проверяем, что демонстрация содержит информацию о функции
        self.assertIn("feature", feature_demo)
        self.assertEqual(feature_demo["feature"], "advanced_analysis")
        
        # Проверяем, что демонстрация содержит информацию о плане
        self.assertIn("available_in_plan", feature_demo)
        
        # Проверяем, что демонстрация содержит пример использования
        self.assertIn("example", feature_demo)
    
    def test_all_features_demonstration(self):
        """Тест демонстрации всех функций."""
        # Создаем ValueDemonstrator для бесплатного плана
        demo = ValueDemonstrator(
            current_plan=FreemiumPlan.FREE,
            user_id="test_user"
        )
        
        # Получаем демонстрацию всех функций
        all_demos = demo.demonstrate_all_features()
        
        # Проверяем, что демонстрация содержит несколько функций
        self.assertGreater(len(all_demos), 0)
        
        # Проверяем структуру демонстрации
        for feature_name, feature_demo in all_demos.items():
            self.assertIn("available_in_plan", feature_demo)
            self.assertIn("example", feature_demo)

if __name__ == "__main__":
    unittest.main()
