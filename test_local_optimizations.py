"""
Тестовый скрипт для проверки компонентов локальных оптимизаций.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import logging

# Отключаем логирование во время тестов
logging.disable(logging.CRITICAL)

class TestLocalOptimizations(unittest.TestCase):
    """Тесты для компонентов локальных оптимизаций."""
    
    def test_imports(self):
        """Проверка корректности импорта компонентов."""
        # Проверяем импорт всех компонентов
        from seo_ai_models.models.llm_integration.local_optimizations import (
            LocalLLMManager,
            HybridProcessingPipeline,
            IntelligentCache,
            OfflineAnalysisMode
        )
        
        # Если импорт прошел успешно, тест пройден
        self.assertTrue(True)
    
    def test_local_llm_manager_init(self):
        """Проверка инициализации LocalLLMManager."""
        from seo_ai_models.models.llm_integration.local_optimizations import LocalLLMManager
        
        # Создаем экземпляр LocalLLMManager
        manager = LocalLLMManager()
        
        # Проверяем атрибуты
        self.assertIsNotNone(manager.models_dir)
        self.assertTrue(hasattr(manager, 'supported_models'))
        self.assertTrue(hasattr(manager, 'available_models'))
    
    def test_hybrid_pipeline_init(self):
        """Проверка инициализации HybridProcessingPipeline."""
        from seo_ai_models.models.llm_integration.local_optimizations import (
            HybridProcessingPipeline,
            LocalLLMManager
        )
        from seo_ai_models.models.llm_integration.service.llm_service import LLMService
        
        # Создаем моки
        llm_service = MagicMock(spec=LLMService)
        local_llm_manager = MagicMock(spec=LocalLLMManager)
        
        # Создаем экземпляр HybridProcessingPipeline
        pipeline = HybridProcessingPipeline(llm_service, local_llm_manager)
        
        # Проверяем атрибуты
        self.assertEqual(pipeline.llm_service, llm_service)
        self.assertEqual(pipeline.local_llm_manager, local_llm_manager)
        self.assertTrue(hasattr(pipeline, 'processing_strategies'))
    
    def test_intelligent_cache_init(self):
        """Проверка инициализации IntelligentCache."""
        from seo_ai_models.models.llm_integration.local_optimizations import IntelligentCache
        
        # Создаем временную директорию для кэша
        with patch('os.makedirs'):
            # Создаем экземпляр IntelligentCache
            cache = IntelligentCache(cache_dir="/tmp/test_cache")
            
            # Проверяем атрибуты
            self.assertEqual(cache.cache_dir, "/tmp/test_cache")
            self.assertTrue(hasattr(cache, 'max_cache_size_mb'))
            self.assertTrue(hasattr(cache, 'default_ttl'))
    
    def test_offline_mode_init(self):
        """Проверка инициализации OfflineAnalysisMode."""
        from seo_ai_models.models.llm_integration.local_optimizations import (
            OfflineAnalysisMode,
            LocalLLMManager,
            IntelligentCache
        )
        
        # Создаем моки
        local_llm_manager = MagicMock(spec=LocalLLMManager)
        cache = MagicMock(spec=IntelligentCache)
        
        # Создаем экземпляр OfflineAnalysisMode
        offline_mode = OfflineAnalysisMode(local_llm_manager, cache)
        
        # Проверяем атрибуты
        self.assertEqual(offline_mode.local_llm_manager, local_llm_manager)
        self.assertEqual(offline_mode.cache, cache)
        self.assertFalse(offline_mode.offline_mode)  # По умолчанию выключен

if __name__ == '__main__':
    unittest.main()
