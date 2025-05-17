"""
Пакет для локальных оптимизаций LLM.

Пакет предоставляет функционал для работы с локальными LLM-моделями,
гибридной обработки (облако + локальные ресурсы), интеллектуального
кэширования и офлайн-режима работы.
"""

from .local_llm_manager import LocalLLMManager
from .hybrid_processing_pipeline import HybridProcessingPipeline
from .intelligent_cache import IntelligentCache
from .offline_analysis_mode import OfflineAnalysisMode

__all__ = [
    'LocalLLMManager',
    'HybridProcessingPipeline',
    'IntelligentCache',
    'OfflineAnalysisMode'
]
