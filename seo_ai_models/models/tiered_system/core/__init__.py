"""
Ядро многоуровневой системы SEO анализа.

Пакет предоставляет базовые компоненты многоуровневой системы,
включая TieredAdvisor, ResourceOptimizer, FeatureGating и LLMTokenManager.
"""

from seo_ai_models.models.tiered_system.core.tiered_advisor import TieredAdvisor, TierPlan
from seo_ai_models.models.tiered_system.core.resource_optimizer import ResourceOptimizer, ResourceType
from seo_ai_models.models.tiered_system.core.feature_gating import FeatureGating
from seo_ai_models.models.tiered_system.core.llm_token_manager import LLMTokenManager

__all__ = [
    'TieredAdvisor',
    'TierPlan',
    'ResourceOptimizer',
    'ResourceType',
    'FeatureGating',
    'LLMTokenManager',
]
