"""
AutoFix Module - Automatic SEO Problem Fixing

Модуль автоматического исправления SEO проблем.
"""

from .engine import AutoFixEngine, FixAction, FixComplexity, FixStatus
from .fixers import (
    MetaTagsFixer,
    ImageAltTagsFixer,
    ContentRefreshFixer,
    SchemaMarkupFixer,
    InternalLinksFixer
)

__all__ = [
    "AutoFixEngine",
    "FixAction",
    "FixComplexity",
    "FixStatus",
    "MetaTagsFixer",
    "ImageAltTagsFixer",
    "ContentRefreshFixer",
    "SchemaMarkupFixer",
    "InternalLinksFixer",
]
