#!/usr/bin/env python3
"""
Полноценный SEO анализ с использованием всех возможностей seo-ai-models.
Используется в GitHub Actions для автоматического анализа legalaipro.ru.
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Добавляем родительскую директорию в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
    from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать модули анализа: {e}")
    print("Установите все зависимости через: pip install -r requirements.txt")
    MODULES_AVAILABLE = False


def analyze_url_full(url):
    """
    Полноценный анализ URL с использованием UnifiedParser и EnhancedContentAnalyzer.    
    Args:
        url: URL для анализа
        
    Returns:
        dict: Полные результаты анализа
    """
    print(f"\n{'='*60}")
    print(f"🔍 SEO АНАЛИЗ: {url}")
    print(f"{'='*60}\n")
    
    if not MODULES_AVAILABLE:
        print("❌ Модули недоступны. Невозможно выполнить анализ.")
        return None
    
    # Инициализируем парсер с правильными параметрами
    print("⚙️  Инициализация UnifiedParser...")
    parser = UnifiedParser()

                    # Парсим URL
    print(f"f📄 Парсинг страницы {url}...")
    parsed_data = parser.parse_url(url)
