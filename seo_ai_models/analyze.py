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
    parser = UnifiedParser(
        force_spa_mode=True,  # Принудительно используем SPA-режим для всех сайтов
        auto_detect_spa=True,  # На случай, если force_spa_mode не сработает
    )

    # Парсим URL
    print(f"📄 Парсинг страницы {url}...")
    try:
        parsed_data = parser.parse_url(url)
    except Exception as e:
        print(f"❌ Ошибка при парсинге: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Проверяем результат парсинга
    if not parsed_data:
        print("❌ Ошибка: Не удалось спарсить страницу (парсер вернул None)")
        return None

    print(f"✅ Страница успешно спарсена")
    print(f"   Ключи в parsed_data: {list(parsed_data.keys())}")

    page_data = parsed_data.get('page_data', {})
    metadata = page_data.get('metadata', {})
    print(f"   Заголовок: {metadata.get('title', 'N/A')}")
    print(f"   Описание: {metadata.get('description', 'N/A')[:100] if metadata.get('description') else 'N/A'}...")

    # Инициализируем анализатор контента
    print("\n📊 Запуск анализа контента...")
    analyzer = EnhancedContentAnalyzer()

    # Выполняем полный анализ
    # Извлекаем текстовый контент и HTML из спарсенных данных
    text_content = parsed_data.get('text', '') or parsed_data.get('content', '') or ''
    html_content = parsed_data.get('html', '') or parsed_data.get('html_content', '') or ''

    print(f"   Длина text_content: {len(text_content)} символов")
    print(f"   Длина html_content: {len(html_content)} символов")

    if len(text_content) == 0 and len(html_content) == 0:
        print("   ⚠️ ВНИМАНИЕ: Получен пустой контент от парсера!")
        print("   Возможные причины:")
        print("   - Сайт использует JavaScript для рендеринга")
        print("   - Сайт блокирует ботов")
        print("   - Проблемы с Playwright")

    # Выполняем полный анализ
    analysis_result = analyzer.analyze_content(text_content, html_content)

    if not analysis_result:
        print("❌ Ошибка: Анализ не вернул результатов")
        return None

    print("✅ Анализ успешно завершен")

    # Добавляем метаданные
    analysis_result['metadata'] = {
        'url': url,
        'analyzed_at': datetime.now().isoformat(),
        'analyzer_version': '1.0.0'
    }

    # Сохраняем результаты в JSON файл
    output_file = "analysis_result.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Результаты сохранены в файл: {output_file}")
    except Exception as e:
        print(f"❌ Ошибка сохранения файла: {e}")
        return None

    print(f"\n{'='*60}")
    print("✅ SEO АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
    print(f"{'='*60}\n")

    return analysis_result

def main():
    """
    Главная функция для запуска из командной строки.
    """
    parser = argparse.ArgumentParser(
        description='Полноценный SEO анализ сайта с использованием seo-ai-models'
    )
    parser.add_argument(
        '--url',
        required=True,
        help='URL сайта для анализа'
    )

    args = parser.parse_args()

    # Запускаем анализ
    result = analyze_url_full(args.url)

    # Возвращаем код выхода
    if result:
        sys.exit(0)  # Успех
    else:
        sys.exit(1)  # Ошибка

if __name__ == "__main__":
    main()
