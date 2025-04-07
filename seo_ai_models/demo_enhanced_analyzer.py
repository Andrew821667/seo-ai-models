#!/usr/bin/env python3
"""
Демонстрационный скрипт для тестирования улучшенного анализатора контента.
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Добавляем родительскую директорию в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer
from seo_ai_models.parsers.spa_parser import SPAParser

def analyze_url(url, keywords):
    """
    Анализирует URL с помощью улучшенного анализатора.
    
    Args:
        url: URL для анализа
        keywords: Список ключевых слов
        
    Returns:
        dict: Результаты анализа
    """
    print(f"Анализ URL: {url}")
    print(f"Ключевые слова: {', '.join(keywords)}")
    
    # Инициализируем SPA Parser
    parser = SPAParser(
        wait_for_load=7000,
        wait_for_timeout=45000,
        record_ajax=True
    )
    
    # Парсим URL
    print("\nПарсинг страницы...")
    parsed_data = parser.analyze_url_sync(url)
    
    if not parsed_data.get('success'):
        print(f"Ошибка парсинга: {parsed_data.get('error')}")
        return None
    
    # Извлекаем контент
    content = ""
    if 'content' in parsed_data and 'all_text' in parsed_data['content'].get('content', {}):
        content = parsed_data['content']['content']['all_text']
    else:
        for paragraph in parsed_data.get('content', {}).get('content', {}).get('paragraphs', []):
            content += paragraph + "\n\n"
    
    print(f"Извлечено {len(content)} символов контента.")
    
    # Получаем HTML-контент
    html_content = parsed_data.get('html', '')
    
    # Анализируем контент
    print("\nАнализ контента...")
    analyzer = EnhancedContentAnalyzer()
    metrics = analyzer.analyze_content(content, html_content)
    
    # Анализируем ключевые слова
    keyword_analysis = analyzer.extract_keywords(content, keywords)
    
    # Формируем результаты
    results = {
        'url': url,
        'timestamp': datetime.now().isoformat(),
        'is_spa': parsed_data.get('site_type', {}).get('is_spa', False),
        'content_metrics': metrics,
        'keyword_analysis': keyword_analysis,
        'parsing_time': parsed_data.get('processing_time', 0)
    }
    
    # Выводим основные метрики
    print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    print(f"Количество слов: {metrics.get('word_count', 0)}")
    print(f"Количество предложений: {metrics.get('sentence_count', 0)}")
    print(f"Средняя длина предложения: {metrics.get('avg_sentence_length', 0):.1f} слов")
    print(f"Читабельность: {metrics.get('flesch_reading_ease', 0):.1f}/100")
    print(f"Плотность ключевых слов: {keyword_analysis.get('density', 0)*100:.2f}%")
    print(f"Покрытие ключевых слов: {keyword_analysis.get('coverage', 0)*100:.1f}%")
    
    if 'main_topics' in metrics and metrics['main_topics']:
        print("\nОсновные темы:")
        for topic in metrics['main_topics']:
            print(f" - {topic}")
    
    print("\nСтруктура контента:")
    print(f" - Заголовков: {metrics.get('headers_count', 0)}")
    print(f" - Параграфов: {metrics.get('paragraphs_count', 0)}")
    print(f" - Списков: {metrics.get('lists_count', 0) if 'lists_count' in metrics else 'Н/Д'}")
    print(f" - Изображений: {metrics.get('images_count', 0) if 'images_count' in metrics else 'Н/Д'}")
    
    return results

def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description="Демонстрация улучшенного анализатора контента")
    parser.add_argument("url", help="URL для анализа")
    parser.add_argument("--keywords", help="Ключевые слова через запятую", default="seo,optimization,content")
    
    args = parser.parse_args()
    
    # Преобразуем строку ключевых слов в список
    keywords = [k.strip() for k in args.keywords.split(",")]
    
    # Анализируем URL
    results = analyze_url(args.url, keywords)
    
    if results:
        # Сохраняем результаты в JSON-файл
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: x if not isinstance(x, datetime) else x.isoformat())
        
        print(f"\nПолные результаты сохранены в файл: {filename}")
    
    return 0 if results else 1

if __name__ == "__main__":
    sys.exit(main())
