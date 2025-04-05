"""
Простой демонстрационный скрипт для компонентов парсинга
"""

import argparse
import json
from seo_ai_models.parsers.parsing_pipeline import ParsingPipeline

def main():
    parser = argparse.ArgumentParser(description="Демонстрация компонентов парсинга")
    parser.add_argument("--url", help="URL для анализа", required=True)
    parser.add_argument("--output", help="Выходной файл (JSON)", default="analysis_result.json")
    args = parser.parse_args()
    
    print(f"Анализ URL: {args.url}")
    
    # Инициализация парсинг-конвейера
    pipeline = ParsingPipeline()
    
    # Анализ URL
    result = pipeline.analyze_url(args.url)
    
    # Сохранение результата в файл
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты анализа сохранены в {args.output}")
    
    # Вывод краткой информации о странице
    if result.get("success"):
        print("\nКраткая информация о странице:")
        print(f"Заголовок: {result.get('content_analysis', {}).get('title', 'Не найден')}")
        
        # Подсчет количества заголовков
        headings = result.get('content_analysis', {}).get('headings', {})
        h1_count = len(headings.get('h1', []))
        h2_count = len(headings.get('h2', []))
        h3_count = len(headings.get('h3', []))
        
        print(f"Заголовки: H1: {h1_count}, H2: {h2_count}, H3: {h3_count}")
        
        # Подсчет количества параграфов
        paragraphs_count = len(result.get('content_analysis', {}).get('content', {}).get('paragraphs', []))
        print(f"Количество параграфов: {paragraphs_count}")
        
        # Информация о ссылках
        links = result.get('meta_analysis', {}).get('links', {})
        internal_count = len(links.get('internal', []))
        external_count = len(links.get('external', []))
        
        print(f"Ссылки: внутренние: {internal_count}, внешние: {external_count}")
    else:
        print(f"Ошибка при анализе: {result.get('error', 'Неизвестная ошибка')}")

if __name__ == "__main__":
    main()
