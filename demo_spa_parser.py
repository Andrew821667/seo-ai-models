"""
Демонстрационный скрипт для интегрированного парсера SPA-сайтов.
"""

import argparse
import logging
import json
import sys
import time
import os
import asyncio

# Настраиваем пути для импорта
sys.path.insert(0, os.path.abspath('.'))

# Теперь можем импортировать наш модуль
from seo_ai_models.parsers.spa_parser import SPAParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Демонстрация интегрированного парсера SPA-сайтов")
    parser.add_argument("url", help="URL для анализа")
    parser.add_argument("--no-cache", action="store_true", help="Отключить кэширование")
    parser.add_argument("--clear-cache", action="store_true", help="Очистить кэш перед анализом")
    parser.add_argument("--output", help="Файл для сохранения результатов")
    parser.add_argument("--timeout", type=int, default=30000, help="Таймаут в миллисекундах")
    parser.add_argument("--crawl", action="store_true", help="Сканировать сайт")
    parser.add_argument("--max-pages", type=int, default=5, help="Максимальное количество страниц для сканирования")
    
    args = parser.parse_args()
    
    try:
        # Создаем парсер
        spa_parser = SPAParser(
            headless=True,
            wait_for_timeout=args.timeout,
            wait_for_load=5000,
            cache_enabled=not args.no_cache,
            max_retries=2
        )
        
        # Очистка кэша при необходимости
        if args.clear_cache:
            spa_parser.clear_cache()
            logger.info("Кэш очищен")
        
        start_time = time.time()
        
        # Сканирование сайта или анализ одного URL
        if args.crawl:
            logger.info(f"Сканирование сайта: {args.url}")
            result = spa_parser.crawl_site_sync(args.url, max_pages=args.max_pages)
        else:
            logger.info(f"Анализ URL: {args.url}")
            result = spa_parser.analyze_url_sync(args.url)
        
        elapsed = time.time() - start_time
        
        # Вывод результатов
        print(f"Время выполнения: {elapsed:.2f} сек")
        
        if args.crawl:
            print(f"Сканировано страниц: {result.get('total_pages', 0)}")
        
        if result.get("success"):
            if "site_type" in result:
                site_type = result["site_type"]
                print(f"Тип сайта: {'SPA' if site_type.get('is_spa') else 'Обычный'}")
                
                frameworks = site_type.get("detected_frameworks", [])
                if frameworks:
                    print(f"Обнаруженные фреймворки: {', '.join(frameworks)}")
            
            if "content" in result and not args.crawl:
                content = result["content"]
                print(f"\nЗаголовок: {content.get('title', 'Не найден')}")
                
                # Количество заголовков
                headings = content.get('headings', {})
                heading_count = sum(len(values) for key, values in headings.items())
                print(f"Количество заголовков: {heading_count}")
                
                # Текстовый контент
                text_content = content.get('content', {})
                if text_content:
                    paragraphs = text_content.get('paragraphs', [])
                    print(f"Количество параграфов: {len(paragraphs)}")
                    
                    all_text = text_content.get('all_text', '')
                    print(f"Общая длина текста: {len(all_text)} символов")
                    
                    if paragraphs:
                        print(f"\nНачало контента: {paragraphs[0][:100]}...")
            
            # AJAX-данные
            if "ajax_data" in result and not args.crawl:
                ajax_data = result["ajax_data"]
                api_calls = ajax_data.get('api_calls', [])
                
                if api_calls:
                    print(f"\nПерехвачено {len(api_calls)} AJAX-запросов:")
                    for i, call in enumerate(api_calls[:3], 1):
                        print(f"  {i}. {call['method']} {call['url']}")
                        
                    if len(api_calls) > 3:
                        print(f"  ...и еще {len(api_calls) - 3} запросов")
        else:
            print(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
        
        # Сохранение результатов в файл
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены в {args.output}")
        
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
