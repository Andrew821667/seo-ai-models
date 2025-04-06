"""
Простой демонстрационный скрипт для парсера SPA
"""

import sys
import time
import json
import logging
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импортируем SPAParser прямо из модуля
sys.path.insert(0, '.')  # добавляем текущую директорию в путь поиска
from seo_ai_models.parsers.spa_parser import SPAParser

async def main_async():
    # Проверяем, что URL передан как аргумент
    if len(sys.argv) < 2:
        print("Использование: python simple_demo.py <URL>")
        return 1
        
    url = sys.argv[1]
    print(f"Анализ URL: {url}")
    
    # Создаем парсер с настроенными параметрами
    parser = SPAParser(
        headless=True,
        wait_for_load=5000,
        wait_for_timeout=30000,
        cache_enabled=True
    )
    
    start_time = time.time()
    
    # Анализируем URL
    result = await parser.analyze_url(url)
    
    elapsed = time.time() - start_time
    
    print(f"Время анализа: {elapsed:.2f} сек")
    
    if result["success"]:
        # Выводим информацию о сайте
        site_type = result.get("site_type", {})
        print(f"Тип сайта: {'SPA' if site_type.get('is_spa') else 'Обычный'}")
        
        # Выводим обнаруженные фреймворки
        frameworks = site_type.get("detected_frameworks", [])
        if frameworks:
            print(f"Обнаруженные фреймворки: {', '.join(frameworks)}")
        
        # Выводим информацию о контенте
        content = result.get("content", {})
        print(f"\nЗаголовок: {content.get('title', 'Не найден')}")
        
        # Подсчет заголовков
        headings = content.get("headings", {})
        heading_count = sum(len(values) for key, values in headings.items())
        print(f"Количество заголовков: {heading_count}")
        
        # Информация о текстовом контенте
        content_data = content.get("content", {})
        paragraphs = content_data.get("paragraphs", [])
        print(f"Количество параграфов: {len(paragraphs)}")
        
        # Пример текста
        if paragraphs:
            print(f"\nПример текста: {paragraphs[0][:100]}...")
        
        # AJAX-информация
        if "ajax_data" in result:
            ajax_data = result["ajax_data"]
            api_calls = ajax_data.get("api_calls", [])
            if api_calls:
                print(f"\nПерехвачено {len(api_calls)} AJAX-запросов")
    else:
        print(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
    
    return 0

if __name__ == "__main__":
    # Запускаем асинхронную функцию
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)
