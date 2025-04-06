"""
Минимальный тест для проверки SPAParser
"""

import sys
import os
import asyncio
import logging
from pprint import pprint

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Добавляем текущую директорию в путь поиска модулей
sys.path.insert(0, os.path.abspath('.'))

# Импорт SPAParser
from seo_ai_models.parsers.spa_parser import SPAParser

def main():
    if len(sys.argv) < 2:
        print("Использование: python minimal_test.py <URL>")
        return 1
        
    url = sys.argv[1]
    print(f"Тестирование парсера SPA на URL: {url}")
    
    parser = SPAParser(
        headless=True,
        wait_for_load=5000,
        wait_for_timeout=30000,
        cache_enabled=True
    )
    
    result = parser.analyze_url_sync(url)
    
    if result["success"]:
        print("Анализ успешен!")
        print(f"Тип сайта: {'SPA' if result['site_type']['is_spa'] else 'Обычный'}")
        if result['site_type'].get('detected_frameworks'):
            print(f"Обнаруженные фреймворки: {', '.join(result['site_type']['detected_frameworks'])}")
        print(f"Заголовок: {result['content']['title']}")
        print(f"Общая длина текста: {result['content']['metadata']['text_length']} символов")
    else:
        print(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
