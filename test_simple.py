
"""
Простой тест для проверки расширенных возможностей парсера.
"""

import sys
import os
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("test_simple")

# Добавляем директорию проекта в путь поиска
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем компоненты
from seo_ai_models.parsers.unified.crawlers.advanced_spa_crawler import AdvancedSPACrawler

def simple_test():
    """Простой тест на статическом сайте"""
    url = "https://example.com"
    
    logger.info(f"Тестирование на простом сайте: {url}")
    
    crawler = AdvancedSPACrawler(
        base_url=url,
        max_pages=1,
        max_depth=1,
        delay=0.5,
        headless=True,
        enable_websocket=True,
        enable_graphql=True,
        enable_client_routing=True,
        wait_for_timeout=5000
    )
    
    # Запускаем сканирование
    start_time = time.time()
    results = crawler.crawl()
    end_time = time.time()
    
    # Выводим результаты
    logger.info(f"Сканирование выполнено за {end_time - start_time:.2f}с")
    logger.info(f"Просканировано {len(results['crawled_urls'])} URL")
    logger.info(f"Найдено {len(results['found_urls'])} URL")
    
    return results

if __name__ == "__main__":
    simple_test()
