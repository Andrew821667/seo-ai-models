
"""
Тестирование расширенных возможностей парсера для проекта SEO AI Models.
"""

import sys
import os
import json
import time
import asyncio
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("test_advanced_parser")

# Добавляем корневую директорию проекта в путь импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем компоненты
try:
    from seo_ai_models.parsers.unified.crawlers.advanced_spa_crawler import AdvancedSPACrawler
    from seo_ai_models.parsers.unified.js_processing.js_processing_integrator import JSProcessingIntegrator
    from seo_ai_models.parsers.unified.protection_bypass import ProtectionBypass
except ImportError as e:
    logger.error(f"Ошибка импорта: {e}")
    sys.exit(1)

def parse_site(url, output_file=None, headless=True, bypass_protection=True, 
              enable_websocket=True, enable_graphql=True, enable_routing=True):
    """
    Парсит сайт с использованием продвинутого SPA-краулера.
    
    Args:
        url: URL для парсинга
        output_file: Файл для сохранения результатов (опционально)
        headless: Использовать ли безголовый режим
        bypass_protection: Пытаться ли обойти защиту от парсинга
        enable_websocket: Включить анализ WebSocket
        enable_graphql: Включить анализ GraphQL
        enable_routing: Включить обработку маршрутизации
    """
    logger.info(f"Начало тестирования расширенного парсера на {url}")
    
    # Создаем продвинутый SPA-краулер
    start_time = time.time()
    
    crawler = AdvancedSPACrawler(
        base_url=url,
        max_pages=3,          # Ограничиваем количество страниц для теста
        max_depth=2,          # Ограничиваем глубину для теста
        delay=1.0,            # Добавляем задержку между запросами
        headless=headless,    # Режим запуска браузера
        enable_websocket=enable_websocket,
        enable_graphql=enable_graphql,
        enable_client_routing=enable_routing,
        emulate_user_behavior=bypass_protection,
        bypass_protection=bypass_protection,
        wait_for_timeout=20000  # Увеличиваем таймаут для более надежной загрузки
    )
    
    # Запускаем сканирование
    logger.info("Запуск сканирования...")
    results = crawler.crawl()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Выводим базовую статистику
    logger.info(f"Сканирование завершено за {duration:.2f}с")
    logger.info(f"Просканировано {len(results['crawled_urls'])} URL")
    logger.info(f"Найдено {len(results['found_urls'])} URL")
    
    # Выводим детальную статистику по компонентам
    if enable_websocket and 'websocket' in results:
        ws_stats = results['websocket']['statistics']
        logger.info(f"WebSocket: сообщений - {ws_stats.get('total_messages', 0)}, соединений - {len(results['websocket'].get('connections', []))}")
        
        # Выводим информацию о активных соединениях
        for conn in results['websocket'].get('connections', []):
            logger.info(f"  WebSocket соединение: {conn.get('endpoint')} ({conn.get('message_count', 0)} сообщений)")
    
    if enable_graphql and 'graphql' in results:
        gql_stats = results['graphql']['statistics']
        logger.info(f"GraphQL: операций - {gql_stats.get('total_operations', 0)}, ответов - {gql_stats.get('total_responses', 0)}")
        
        # Выводим типы операций
        if 'operations_by_type' in gql_stats:
            for op_type, count in gql_stats['operations_by_type'].items():
                logger.info(f"  GraphQL {op_type}: {count} операций")
    
    if enable_routing and 'routing' in results:
        route_stats = results['routing']['statistics']
        logger.info(f"Маршрутизация: изменений - {route_stats.get('total_changes', 0)}, уникальных маршрутов - {route_stats.get('unique_routes', 0)}")
        
        # Выводим информацию о фреймворке, если обнаружен
        if 'framework' in route_stats and route_stats['framework']:
            logger.info(f"  Обнаружен фреймворк маршрутизации: {route_stats['framework'].get('name')}")
            
        # Выводим уникальные маршруты
        unique_routes = results['routing'].get('unique_routes', [])
        for route in unique_routes[:5]:  # Ограничиваем выводом первых 5 маршрутов
            logger.info(f"  Маршрут: {route.get('path')} - {(route.get('route_info') or {}).get('name', 'неизвестный')}")
    
    # Сохраняем результаты в файл, если указан
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены в файл: {output_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {str(e)}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование расширенного парсера")
    parser.add_argument("url", help="URL для тестирования")
    parser.add_argument("--output", help="Файл для сохранения результатов")
    parser.add_argument("--no-headless", action="store_true", help="Показывать браузер во время работы")
    parser.add_argument("--no-bypass", action="store_true", help="Отключить обход защиты")
    parser.add_argument("--no-websocket", action="store_true", help="Отключить анализ WebSocket")
    parser.add_argument("--no-graphql", action="store_true", help="Отключить анализ GraphQL")
    parser.add_argument("--no-routing", action="store_true", help="Отключить анализ маршрутизации")
    
    args = parser.parse_args()
    
    # Запускаем тест
    parse_site(
        url=args.url,
        output_file=args.output,
        headless=not args.no_headless,
        bypass_protection=not args.no_bypass,
        enable_websocket=not args.no_websocket,
        enable_graphql=not args.no_graphql,
        enable_routing=not args.no_routing
    )
