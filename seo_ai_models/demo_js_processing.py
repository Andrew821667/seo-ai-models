"""
Демонстрация использования расширенных возможностей обработки JavaScript
в проекте SEO AI Models.
"""

import json
import logging
import argparse
import time
from typing import Dict, Any

from seo_ai_models.parsers.unified.crawlers.advanced_spa_crawler import AdvancedSPACrawler
from seo_ai_models.parsers.unified.js_processing.js_processing_integrator import (
    JSProcessingIntegrator,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("demo_js_processing")


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Демонстрация расширенной обработки JavaScript")
    parser.add_argument("url", help="URL для сканирования")
    parser.add_argument(
        "--max-pages", type=int, default=5, help="Максимальное количество страниц для сканирования"
    )
    parser.add_argument(
        "--max-depth", type=int, default=2, help="Максимальная глубина сканирования"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Задержка между запросами в секундах"
    )
    parser.add_argument("--no-websocket", action="store_true", help="Отключить анализ WebSocket")
    parser.add_argument("--no-graphql", action="store_true", help="Отключить анализ GraphQL")
    parser.add_argument(
        "--no-routing", action="store_true", help="Отключить обработку маршрутизации"
    )
    parser.add_argument(
        "--emulate-user", action="store_true", help="Эмулировать поведение пользователя"
    )
    parser.add_argument(
        "--bypass-protection", action="store_true", help="Включить обход защиты от ботов"
    )
    parser.add_argument("--output", help="Файл для сохранения результатов")
    parser.add_argument("--headless", action="store_true", help="Запустить в безголовом режиме")

    return parser.parse_args()


def main():
    """Основная функция демонстрации"""
    args = parse_arguments()

    logger.info(f"Запуск демонстрации расширенной обработки JavaScript для {args.url}")

    # Создание интегратора обработки JavaScript
    js_integrator = JSProcessingIntegrator(
        enable_websocket=not args.no_websocket,
        enable_graphql=not args.no_graphql,
        enable_client_routing=not args.no_routing,
    )

    # Регистрация шаблонов (примеры)
    if not args.no_websocket:
        # Регистрация шаблонов WebSocket
        js_integrator.register_websocket_pattern(
            name="chat_messages",
            key_fields=["type", "messageId"],
            value_fields=["text", "sender", "timestamp"],
            type_identifier="chat",
        )

        js_integrator.register_websocket_pattern(
            name="notifications",
            key_fields=["type", "id"],
            value_fields=["content", "priority", "timestamp"],
            type_identifier="notification",
        )

    if not args.no_graphql:
        # Регистрация шаблонов GraphQL
        js_integrator.register_graphql_pattern(
            name="user_data",
            field_path="data.user",
            operation_type="query",
            operation_name="GetUser",
        )

        js_integrator.register_graphql_pattern(
            name="articles", field_path="data.articles.items", operation_type="query"
        )

    if not args.no_routing:
        # Регистрация шаблонов маршрутов
        js_integrator.register_route_pattern(
            pattern=r"/product/([a-zA-Z0-9-]+)", name="product_detail", params=["product_id"]
        )

        js_integrator.register_route_pattern(
            pattern=r"/category/([a-zA-Z0-9-]+)", name="category_list", params=["category_id"]
        )

        js_integrator.register_route_pattern(
            pattern=r"/user/([a-zA-Z0-9-]+)/profile", name="user_profile", params=["user_id"]
        )

    # Создание продвинутого SPA-краулера
    crawler = AdvancedSPACrawler(
        base_url=args.url,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay=args.delay,
        headless=args.headless,
        enable_websocket=not args.no_websocket,
        enable_graphql=not args.no_graphql,
        enable_client_routing=not args.no_routing,
        emulate_user_behavior=args.emulate_user,
        bypass_protection=args.bypass_protection,
    )

    # Запуск сканирования
    start_time = time.time()
    logger.info("Начало сканирования...")
    results = crawler.crawl()
    end_time = time.time()

    # Вывод статистики
    logger.info(f"Сканирование завершено за {end_time - start_time:.2f}с")
    logger.info(f"Просканировано {len(results['crawled_urls'])} URL")
    logger.info(f"Найдено {len(results['found_urls'])} URL")

    # Вывод информации о JavaScript-обработке
    if not args.no_websocket and "websocket" in results:
        ws_stats = results["websocket"]["statistics"]
        logger.info(
            f"WebSocket: сообщений - {ws_stats.get('total_messages', 0)}, соединений - {len(results['websocket'].get('connections', []))}"
        )

    if not args.no_graphql and "graphql" in results:
        gql_stats = results["graphql"]["statistics"]
        logger.info(
            f"GraphQL: операций - {gql_stats.get('total_operations', 0)}, ответов - {gql_stats.get('total_responses', 0)}"
        )

    if not args.no_routing and "routing" in results:
        route_stats = results["routing"]["statistics"]
        logger.info(
            f"Маршрутизация: изменений - {route_stats.get('total_changes', 0)}, уникальных маршрутов - {route_stats.get('unique_routes', 0)}"
        )

        if route_stats.get("framework"):
            logger.info(
                f"Обнаружен фреймворк маршрутизации: {route_stats['framework'].get('name')}"
            )

    # Сохранение результатов в файл, если указан
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены в файл: {args.output}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {str(e)}")

    # Вывод итоговой информации
    logger.info("Демонстрация завершена")

    return results


if __name__ == "__main__":
    main()
