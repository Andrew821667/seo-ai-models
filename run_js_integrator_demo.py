
"""
Демонстрация использования улучшенных JavaScript-компонентов.
"""

import sys
import os
import json
import logging
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("js_integrator_demo")

# Добавляем директорию проекта в путь импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем компоненты
from seo_ai_models.parsers.unified.js_integrator import JSIntegrator

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Демонстрация улучшенных JavaScript-компонентов")
    
    parser.add_argument("url", help="URL для сканирования")
    parser.add_argument("--output", help="Файл для сохранения результатов")
    parser.add_argument("--max-pages", type=int, default=3, help="Максимальное количество страниц")
    parser.add_argument("--max-depth", type=int, default=2, help="Максимальная глубина сканирования")
    parser.add_argument("--delay", type=float, default=1.0, help="Задержка между запросами")
    parser.add_argument("--disable-websocket", action="store_true", help="Отключить анализ WebSocket")
    parser.add_argument("--disable-graphql", action="store_true", help="Отключить анализ GraphQL")
    parser.add_argument("--disable-routing", action="store_true", help="Отключить обработку маршрутизации")
    parser.add_argument("--emulate-user", action="store_true", help="Эмулировать поведение пользователя")
    parser.add_argument("--bypass-protection", action="store_true", help="Обход защиты от ботов")
    
    return parser.parse_args()

def main():
    """Основная функция демонстрации"""
    args = parse_args()
    
    logger.info(f"Запуск демонстрации на {args.url}")
    
    # Создание интегратора
    integrator = JSIntegrator(
        enable_websocket=not args.disable_websocket,
        enable_graphql=not args.disable_graphql,
        enable_client_routing=not args.disable_routing,
        emulate_user_behavior=args.emulate_user,
        bypass_protection=args.bypass_protection
    )
    
    # Дополнительные опции для краулера
    crawler_options = {
        "max_pages": args.max_pages,
        "max_depth": args.max_depth,
        "delay": args.delay,
        "headless": True
    }
    
    # Парсинг сайта
    logger.info("Начало парсинга...")
    result = integrator.parse_site(args.url, **crawler_options)
    
    # Получение объединенных результатов
    combined_results = integrator.get_combined_results(result)
    
    # Вывод сводки
    logger.info("=== Сводка результатов ===")
    
    if combined_results["summary"]["websocket_detected"]:
        logger.info("✅ Обнаружена WebSocket-коммуникация")
    else:
        logger.info("❌ WebSocket-коммуникация не обнаружена")
        
    if combined_results["summary"]["graphql_detected"]:
        logger.info("✅ Обнаружены GraphQL-операции")
    else:
        logger.info("❌ GraphQL-операции не обнаружены")
        
    if combined_results["summary"]["client_routing_detected"]:
        logger.info("✅ Обнаружена клиентская маршрутизация")
    else:
        logger.info("❌ Клиентская маршрутизация не обнаружена")
    
    # Вывод обнаруженных технологий
    if combined_results["technologies"]:
        logger.info("=== Обнаруженные технологии ===")
        for tech in combined_results["technologies"]:
            logger.info(f"  • {tech['name']} ({tech['type']})")
    
    # Сохранение результатов в файл
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {args.output}")
    
    logger.info("Демонстрация завершена")

if __name__ == "__main__":
    main()
