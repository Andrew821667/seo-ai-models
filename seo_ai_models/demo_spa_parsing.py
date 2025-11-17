"""
Демонстрационный скрипт для парсинга SPA-сайтов в проекте SEO AI Models.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Optional

from seo_ai_models.parsers.adaptive_parsing_pipeline import AdaptiveParsingPipeline
from seo_ai_models.parsers.utils.spa_detector import SPADetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_results(results: Dict[str, Any], output_file: Optional[str] = None) -> None:
    """
    Сохраняет или выводит результаты.

    Args:
        results: Результаты для сохранения
        output_file: Имя файла для сохранения (если None, печать в stdout)
    """
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {output_file}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Демонстрация парсинга SPA-сайтов")
    subparsers = parser.add_subparsers(dest="command", help="Команда для выполнения")

    # Команда для определения типа сайта
    detect_parser = subparsers.add_parser("detect", help="Определить тип сайта (SPA или обычный)")
    detect_parser.add_argument("url", help="URL для анализа")
    detect_parser.add_argument("--output", help="Файл для сохранения результатов")

    # Команда для анализа URL
    analyze_parser = subparsers.add_parser("analyze", help="Анализировать URL")
    analyze_parser.add_argument("url", help="URL для анализа")
    analyze_parser.add_argument(
        "--force-spa", action="store_true", help="Принудительно использовать режим SPA"
    )
    analyze_parser.add_argument(
        "--no-detect", action="store_true", help="Отключить автоопределение типа сайта"
    )
    analyze_parser.add_argument("--output", help="Файл для сохранения результатов")

    # Команда для сканирования сайта
    crawl_parser = subparsers.add_parser("crawl", help="Сканировать сайт")
    crawl_parser.add_argument("url", help="Начальный URL для сканирования")
    crawl_parser.add_argument(
        "--max-pages", type=int, default=10, help="Максимальное количество страниц"
    )
    crawl_parser.add_argument(
        "--force-spa", action="store_true", help="Принудительно использовать режим SPA"
    )
    crawl_parser.add_argument(
        "--no-detect", action="store_true", help="Отключить автоопределение типа сайта"
    )
    crawl_parser.add_argument("--output", help="Файл для сохранения результатов")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "detect":
            detector = SPADetector()
            logger.info(f"Определение типа сайта для {args.url}")

            from seo_ai_models.parsers.utils.request_utils import fetch_url

            html_content, _, error = fetch_url(args.url)

            if error or not html_content:
                logger.error(f"Ошибка при получении {args.url}: {error}")
                return

            result = detector.analyze_html(html_content)

            print(f"URL: {args.url}")
            print(f"Является SPA: {result['is_spa']}")
            print(f"Уверенность: {result['confidence']:.2f}")

            if result["detected_frameworks"]:
                print(f"Обнаруженные фреймворки: {', '.join(result['detected_frameworks'])}")

            if result["dynamic_features"]:
                print(f"Динамические функции: {', '.join(result['dynamic_features'])}")

            save_results(result, args.output)

        elif args.command == "analyze":
            pipeline = AdaptiveParsingPipeline(force_spa_mode=args.force_spa)

            logger.info(f"Анализ URL: {args.url}")
            result = pipeline.analyze_url(args.url, not args.no_detect)

            if result["success"]:
                print(f"URL: {args.url}")
                if "site_type" in result:
                    print(f"Тип сайта: {'SPA' if result['site_type']['is_spa'] else 'обычный'}")

                if "content" in result and result["content"]:
                    print(f"Извлечено {len(result['content'].get('all_text', ''))} символов текста")
                    print(
                        f"Обнаружено {sum(len(headings) for headings in result['content'].get('headings', {}).values())} заголовков"
                    )

                if "metadata" in result and result["metadata"]:
                    print(f"Извлечено {len(result['metadata'])} метаэлементов")
            else:
                print(f"Ошибка при анализе {args.url}: {result.get('error', 'неизвестная ошибка')}")

            save_results(result, args.output)

        elif args.command == "crawl":
            pipeline = AdaptiveParsingPipeline(force_spa_mode=args.force_spa)

            logger.info(f"Сканирование сайта: {args.url}")
            result = pipeline.crawl_site(
                args.url, detect_type=not args.no_detect, max_pages=args.max_pages
            )

            if result["success"]:
                print(f"URL: {args.url}")
                print(f"Тип сайта: {'SPA' if result['site_type']['is_spa'] else 'обычный'}")
                print(f"Просканировано URL: {len(result['crawled_urls'])}")
                print(f"Найдено URL: {len(result['found_urls'])}")
                print(f"Неудачных URL: {len(result.get('failed_urls', []))}")
                print(f"Проанализировано контента: {len(result['content'])}")
            else:
                print(
                    f"Ошибка при сканировании {args.url}: {result.get('error', 'неизвестная ошибка')}"
                )

            save_results(result, args.output)

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main()
