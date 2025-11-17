"""
Тестирование расширенных возможностей унифицированного парсера SEO AI Models.
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.parsers.unified.site_analyzer import SiteAnalyzer


def test_parser(args):
    """
    Запускает тесты унифицированного парсера с расширенными возможностями.
    """
    print(f"Тестирование расширенных возможностей унифицированного парсера")
    print("-" * 60)

    # Создаем экземпляр парсера с нужными опциями
    parser = UnifiedParser(
        auto_detect_spa=args.auto_detect_spa,
        force_spa_mode=args.force_spa,
        extract_semantic=args.semantic,
        parallel_parsing=args.parallel,
        max_workers=args.workers,
    )

    if args.url:
        print(f"1. Парсинг URL с новыми возможностями: {args.url}")
        result = parser.parse_url(args.url)

        if result.get("success", False):
            page_data = result.get("page_data", {})

            print(f"\nРезультаты парсинга:")
            print(f"URL: {page_data.get('url', '')}")
            print(f"Заголовок: {page_data.get('structure', {}).get('title', '')}")
            print(f"Слов: {page_data.get('content', {}).get('word_count', 0)}")

            # Выводим тип сайта
            site_type = page_data.get("html_stats", {}).get("site_type", {})
            print(
                f"Тип сайта: {'SPA' if site_type.get('is_spa', False) else 'Стандартный'} "
                + f"(уверенность: {site_type.get('confidence', 0):.2f})"
            )

            # Выводим результаты семантического анализа
            semantic = page_data.get("performance", {}).get("semantic_analysis", {})
            if semantic:
                print("\nСемантический анализ:")
                print(f"Семантическая плотность: {semantic.get('semantic_density', 0):.2f}")
                print(f"Семантический охват: {semantic.get('semantic_coverage', 0):.2f}")
                print(f"Тематическая когерентность: {semantic.get('topical_coherence', 0):.2f}")
                print(
                    f"Контекстуальная релевантность: {semantic.get('contextual_relevance', 0):.2f}"
                )

                # Выводим топ ключевые слова
                keywords = semantic.get("keywords", {})
                if keywords:
                    print("\nТоп ключевые слова:")
                    for word, weight in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[
                        :5
                    ]:
                        print(f"- {word}: {weight:.2f}")
        else:
            print(f"Ошибка при парсинге URL: {result.get('error', 'Неизвестная ошибка')}")

    if args.search:
        print(f"\n2. Тестирование поиска с API: {args.search}")
        search_result = parser.parse_search_results(args.search, results_count=5)

        if search_result.get("success", False) and "search_data" in search_result:
            search_data = search_result["search_data"]
            print(f"Найдено результатов: {len(search_data.get('results', []))}")

            # Выводим топ результаты
            for i, result in enumerate(search_data.get("results", [])[:3]):
                print(f"\nРезультат #{i+1}:")
                print(f"Позиция: {result.get('position', 0)}")
                print(f"Заголовок: {result.get('title', '')}")
                print(f"URL: {result.get('url', '')}")

            # Выводим связанные запросы
            related = search_data.get("related_queries", [])
            if related:
                print(f"\nСвязанные запросы: {', '.join(related[:5])}")
        else:
            print(f"Ошибка при поиске: {search_result.get('error', 'Неизвестная ошибка')}")

    if args.parallel:
        print(f"\n3. Тестирование многопоточного парсинга")
        urls = [f"https://example.com", f"https://google.com", f"https://github.com"]
        print(f"Парсинг {len(urls)} URL с использованием {args.workers} потоков")

        start_time = time.time()
        for url in urls:
            print(f"Парсинг {url}...")
            result = parser.parse_url(url)
            print(f"  Результат: {'Успешно' if result.get('success', False) else 'Ошибка'}")

        sequential_time = time.time() - start_time
        print(f"Время последовательного парсинга: {sequential_time:.2f} сек")

        # В реальности здесь был бы параллельный парсинг через crawler

    print("\nТестирование завершено успешно!")


def main():
    parser = argparse.ArgumentParser(
        description="Тестирование расширенных возможностей унифицированного парсера"
    )
    parser.add_argument("--url", help="URL для парсинга")
    parser.add_argument("--search", help="Поисковый запрос для тестирования API поиска")
    parser.add_argument(
        "--auto-detect-spa", action="store_true", help="Автоматически определять SPA-сайты"
    )
    parser.add_argument(
        "--force-spa", action="store_true", help="Принудительно использовать режим SPA"
    )
    parser.add_argument("--semantic", action="store_true", help="Включить семантический анализ")
    parser.add_argument("--parallel", action="store_true", help="Включить многопоточный парсинг")
    parser.add_argument("--workers", type=int, default=5, help="Количество потоков")
    parser.add_argument("--output", help="Файл для сохранения результатов (JSON)")

    args = parser.parse_args()

    # Если не указаны аргументы, используем пример
    if not any([args.url, args.search, args.parallel]):
        args.url = "https://example.com"
        args.semantic = True
        args.auto_detect_spa = True

    test_parser(args)


if __name__ == "__main__":
    main()
