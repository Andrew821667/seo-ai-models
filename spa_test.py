"""
Тестирование расширенного SPA-краулера для проекта SEO AI Models в Google Colab.
Скрипт демонстрирует возможности парсинга одностраничных приложений (SPA).
"""

import os
import sys
import json
import time
import logging
from pprint import pprint

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Проверяем, что мы находимся в директории проекта
if not os.path.exists('/content/seo-ai-models'):
    logger.error("Директория проекта не найдена. Убедитесь, что репозиторий клонирован в /content/seo-ai-models")
    raise RuntimeError("Репозиторий проекта не найден")

# Переходим в директорию проекта
os.chdir('/content/seo-ai-models')

# Добавляем директорию проекта в PYTHONPATH
if '/content/seo-ai-models' not in sys.path:
    sys.path.insert(0, '/content/seo-ai-models')

try:
    from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
    from seo_ai_models.parsers.unified.crawlers.enhanced_spa_crawler import EnhancedSPACrawler
except ImportError:
    logger.error("Не удалось импортировать компоненты SEO AI Models. Проверьте пути и установку пакета.")
    raise

def test_spa_parsing(url, output_file=None, headless=True, wait_time=3000):
    """
    Тестирует парсинг SPA-сайта.
    
    Args:
        url: URL SPA-сайта для парсинга
        output_file: Файл для сохранения результатов (опционально)
        headless: Запускать браузер в фоновом режиме
        wait_time: Время ожидания загрузки контента в мс
    """
    print(f"\n{'='*60}")
    print(f"Тестирование парсинга SPA-сайта: {url}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Инициализируем парсер с принудительным SPA-режимом
    parser = UnifiedParser(
        force_spa_mode=True,
        spa_settings={
            "headless": headless,
            "wait_for_idle": wait_time,
            "wait_for_timeout": wait_time + 5000,
            "browser_type": "chromium",
            "intercept_ajax": True
        },
        extract_semantic=True
    )
    
    logger.info(f"Запуск парсинга SPA-сайта: {url}")
    
    try:
        # Парсим URL с помощью унифицированного парсера
        result = parser.parse_url(url)
        
        if result.get("success", False):
            page_data = result.get("page_data", {})
            
            # Выводим основную информацию
            print(f"\nОсновные результаты парсинга SPA:")
            print(f"URL: {page_data.get('url', '')}")
            print(f"Заголовок: {page_data.get('structure', {}).get('title', '')}")
            print(f"Слов в контенте: {page_data.get('content', {}).get('word_count', 0)}")
            
            # Выводим типы JavaScript-фреймворков на странице (если есть)
            site_type = page_data.get("html_stats", {}).get("site_type", {})
            print(f"\nДетекция SPA:")
            print(f"Уверенность: {site_type.get('confidence', 0):.2f}")
            print(f"Метод определения: {site_type.get('detection_method', 'Неизвестно')}")
            
            # Выводим структуру заголовков
            headings = page_data.get("structure", {}).get("headings", {})
            if headings:
                print("\nСтруктура заголовков:")
                for level, titles in headings.items():
                    if titles:
                        print(f"  {level.upper()}: {len(titles)} заголовков")
                        for i, title in enumerate(titles[:3], 1):
                            print(f"    {i}. {title[:50]}{'...' if len(title) > 50 else ''}")
                        if len(titles) > 3:
                            print(f"    ... и еще {len(titles) - 3}")
            
            # Выводим ссылки
            links = page_data.get("structure", {}).get("links", {})
            internal_links = len(links.get("internal", []))
            external_links = len(links.get("external", []))
            print(f"\nСсылки: {internal_links + external_links} всего")
            print(f"  Внутренние: {internal_links}")
            print(f"  Внешние: {external_links}")
            
            # Выводим информацию о JavaScript-ресурсах (если есть)
            html_stats = page_data.get("html_stats", {})
            if "js_resources" in html_stats:
                js_resources = html_stats.get("js_resources", [])
                print(f"\nJavaScript-ресурсы: {len(js_resources)}")
                for i, res in enumerate(js_resources[:3], 1):
                    print(f"  {i}. {res[:50]}{'...' if len(res) > 50 else ''}")
                if len(js_resources) > 3:
                    print(f"  ... и еще {len(js_resources) - 3}")
            
            # Выводим семантический анализ
            semantic = page_data.get("performance", {}).get("semantic_analysis", {})
            if semantic:
                print("\nСемантический анализ:")
                print(f"  Семантическая плотность: {semantic.get('semantic_density', 0):.2f}")
                print(f"  Семантический охват: {semantic.get('semantic_coverage', 0):.2f}")
                print(f"  Тематическая когерентность: {semantic.get('topical_coherence', 0):.2f}")
                
                # Выводим топ ключевые слова
                keywords = semantic.get("keywords", {})
                if keywords:
                    print("\nТоп ключевые слова:")
                    for word, weight in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"  - {word}: {weight:.2f}")
            
            # Сохраняем результаты если указан выходной файл
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\nРезультаты сохранены в: {output_file}")
        else:
            print(f"Ошибка при парсинге SPA-сайта: {result.get('error', 'Неизвестная ошибка')}")
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении парсинга SPA: {str(e)}")
        print(f"Произошла ошибка: {str(e)}")
    
    # Выводим затраченное время
    elapsed_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {elapsed_time:.2f} секунд")

def test_direct_spa_crawler(url, max_pages=5, output_file=None, headless=True):
    """
    Напрямую тестирует EnhancedSPACrawler для сканирования SPA-сайта.
    
    Args:
        url: URL SPA-сайта для сканирования
        max_pages: Максимальное количество страниц для сканирования
        output_file: Файл для сохранения результатов (опционально)
        headless: Запускать браузер в фоновом режиме
    """
    print(f"\n{'='*60}")
    print(f"Тестирование EnhancedSPACrawler для сайта: {url}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Создаем экземпляр SPA-краулера
        crawler = EnhancedSPACrawler(
            base_url=url,
            max_pages=max_pages,
            max_depth=2,  # Ограничиваем глубину для тестирования
            delay=1.0,
            headless=headless,
            wait_for_idle=3000,
            wait_for_timeout=10000,
            browser_type="chromium",
            intercept_ajax=True
        )
        
        logger.info(f"Запуск SPA-краулера для: {url}")
        print(f"Сканирование сайта {url} (ограничение: {max_pages} страниц)")
        
        # Выполняем сканирование
        crawl_result = crawler.crawl()
        
        # Выводим результаты
        print(f"\nРезультаты сканирования SPA:")
        print(f"Найдено URL: {len(crawl_result.get('found_urls', []))}")
        print(f"Просканировано URL: {len(crawl_result.get('crawled_urls', []))}")
        print(f"Неудачных URL: {len(crawl_result.get('failed_urls', []))}")
        
        # Выводим список просканированных URL
        print("\nПросканированные URL:")
        for i, url in enumerate(crawl_result.get('crawled_urls', [])[:5], 1):
            print(f"  {i}. {url}")
        
        if len(crawl_result.get('crawled_urls', [])) > 5:
            print(f"  ... и еще {len(crawl_result.get('crawled_urls', [])) - 5}")
        
        # Информация о AJAX-запросах
        ajax_requests = crawl_result.get('ajax_requests', {})
        if ajax_requests:
            print(f"\nПерехваченные AJAX-запросы: {len(ajax_requests)}")
            for i, (url, data) in enumerate(list(ajax_requests.items())[:3], 1):
                print(f"  {i}. {data.get('method', 'GET')} {url[:50]}...")
            
            if len(ajax_requests) > 3:
                print(f"  ... и еще {len(ajax_requests) - 3}")
        
        # Сохраняем результаты если указан выходной файл
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(crawl_result, f, ensure_ascii=False, indent=2)
            print(f"\nРезультаты сохранены в: {output_file}")
    
    except Exception as e:
        logger.error(f"Ошибка при использовании SPA-краулера: {str(e)}")
        print(f"Произошла ошибка: {str(e)}")
    
    # Выводим затраченное время
    elapsed_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {elapsed_time:.2f} секунд")

# Пример использования:
# test_spa_parsing("https://react-redux.realworld.io/")
# test_direct_spa_crawler("https://reactjs.org", max_pages=3)
