"""
Тестирование интеграции с поисковым API для проекта SEO AI Models в Google Colab.
Скрипт демонстрирует возможности поиска и анализа поисковой выдачи.
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
    from seo_ai_models.parsers.unified.analyzers.search_api_integration import SearchAPIIntegration
except ImportError:
    logger.error("Не удалось импортировать компоненты SEO AI Models. Проверьте пути и установку пакета.")
    raise

def test_search_api(query, api_key=None, output_file=None, analyze_results=False, results_count=10):
    """
    Тестирует интеграцию с поисковым API.
    
    Args:
        query: Поисковый запрос
        api_key: API-ключ для поискового API (опционально)
        output_file: Файл для сохранения результатов (опционально)
        analyze_results: Анализировать контент топ-результатов
        results_count: Количество результатов для получения
    """
    print(f"\n{'='*60}")
    print(f"Тестирование поискового API для запроса: {query}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Определяем доступные API ключи
    api_keys = None
    if api_key:
        api_keys = [api_key]
    elif os.environ.get("SERPAPI_KEY"):
        api_keys = [os.environ.get("SERPAPI_KEY")]
    elif os.environ.get("SERPAPI_KEYS"):
        api_keys = os.environ.get("SERPAPI_KEYS").split(",")
    
    # Выбираем провайдера API
    if api_keys:
        api_provider = "serpapi"
        print(f"Используем API-ключи для провайдера: {api_provider}")
    else:
        api_provider = "custom"
        print("API-ключи не найдены, используется имитация поисковой выдачи")
    
    try:
        # Сначала протестируем напрямую SearchAPIIntegration
        logger.info(f"Инициализация SearchAPIIntegration для {api_provider}")
        search_api = SearchAPIIntegration(
            api_keys=api_keys,
            api_provider=api_provider,
            results_count=results_count
        )
        
        print(f"Выполнение поискового запроса: {query}")
        search_results = search_api.search(query)
        
        # Выводим общую информацию о результатах
        print(f"\nРезультаты поиска:")
        print(f"Запрос: {search_results.get('query', query)}")
        print(f"Найдено результатов: {len(search_results.get('results', []))}")
        
        # Выводим топ результаты
        results = search_results.get('results', [])
        print("\nТоп результаты:")
        for i, result in enumerate(results[:5], 1):
            print(f"\n  {i}. {result.get('title', 'Без заголовка')}")
            print(f"     URL: {result.get('url', 'Нет URL')}")
            print(f"     Сниппет: {result.get('snippet', 'Нет сниппета')[:100]}...")
        
        # Выводим связанные запросы
        related_queries = search_results.get('related_queries', [])
        if related_queries:
            print("\nСвязанные запросы:")
            for i, rel_query in enumerate(related_queries[:8], 1):
                print(f"  {i}. {rel_query}")
        
        # Выводим "Люди также спрашивают"
        paa = search_results.get('people_also_ask', [])
        if paa:
            print("\nЛюди также спрашивают:")
            for i, question in enumerate(paa[:3], 1):
                print(f"  {i}. {question.get('question', '')}")
        
        # Выводим информацию о SERP-функциях
        search_features = search_results.get('search_features', {})
        if search_features:
            print("\nSERP-функции:")
            for feature, present in search_features.items():
                if present:
                    print(f"  - {feature}")
        
        # Теперь тестируем через UnifiedParser, если нужен анализ контента
        if analyze_results:
            print(f"\n{'='*60}")
            print(f"Тестирование UnifiedParser.parse_search_results()")
            print(f"{'='*60}")
            
            parser = UnifiedParser(
                search_api_keys=api_keys,
                search_engine="google" if api_provider == "serpapi" else "custom"
            )
            
            print(f"Выполнение поискового запроса и анализ контента: {query}")
            parser_results = parser.parse_search_results(
                query, 
                results_count=min(5, results_count),  # Ограничиваем для анализа
                analyze_content=True
            )
            
            if parser_results.get("success", False) and "search_data" in parser_results:
                search_data = parser_results["search_data"]
                
                # Выводим результаты с анализом содержимого
                print("\nРезультаты с анализом содержимого:")
                for i, result in enumerate(search_data.get("results", [])[:3], 1):
                    print(f"\n  {i}. {result.get('title', 'Без заголовка')}")
                    print(f"     URL: {result.get('url', 'Нет URL')}")
                    
                    # Если есть детальный анализ, выводим его
                    detailed = result.get("detailed_analysis")
                    if detailed:
                        print(f"     * Слов на странице: {detailed.get('content', {}).get('word_count', 'Н/Д')}")
                        print(f"     * Заголовок H1: {detailed.get('structure', {}).get('headings', {}).get('h1', ['Нет'])[0] if 'h1' in detailed.get('structure', {}).get('headings', {}) and detailed.get('structure', {}).get('headings', {})['h1'] else 'Нет'}")
                        
                        # Метрики семантического анализа
                        semantic = detailed.get("performance", {}).get("semantic_analysis", {})
                        if semantic:
                            print(f"     * Семантическая плотность: {semantic.get('semantic_density', 0):.2f}")
                            print(f"     * Тематическая когерентность: {semantic.get('topical_coherence', 0):.2f}")
            else:
                print(f"Ошибка при использовании UnifiedParser: {parser_results.get('error', 'Неизвестная ошибка')}")
        
        # Сохраняем результаты если указан выходной файл
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(search_results, f, ensure_ascii=False, indent=2)
            print(f"\nРезультаты сохранены в: {output_file}")
    
    except Exception as e:
        logger.error(f"Ошибка при использовании поискового API: {str(e)}")
        print(f"Произошла ошибка: {str(e)}")
    
    # Выводим затраченное время
    elapsed_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {elapsed_time:.2f} секунд")

# Пример использования:
# test_search_api("seo optimization")
# test_search_api("machine learning frameworks", analyze_results=True)
