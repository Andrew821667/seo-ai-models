"""
Демонстрационный скрипт для SERP-анализа LLM-поисковиков.

Скрипт демонстрирует использование компонентов SERP-анализа для:
1. Анализа результатов LLM-поисковиков
2. Отслеживания цитирования контента
3. Сравнения с конкурентами
4. Бенчмаркинга по отраслям
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

from seo_ai_models.models.llm_integration.service.llm_service import LLMService
from seo_ai_models.models.llm_integration.service.prompt_generator import PromptGenerator
from seo_ai_models.models.llm_integration.service.cost_estimator import CostEstimator
from seo_ai_models.models.llm_integration.serp_analysis.llm_serp_analyzer import LLMSerpAnalyzer
from seo_ai_models.models.llm_integration.serp_analysis.citation_analyzer import CitationAnalyzer
from seo_ai_models.models.llm_integration.serp_analysis.competitor_tracking import CompetitorTracking
from seo_ai_models.models.llm_integration.serp_analysis.industry_benchmarker import IndustryBenchmarker

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_services(openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Настраивает сервисы для демонстрации.
    
    Args:
        openai_api_key: API ключ OpenAI (опционально)
        
    Returns:
        Dict[str, Any]: Словарь с сервисами
    """
    # Если API ключ не указан, пытаемся получить его из переменной окружения
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Создаем экземпляры базовых сервисов
    llm_service = LLMService(openai_api_key=openai_api_key)
    prompt_generator = PromptGenerator()
    cost_estimator = CostEstimator()
    
    # Создаем экземпляры компонентов SERP-анализа
    serp_analyzer = LLMSerpAnalyzer(llm_service, prompt_generator)
    citation_analyzer = CitationAnalyzer(llm_service, prompt_generator)
    competitor_tracking = CompetitorTracking(llm_service, prompt_generator)
    industry_benchmarker = IndustryBenchmarker(llm_service, prompt_generator)
    
    # Возвращаем словарь с сервисами
    return {
        "llm_service": llm_service,
        "prompt_generator": prompt_generator,
        "cost_estimator": cost_estimator,
        "serp_analyzer": serp_analyzer,
        "citation_analyzer": citation_analyzer,
        "competitor_tracking": competitor_tracking,
        "industry_benchmarker": industry_benchmarker
    }

def demonstrate_serp_analyzer(services: Dict[str, Any], query: str, content: str) -> None:
    """
    Демонстрирует работу LLMSerpAnalyzer.
    
    Args:
        services: Словарь с сервисами
        query: Поисковый запрос
        content: Контент для анализа
    """
    logger.info("=== Демонстрация LLMSerpAnalyzer ===")
    
    serp_analyzer = services["serp_analyzer"]
    
    # Анализируем SERP для запроса и контента
    result = serp_analyzer.analyze_serp(
        query=query,
        content=content,
        llm_engines=["perplexity", "claude_search"],
        num_samples=1,  # Уменьшаем для демонстрации
        budget=50.0     # Ограничиваем бюджет для демонстрации
    )
    
    # Выводим основные результаты
    logger.info(f"Запрос: {result.get('query')}")
    logger.info(f"Поисковики: {', '.join(result.get('engines', []))}")
    logger.info(f"Частота цитирования: {result.get('citation_rate', 0):.2%}")
    logger.info(f"Оценка видимости: {result.get('visibility_score', 0):.2f}")
    
    # Выводим результаты по поисковикам
    for engine, engine_result in result.get("engines_results", {}).items():
        logger.info(f"--- Результаты для {engine} ---")
        logger.info(f"Частота цитирования: {engine_result.get('citation_rate', 0):.2%}")
        logger.info(f"Видимость: {engine_result.get('visibility_score', 0):.2f}")
    
    # Сравниваем два варианта контента
    if len(content) > 500:
        alternative_content = content[:len(content)//2] + "\nЭто альтернативный вариант контента."
        
        logger.info("\n=== Сравнение вариантов контента ===")
        comparison_result = serp_analyzer.compare_serp_results(
            query=query,
            contents=[content, alternative_content],
            llm_engines=["perplexity"],
            num_samples=1,
            budget=30.0
        )
        
        logger.info(f"Лучший вариант: #{comparison_result.get('best_variant', {}).get('index', 0) + 1}")
        logger.info(f"Видимость лучшего варианта: {comparison_result.get('best_variant', {}).get('visibility_score', 0):.2f}")
        logger.info(f"Оценки видимости вариантов: {[f'{score:.2f}' for score in comparison_result.get('visibility_scores', [])]}")

def demonstrate_citation_analyzer(services: Dict[str, Any], content: str, queries: List[str]) -> None:
    """
    Демонстрирует работу CitationAnalyzer.
    
    Args:
        services: Словарь с сервисами
        content: Контент для анализа
        queries: Список поисковых запросов
    """
    logger.info("\n=== Демонстрация CitationAnalyzer ===")
    
    citation_analyzer = services["citation_analyzer"]
    
    # Анализируем цитирование контента для запросов
    result = citation_analyzer.analyze_citation(
        content=content,
        queries=queries,
        llm_engines=["perplexity"],
        num_samples=1,
        budget=50.0
    )
    
    # Выводим основные результаты
    logger.info(f"Количество запросов: {result.get('queries_count')}")
    logger.info(f"Частота цитирования: {result.get('citation_rate', 0):.2%}")
    logger.info(f"Оценка видимости: {result.get('visibility_score', 0):.2f}")
    logger.info(f"Оценка цитируемости: {result.get('citability_score', 0)}/10")
    
    # Выводим результаты по запросам
    for query_result in result.get("queries_results", []):
        logger.info(f"--- Результаты для запроса: {query_result.get('query')} ---")
        logger.info(f"Частота цитирования: {query_result.get('citation_rate', 0):.2%}")
        logger.info(f"Видимость: {query_result.get('visibility_score', 0):.2f}")
    
    # Выводим факторы цитирования
    logger.info("\n--- Факторы цитирования ---")
    citation_factors = result.get("citation_factors", {})
    
    # Структура контента
    content_structure = citation_factors.get("content_structure", {})
    logger.info(f"Длина контента: {content_structure.get('content_length', 0)} символов")
    logger.info(f"Количество заголовков: {content_structure.get('headers_count', 0)}")
    logger.info(f"Количество параграфов: {content_structure.get('paragraphs_count', 0)}")
    
    # Ключевые слова
    keywords_info = citation_factors.get("keywords", {})
    logger.info(f"Плотность ключевых слов: {keywords_info.get('keyword_density', 0):.2%}")
    
    # Корреляции
    correlation_info = citation_factors.get("citation_correlation", {})
    logger.info(f"Корреляция длины: {correlation_info.get('length_correlation', 0):.2f}")
    logger.info(f"Корреляция плотности ключевых слов: {correlation_info.get('keyword_density_correlation', 0):.2f}")

def demonstrate_competitor_tracking(services: Dict[str, Any], query: str, our_content: str, competitors: List[Dict[str, Any]]) -> None:
    """
    Демонстрирует работу CompetitorTracking.
    
    Args:
        services: Словарь с сервисами
        query: Поисковый запрос
        our_content: Наш контент
        competitors: Список данных о конкурентах
    """
    logger.info("\n=== Демонстрация CompetitorTracking ===")
    
    competitor_tracking = services["competitor_tracking"]
    
    # Отслеживаем цитируемость конкурентов
    result = competitor_tracking.track_competitors(
        query=query,
        competitors=competitors,
        our_content=our_content,
        llm_engines=["perplexity"],
        num_samples=1,
        budget=50.0
    )
    
    # Выводим основные результаты
    logger.info(f"Запрос: {result.get('query')}")
    logger.info(f"Количество конкурентов: {result.get('competitors_count')}")
    
    # Выводим рейтинг
    logger.info("\n--- Рейтинг ---")
    for ranked_item in result.get("ranking", []):
        name = ranked_item.get("name", "")
        visibility = ranked_item.get("visibility_score", 0)
        is_our = "(Наш контент)" if ranked_item.get("is_our_content", False) else ""
        logger.info(f"{ranked_item.get('rank')}. {name} {is_our} - Видимость: {visibility:.2f}")
    
    # Выводим информацию о топовом конкуренте
    top_competitor = result.get("competitor_strategies", {}).get("top_competitor", {})
    if top_competitor:
        logger.info(f"\n--- Топовый конкурент: {top_competitor.get('name')} ---")
        logger.info(f"Видимость: {top_competitor.get('visibility_score', 0):.2f}")
        logger.info(f"Частота цитирования: {top_competitor.get('citation_rate', 0):.2%}")
        logger.info(f"Длина контента: {top_competitor.get('content_length', 0)} символов")
    
    # Выводим рекомендации
    logger.info("\n--- Рекомендации ---")
    for recommendation in result.get("competitor_strategies", {}).get("recommendations", []):
        logger.info(f"- {recommendation}")

def demonstrate_industry_benchmarker(services: Dict[str, Any], content: str, queries: List[str], industry: str, content_type: str) -> None:
    """
    Демонстрирует работу IndustryBenchmarker.
    
    Args:
        services: Словарь с сервисами
        content: Контент для анализа
        queries: Список поисковых запросов
        industry: Отрасль
        content_type: Тип контента
    """
    logger.info("\n=== Демонстрация IndustryBenchmarker ===")
    
    industry_benchmarker = services["industry_benchmarker"]
    
    # Получаем бенчмарки для отрасли и типа контента
    benchmarks = industry_benchmarker.get_industry_benchmarks(industry, content_type)
    
    logger.info(f"Отрасль: {benchmarks.get('industry')}")
    logger.info(f"Тип контента: {benchmarks.get('content_type')}")
    
    # Выводим основные бенчмарки
    benchmark_data = benchmarks.get("benchmarks", {})
    logger.info(f"Бенчмарк частоты цитирования: {benchmark_data.get('citation_rate', 0):.2%}")
    logger.info(f"Бенчмарк видимости: {benchmark_data.get('visibility_score', 0):.2f}")
    
    # Выводим бенчмарки метрик контента
    content_metrics = benchmark_data.get("content_metrics", {})
    logger.info(f"Минимальная длина контента: {content_metrics.get('min_content_length', 0)} символов")
    logger.info(f"Оптимальная длина контента: {content_metrics.get('optimal_content_length', 0)} символов")
    logger.info(f"Оптимальное количество заголовков: {content_metrics.get('optimal_headers_count', 0)}")
    
    # Анализируем контент относительно бенчмарков
    logger.info("\n--- Анализ контента относительно бенчмарков ---")
    result = industry_benchmarker.analyze_industry_benchmarks(
        content=content,
        queries=queries,
        industry=industry,
        content_type=content_type,
        llm_engines=["perplexity"],
        num_samples=1,
        budget=50.0
    )
    
    # Выводим основные результаты
    logger.info(f"Частота цитирования: {result.get('citation_rate', 0):.2%} (vs бенчмарк: {result.get('citation_rate_vs_benchmark', 0):.2f})")
    logger.info(f"Видимость: {result.get('visibility_score', 0):.2f} (vs бенчмарк: {result.get('visibility_vs_benchmark', 0):.2f})")
    
    # Выводим анализ относительно бенчмарков
    logger.info("\n--- Анализ относительно бенчмарков ---")
    benchmark_analysis = result.get("benchmark_analysis", {})
    logger.info(f"Общий анализ: {benchmark_analysis.get('overall', '')}")
    logger.info(f"Анализ частоты цитирования: {benchmark_analysis.get('citation_rate', '')}")
    logger.info(f"Анализ видимости: {benchmark_analysis.get('visibility_score', '')}")
    logger.info(f"Анализ длины контента: {benchmark_analysis.get('content_length', '')}")
    
    # Выводим рекомендации
    logger.info("\n--- Рекомендации ---")
    for recommendation in result.get("recommendations", []):
        logger.info(f"- {recommendation}")

def get_sample_data() -> Dict[str, Any]:
    """
    Создает тестовые данные для демонстрации.
    
    Returns:
        Dict[str, Any]: Тестовые данные
    """
    # Пример контента для анализа
    our_content = """
    # Руководство по SEO-оптимизации для LLM-поисковиков
    
    Поисковые системы, основанные на больших языковых моделях (LLM), становятся все более популярными. В отличие от традиционных поисковиков, LLM-поисковики не просто показывают список результатов, а генерируют ответы на основе найденной информации.
    
    ## Что такое LLM-поисковики
    
    LLM-поисковики, такие как Perplexity, Claude Search, You.com и другие, используют большие языковые модели для генерации ответов на запросы пользователей. Они анализируют контент из различных источников и формируют связный ответ.
    
    ## Особенности оптимизации для LLM-поисковиков
    
    ### 1. Структура и ясность контента
    
    Для LLM-поисковиков особенно важна четкая структура контента. Используйте:
    - Информативные заголовки H1, H2, H3
    - Короткие, ясные параграфы
    - Маркированные и нумерованные списки
    - Таблицы для сравнения данных
    
    ### 2. Фактическая точность и цитируемость
    
    LLM-модели стремятся предоставлять точную информацию. Повысьте вероятность цитирования вашего контента:
    - Включайте конкретные факты и статистику
    - Указывайте источники информации
    - Используйте актуальные данные
    - Приводите примеры из практики
    
    ### 3. Полнота и экспертность
    
    Создавайте контент, демонстрирующий глубокое понимание темы:
    - Рассматривайте вопрос с разных сторон
    - Отвечайте на сопутствующие вопросы
    - Приводите экспертные мнения
    - Объясняйте сложные концепции простым языком
    
    ## Практические рекомендации
    
    1. **Оптимизируйте для конкретных запросов** - определите, на какие вопросы должен отвечать ваш контент
    2. **Регулярно обновляйте информацию** - LLM ценят актуальность
    3. **Используйте таблицы и визуализации** - они повышают информативность
    4. **Проверяйте видимость в LLM-поисковиках** - анализируйте, как часто ваш контент цитируется
    
    Следуя этим рекомендациям, вы сможете повысить видимость и цитируемость вашего контента в новом поколении поисковых систем.
    """
    
    # Пример контента конкурентов
    competitor1_content = """
    # SEO для LLM-поисковых систем: новый подход к оптимизации
    
    В 2024 году поисковые системы на основе больших языковых моделей (LLM) изменили подход к поиску информации. Вместо простого предоставления списка ссылок, они генерируют прямые ответы на вопросы пользователей, цитируя различные источники.
    
    ## Как работают LLM-поисковики
    
    LLM-поисковики анализируют контент из интернета и формируют связные ответы, ссылаясь на наиболее надежные и релевантные источники. Ключевые игроки на этом рынке:
    - Perplexity AI
    - Google SGE
    - Bing Copilot
    - Claude Search
    - You.com
    
    ## Ключевые факторы цитируемости в LLM-поисковиках
    
    1. **Достоверность информации** - LLM отдают предпочтение контенту с точными фактами и данными
    2. **Структурированность** - четкая организация с заголовками, списками и таблицами
    3. **Полнота охвата темы** - всесторонний анализ вопроса
    4. **Уникальность и оригинальность** - новые исследования и данные
    5. **Авторитетность источника** - экспертность и репутация автора
    
    ## Стратегии оптимизации для LLM
    
    ### Техническая оптимизация
    - Используйте семантическую разметку Schema.org
    - Обеспечьте быструю загрузку страниц
    - Создайте четкую структуру сайта
    
    ### Контентная оптимизация
    - Отвечайте на конкретные вопросы в начале статьи
    - Используйте подзаголовки в виде вопросов
    - Включайте данные исследований и статистику
    - Создавайте сравнительные таблицы
    
    ## Заключение
    
    Оптимизация для LLM-поисковиков требует более глубокого подхода к созданию контента. Фокусируйтесь на экспертности, структуре и информативности, чтобы максимизировать цитируемость вашего контента.
    """
    
    competitor2_content = """
    # Повышение видимости сайта в LLM-поисковиках в 2024 году
    
    LLM-поисковики кардинально меняют ландшафт поисковой оптимизации. Вместо традиционной парадигмы "10 синих ссылок" пользователи получают прямые ответы, сгенерированные искусственным интеллектом на основе информации из различных источников.
    
    Наше исследование, проведенное на 500 сайтах, показало, что 78% контента, цитируемого в LLM-поисковиках, соответствует определенным паттернам, которые мы разберем в этой статье.
    
    ## Что такое "цитируемость" и почему это важно
    
    Цитируемость - это вероятность того, что ваш контент будет использован LLM-поисковиком при формировании ответа. Фактически, это новая "первая позиция" в поисковой выдаче. Наши данные показывают, что цитируемый контент получает в 3.5 раза больше переходов, чем нецитируемый.
    
    ## Ключевые метрики цитируемости
    
    | Метрика | Влияние на цитируемость |
    |---------|-------------------------|
    | Фактическая точность | Очень высокое |
    | Структурированность | Высокое |
    | Уникальность данных | Высокое |
    | Авторитетность домена | Среднее |
    | Скорость загрузки | Низкое |
    
    ## Практические шаги по повышению цитируемости
    
    1. **Структурируйте контент вокруг вопросов**
       * Используйте вопросы в H2 и H3 заголовках
       * Давайте прямые ответы сразу после вопросов
       * Добавляйте связанные вопросы и ответы
    
    2. **Добавляйте уникальные данные**
       * Проводите собственные исследования и опросы
       * Включайте оригинальную статистику
       * Анализируйте данные и делайте выводы
    
    3. **Усильте формат контента**
       * Создавайте сравнительные таблицы
       * Используйте маркированные списки для перечислений
       * Добавляйте тезисы и выводы после каждого раздела
    
    Наше исследование показало, что внедрение этих практик повышает вероятность цитирования контента в LLM-поисковиках на 65-120%.
    """
    
    # Тестовые данные
    return {
        "our_content": our_content,
        "competitors": [
            {
                "id": "competitor1",
                "name": "SEO Expert Blog",
                "content": competitor1_content
            },
            {
                "id": "competitor2",
                "name": "Digital Marketing Research",
                "content": competitor2_content
            }
        ],
        "query": "как оптимизировать контент для LLM-поисковиков",
        "queries": [
            "как оптимизировать контент для LLM-поисковиков",
            "что такое LLM-поисковики",
            "факторы цитируемости в поисковых системах с ИИ"
        ],
        "industry": "technology",
        "content_type": "how_to_guide"
    }

def main():
    parser = argparse.ArgumentParser(description="Демонстрация SERP-анализа для LLM-поисковиков")
    parser.add_argument("--api_key", help="API ключ OpenAI")
    parser.add_argument("--demo", choices=["all", "serp", "citation", "competitors", "benchmarks"],
                      default="all", help="Какую демонстрацию запустить")
    
    args = parser.parse_args()
    
    # Настраиваем сервисы
    services = setup_services(args.api_key)
    
    # Получаем тестовые данные
    sample_data = get_sample_data()
    
    # Запускаем выбранную демонстрацию
    if args.demo in ["all", "serp"]:
        demonstrate_serp_analyzer(services, sample_data["query"], sample_data["our_content"])
    
    if args.demo in ["all", "citation"]:
        demonstrate_citation_analyzer(services, sample_data["our_content"], sample_data["queries"])
    
    if args.demo in ["all", "competitors"]:
        demonstrate_competitor_tracking(services, sample_data["query"], sample_data["our_content"], sample_data["competitors"])
    
    if args.demo in ["all", "benchmarks"]:
        demonstrate_industry_benchmarker(services, sample_data["our_content"], sample_data["queries"], 
                                     sample_data["industry"], sample_data["content_type"])
    
    logger.info("\n=== Демонстрация завершена ===")

if __name__ == "__main__":
    main()
