"""
Тестирование компонентов для микро-бизнеса.

Скрипт тестирует компоненты многоуровневой системы для микро-бизнеса:
- MicroAdvisor
- StatisticalAnalyzer
- BasicRecommender
- LightweightOptimizer
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Импортируем компоненты для микро-бизнеса
from seo_ai_models.models.tiered_system.micro.micro_advisor import MicroAdvisor
from seo_ai_models.models.tiered_system.micro.statistical_analyzer import StatisticalAnalyzer
from seo_ai_models.models.tiered_system.micro.basic_recommender import BasicRecommender
from seo_ai_models.models.tiered_system.micro.lightweight_optimizer import LightweightOptimizer


def load_test_content():
    """Загружает тестовый контент."""
    test_content_path = "test_content.txt"
    
    # Если файл не существует, создаем тестовый контент
    if not os.path.exists(test_content_path):
        test_content = """
# Оптимизация веб-сайта для поисковых систем

Оптимизация для поисковых систем (SEO) - это процесс улучшения видимости сайта в поисковых системах. 
Правильно оптимизированный сайт имеет больше шансов попасть на высокие позиции в поисковой выдаче.

## Технические аспекты SEO

При оптимизации сайта необходимо учитывать множество технических факторов. Скорость загрузки страниц, 
мобильная адаптивность и правильная HTML-структура - все это влияет на позиции сайта в выдаче.

## Контентная оптимизация

Качественный контент - основа успешного SEO. Тексты должны быть информативными, полезными для пользователей
и оптимизированными под ключевые слова. При этом важно избегать переоптимизации и писать естественные тексты.

Ключевые слова следует равномерно распределять по тексту, включать их в заголовки и мета-теги.

## Ссылочная масса

Внешние ссылки остаются важным фактором ранжирования. Качественные ссылки с авторитетных сайтов 
могут существенно повысить позиции сайта в поисковой выдаче.

Для успешного продвижения сайта необходимо регулярно анализировать результаты и корректировать стратегию SEO.
        """
        with open(test_content_path, "w", encoding="utf-8") as f:
            f.write(test_content)
    
    # Читаем контент из файла
    with open(test_content_path, "r", encoding="utf-8") as f:
        return f.read()


def test_micro_advisor():
    """Тестирование MicroAdvisor."""
    logger.info("=== Тестирование MicroAdvisor ===")
    
    # Загружаем тестовый контент
    content = load_test_content()
    
    # Создаем MicroAdvisor с настройками для тестирования
    advisor = MicroAdvisor(config={
        'min_word_count': 100,
        'min_heading_count': 2,
        'min_paragraphs': 3
    })
    
    # Определяем ключевые слова
    keywords = ["SEO", "оптимизация", "поисковые системы", "ключевые слова"]
    
    # Анализируем контент
    logger.info("Анализ контента с помощью MicroAdvisor...")
    results = advisor.analyze_content(
        content=content,
        keywords=keywords,
        url="https://example.com/seo-guide"
    )
    
    # Выводим результаты
    logger.info(f"Базовые метрики: {results.get('basic_metrics', {})}")
    logger.info(f"Метрики ключевых слов: {results.get('keywords_basic', {})}")
    logger.info(f"Метрики читабельности: {results.get('readability', {})}")
    logger.info(f"Метрики структуры: {results.get('structure_basic', {})}")
    
    # Выводим рекомендации
    recommendations = results.get('core_recommendations', [])
    logger.info(f"Сгенерировано {len(recommendations)} рекомендаций:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. [{rec.get('priority', 'medium')}] {rec.get('recommendation', '')}")
    
    return results


def test_statistical_analyzer():
    """Тестирование StatisticalAnalyzer."""
    logger.info("=== Тестирование StatisticalAnalyzer ===")
    
    # Загружаем тестовый контент
    content = load_test_content()
    
    # Создаем StatisticalAnalyzer
    analyzer = StatisticalAnalyzer()
    
    # Определяем ключевые слова
    keywords = ["SEO", "оптимизация", "поисковые системы", "ключевые слова"]
    
    # Анализируем контент
    logger.info("Анализ контента с помощью StatisticalAnalyzer...")
    results = analyzer.analyze_content(
        content=content,
        keywords=keywords
    )
    
    # Выводим результаты
    logger.info(f"Извлеченные n-граммы: {results.get('ngrams', {}).get('1-grams', [])[:3]}")
    
    extracted_keywords = results.get('extracted_keywords', [])
    logger.info(f"Извлечено {len(extracted_keywords)} ключевых слов:")
    for i, kw in enumerate(extracted_keywords[:5], 1):
        logger.info(f"{i}. {kw.get('keyword', '')} (score: {kw.get('score', 0)})")
    
    topic_analysis = results.get('topic_analysis', {})
    logger.info(f"Анализ тематики: {topic_analysis.get('topic_focus_category', '')}")
    
    complexity = results.get('complexity_analysis', {})
    logger.info(f"Анализ сложности: {complexity.get('complexity_category', '')}")
    
    sentiment = results.get('sentiment_analysis', {})
    logger.info(f"Анализ тональности: {sentiment.get('sentiment_category', '')}")
    
    return results


def test_basic_recommender():
    """Тестирование BasicRecommender."""
    logger.info("=== Тестирование BasicRecommender ===")
    
    # Загружаем тестовый контент
    content = load_test_content()
    
    # Создаем BasicRecommender
    recommender = BasicRecommender()
    
    # Получаем метрики контента
    advisor = MicroAdvisor()
    advisor_results = advisor.analyze_content(
        content=content,
        keywords=["SEO", "оптимизация", "поисковые системы", "ключевые слова"]
    )
    
    # Генерируем рекомендации
    logger.info("Генерация рекомендаций с помощью BasicRecommender...")
    results = recommender.generate_recommendations(
        metrics=advisor_results.get('basic_metrics', {}),
        keywords_metrics=advisor_results.get('keywords_basic', {}),
        structure_metrics=advisor_results.get('structure_basic', {}),
        readability_metrics=advisor_results.get('readability', {}),
        content_type='article',
        industry='technology'
    )
    
    # Выводим результаты
    recommendations = results.get('recommendations', [])
    logger.info(f"Сгенерировано {len(recommendations)} рекомендаций:")
    for i, rec in enumerate(recommendations[:5], 1):
        logger.info(f"{i}. [{rec.get('priority', 'medium')}] {rec.get('title', '')}")
    
    categorized = results.get('categorized_recommendations', {})
    logger.info(f"Рекомендации по категориям: {list(categorized.keys())}")
    
    logger.info(f"Статистика рекомендаций:")
    logger.info(f"  - Высокий приоритет: {results.get('high_priority_count', 0)}")
    logger.info(f"  - Средний приоритет: {results.get('medium_priority_count', 0)}")
    logger.info(f"  - Низкий приоритет: {results.get('low_priority_count', 0)}")
    
    return results


def test_lightweight_optimizer():
    """Тестирование LightweightOptimizer."""
    logger.info("=== Тестирование LightweightOptimizer ===")
    
    # Загружаем тестовый контент
    content = load_test_content()
    
    # Создаем LightweightOptimizer
    optimizer = LightweightOptimizer()
    
    # Определяем ключевые слова
    keywords = ["SEO", "оптимизация", "поисковые системы", "ключевые слова"]
    
    # Получаем рекомендации
    advisor = MicroAdvisor()
    advisor_results = advisor.analyze_content(
        content=content,
        keywords=keywords
    )
    recommendations = advisor_results.get('core_recommendations', [])
    
    # Оптимизируем контент
    logger.info("Оптимизация контента с помощью LightweightOptimizer...")
    results = optimizer.optimize_content(
        content=content,
        keywords=keywords,
        recommendations=recommendations
    )
    
    # Выводим результаты
    changes = results.get('changes', [])
    logger.info(f"Выполнено {len(changes)} оптимизаций:")
    for i, change in enumerate(changes, 1):
        logger.info(f"{i}. [{change.get('type', '')}] {change.get('description', '')}")
    
    stats = results.get('optimization_stats', {})
    logger.info(f"Статистика оптимизации:")
    logger.info(f"  - Изменение длины: {stats.get('length_change', 0)} символов ({stats.get('length_change_percent', 0)}%)")
    logger.info(f"  - Изменение ключевых слов: +{stats.get('keyword_change', 0)}")
    
    meta_title = results.get('meta_title', '')
    meta_description = results.get('meta_description', '')
    logger.info(f"Мета-заголовок: {meta_title}")
    logger.info(f"Мета-описание: {meta_description}")
    
    return results


def main():
    """Основная функция тестирования."""
    logger.info("Запуск тестирования компонентов для микро-бизнеса")
    
    # Тестируем все компоненты
    advisor_results = test_micro_advisor()
    analyzer_results = test_statistical_analyzer()
    recommender_results = test_basic_recommender()
    optimizer_results = test_lightweight_optimizer()
    
    logger.info("Тестирование завершено")


if __name__ == "__main__":
    main()
