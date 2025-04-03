"""Демонстрация улучшенных компонентов SEO AI Models.

Этот скрипт демонстрирует использование улучшенных компонентов SEO AI Models:
- Улучшенный TextProcessor
- Расширенный EEATAnalyzer
- EnhancedEEATAnalyzer с ML
- Улучшенный ContentAnalyzer
"""

import sys
import os
import json
from pathlib import Path
from pprint import pprint

# Добавляем корень проекта в путь импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем улучшенные компоненты
from seo_ai_models.common.utils.text_processing import TextProcessor
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.enhanced_eeat_analyzer import EnhancedEEATAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor


def pretty_print_dict(d, title=None, max_depth=1, current_depth=0):
    """Красивый вывод словаря с ограничением глубины."""
    if title:
        print(f"\n{'-' * 40}\n{title}\n{'-' * 40}")
    
    if not isinstance(d, dict) or current_depth >= max_depth:
        print(d)
        return
    
    for key, value in d.items():
        if isinstance(value, dict) and current_depth < max_depth - 1:
            print(f"{key}:")
            pretty_print_dict(value, title=None, max_depth=max_depth, current_depth=current_depth+1)
        elif isinstance(value, list) and value and isinstance(value[0], dict) and current_depth < max_depth - 1:
            print(f"{key}: [список из {len(value)} элементов]")
            if len(value) > 0:
                pretty_print_dict(value[0], title=None, max_depth=max_depth, current_depth=current_depth+1)
        else:
            print(f"{key}: {value}")


def demo_text_processor():
    """Демонстрация улучшенного TextProcessor."""
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ УЛУЧШЕННОГО TEXTPROCESSOR")
    print("=" * 80)
    
    # Создаем экземпляр TextProcessor
    processor = TextProcessor()
    
    # Тестовый текст
    test_text = """
    # Руководство по SEO-оптимизации в 2025 году
    
    В этой статье рассмотрены актуальные методы SEO-оптимизации, 
    которые будут работать в 2025 году. Мы проанализировали последние данные
    и изменения в алгоритмах поисковых систем.
    
    ## Важность E-E-A-T в 2025 году
    
    Google продолжает усиливать значимость E-E-A-T (Experience, Expertise, 
    Authoritativeness, Trustworthiness) при ранжировании сайтов.
    
    ### Как улучшить E-E-A-T
    
    * Добавьте информацию об авторах
    * Цитируйте авторитетные источники
    * Регулярно обновляйте контент
    
    ## Технические факторы
    
    Технические аспекты SEO остаются важными:
    
    1. Скорость загрузки сайта
    2. Мобильная оптимизация
    3. Структурированные данные
    
    Подробнее читайте в нашем исследовании: https://example.com/seo-research-2025
    
    Последнее обновление: 15 марта 2025 года
    Автор: SEO эксперт с 10-летним опытом
    """
    
    # Определение языка
    language = processor.detect_language(test_text)
    print(f"Определенный язык текста: {language}")
    
    # Токенизация
    tokens = processor.tokenize(test_text, remove_stopwords=True)
    print(f"\nТокены (первые 10): {tokens[:10]}")
    print(f"Всего токенов: {len(tokens)}")
    
    # Лемматизация
    lemmas = processor.lemmatize(tokens[:10], language=language)
    print(f"\nЛемматизированные токены: {lemmas}")
    
    # Анализ структуры
    structure = processor.analyze_text_structure(test_text)
    pretty_print_dict(structure, "Анализ структуры текста")
    
    # Извлечение заголовков
    headers = processor.extract_headers(test_text)
    print("\nИзвлеченные заголовки:")
    for header in headers:
        print(f"Уровень {header['level']}: {header['text']}")
    
    # Расчет читабельности
    readability = processor.calculate_readability(test_text, language)
    pretty_print_dict(readability, "Метрики читабельности")
    
    # Извлечение ключевых слов
    keywords = processor.extract_keywords(test_text, max_keywords=5, language=language)
    print("\nИзвлеченные ключевые слова:")
    for keyword, weight in keywords:
        print(f"{keyword}: {weight:.2f}")


def demo_eeat_analyzer():
    """Демонстрация улучшенного EEATAnalyzer."""
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ УЛУЧШЕННОГО EEAT ANALYZER")
    print("=" * 80)
    
    # Создаем экземпляр EEATAnalyzer
    analyzer = EEATAnalyzer()
    
    # Тестовый текст
    test_text = """
    # Как правильно инвестировать в акции в 2025 году
    
    По данным исследования Morgan Stanley, проведенного в январе 2025 года,
    инвестиции в акции технологического сектора остаются наиболее привлекательными.
    Профессор экономики Джон Смит из Гарвардского университета рекомендует
    диверсифицировать портфель и включить в него ETF на индекс S&P 500.
    
    ## Стратегии инвестирования
    
    Согласно статистике, долгосрочные инвестиции приносят в среднем 7% годовых.
    Я лично использовал эту стратегию в течение 5 лет и могу подтвердить ее эффективность.
    
    ### Риски и управление капиталом
    
    * Не инвестируйте более 5% капитала в одну акцию
    * Создайте резервный фонд перед началом инвестирования
    * Учитывайте инфляцию при планировании доходности
    
    Источники:
    1. Financial Times, отчет за Q1 2025
    2. Исследование Morgan Stanley "Investment Outlook 2025"
    3. SEC.gov - руководство для начинающих инвесторов
    
    ВАЖНО: Данный материал не является финансовой консультацией. Проконсультируйтесь с
    профессиональным финансовым советником перед принятием инвестиционных решений.
    
    Последнее обновление: 10 апреля 2025 года
    Автор: Сертифицированный финансовый аналитик (CFA) с 15-летним опытом работы
    """
    
    # Анализ для финансовой отрасли (YMYL)
    finance_results = analyzer.analyze(test_text, industry='finance')
    
    # Выводим результаты
    print("\nРезультаты анализа для финансовой отрасли (YMYL):\n")
    print(f"Experience Score: {finance_results['experience_score']:.2f}")
    print(f"Expertise Score: {finance_results['expertise_score']:.2f}")
    print(f"Authority Score: {finance_results['authority_score']:.2f}")
    print(f"Trust Score: {finance_results['trust_score']:.2f}")
    print(f"Structural Score: {finance_results['structural_score']:.2f}")
    print(f"YMYL Status: {'Да' if finance_results['ymyl_status'] == 1 else 'Нет'}")
    print(f"Overall E-E-A-T Score: {finance_results['overall_eeat_score']:.2f}")
    
    print("\nРекомендации:")
    for i, recommendation in enumerate(finance_results['recommendations'], 1):
        print(f"{i}. {recommendation}")
    
    # Анализ для обычной блог-отрасли (не YMYL)
    blog_results = analyzer.analyze(test_text, industry='blog')
    
    print("\n\nСравнение общих оценок E-E-A-T:")
    print(f"Финансы (YMYL): {finance_results['overall_eeat_score']:.2f}")
    print(f"Блог (не YMYL): {blog_results['overall_eeat_score']:.2f}")


def demo_enhanced_eeat_analyzer():
    """Демонстрация EnhancedEEATAnalyzer с ML-моделью."""
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ ENHANCED EEAT ANALYZER С ML")
    print("=" * 80)
    
    # Проверяем наличие модели
    model_path = Path("seo_ai_models/data/models/eeat/eeat_best_model.joblib")
    
    # Если модель не существует, создаем простую тестовую модель
    if not model_path.exists():
        print("\nМодель не найдена. Демонстрация будет выполнена без ML-модели.")
        print("В реальном проекте модель будет использовать eeat_best_model.joblib")
        analyzer = EnhancedEEATAnalyzer()
    else:
        print(f"\nИспользуется модель: {model_path}")
        analyzer = EnhancedEEATAnalyzer(model_path=str(model_path))
    
    # Тестовый текст (тот же, что и для обычного EEATAnalyzer)
    test_text = """
    # Как правильно инвестировать в акции в 2025 году
    
    По данным исследования Morgan Stanley, проведенного в январе 2025 года,
    инвестиции в акции технологического сектора остаются наиболее привлекательными.
    Профессор экономики Джон Смит из Гарвардского университета рекомендует
    диверсифицировать портфель и включить в него ETF на индекс S&P 500.
    
    ## Стратегии инвестирования
    
    Согласно статистике, долгосрочные инвестиции приносят в среднем 7% годовых.
    Я лично использовал эту стратегию в течение 5 лет и могу подтвердить ее эффективность.
    
    ### Риски и управление капиталом
    
    * Не инвестируйте более 5% капитала в одну акцию
    * Создайте резервный фонд перед началом инвестирования
    * Учитывайте инфляцию при планировании доходности
    
    Источники:
    1. Financial Times, отчет за Q1 2025
    2. Исследование Morgan Stanley "Investment Outlook 2025"
    3. SEC.gov - руководство для начинающих инвесторов
    
    ВАЖНО: Данный материал не является финансовой консультацией. Проконсультируйтесь с
    профессиональным финансовым советником перед принятием инвестиционных решений.
    
    Последнее обновление: 10 апреля 2025 года
    Автор: Сертифицированный финансовый аналитик (CFA) с 15-летним опытом работы
    """
    
    # Анализ с использованием EnhancedEEATAnalyzer
    results = analyzer.analyze(test_text, industry='finance')
    
    # Выводим результаты
    print("\nРезультаты анализа с использованием EnhancedEEATAnalyzer:\n")
    print(f"Experience Score: {results['experience_score']:.2f}")
    print(f"Expertise Score: {results['expertise_score']:.2f}")
    print(f"Authority Score: {results['authority_score']:.2f}")
    print(f"Trust Score: {results['trust_score']:.2f}")
    print(f"Structural Score: {results['structural_score']:.2f}")
    print(f"ML-модель использовалась: {'Да' if results.get('ml_model_used', False) else 'Нет'}")
    
    if 'original_overall_eeat_score' in results:
        print(f"\nБазовая оценка E-E-A-T: {results['original_overall_eeat_score']:.2f}")
        print(f"ML-улучшенная оценка E-E-A-T: {results['overall_eeat_score']:.2f}")
        print(f"Разница: {results['overall_eeat_score'] - results['original_overall_eeat_score']:.2f}")
    else:
        print(f"\nОбщая оценка E-E-A-T: {results['overall_eeat_score']:.2f}")
    
    print("\nРекомендации:")
    for i, recommendation in enumerate(results['recommendations'], 1):
        print(f"{i}. {recommendation}")


def demo_content_analyzer():
    """Демонстрация улучшенного ContentAnalyzer."""
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ УЛУЧШЕННОГО CONTENT ANALYZER")
    print("=" * 80)
    
    # Создаем экземпляр ContentAnalyzer
    analyzer = ContentAnalyzer()
    
    # Тестовый текст (Markdown)
    test_markdown = """
    # Руководство по SEO-оптимизации в 2025 году
    
    В этой статье рассмотрены актуальные методы SEO-оптимизации, 
    которые будут работать в 2025 году. Мы проанализировали последние данные
    и изменения в алгоритмах поисковых систем.
    
    ## Важность E-E-A-T в 2025 году
    
    Google продолжает усиливать значимость E-E-A-T (Experience, Expertise, 
    Authoritativeness, Trustworthiness) при ранжировании сайтов.
    
    ### Как улучшить E-E-A-T
    
    * Добавьте информацию об авторах
    * Цитируйте авторитетные источники
    * Регулярно обновляйте контент
    
    ## Технические факторы
    
    Технические аспекты SEO остаются важными:
    
    1. Скорость загрузки сайта
    2. Мобильная оптимизация
    3. Структурированные данные
    
    Подробнее читайте в нашем исследовании: https://example.com/seo-research-2025
    
    Последнее обновление: 15 марта 2025 года
    Автор: SEO эксперт с 10-летним опытом
    """
    
    # Тестовые ключевые слова
    test_keywords = ["SEO", "оптимизация", "E-E-A-T", "контент", "Google"]
    
    # Анализ текста
    metrics = analyzer.analyze_text(test_markdown)
    
    # Выводим метрики
    print("\nОсновные метрики контента:\n")
    print(f"Количество слов: {metrics['word_count']}")
    print(f"Количество предложений: {metrics['sentence_count']}")
    print(f"Средняя длина предложения: {metrics['avg_sentence_length']:.2f}")
    print(f"Оценка читабельности: {metrics['readability']:.2f}")
    print(f"Оценка заголовков: {metrics['header_score']:.2f}")
    print(f"Оценка структуры: {metrics['structure_score']:.2f}")
    print(f"Семантическая глубина: {metrics['semantic_depth']:.2f}")
    print(f"Тематическая релевантность: {metrics['topic_relevance']:.2f}")
    print(f"Потенциал вовлечения: {metrics['engagement_potential']:.2f}")
    
    # Анализ ключевых слов
    keyword_metrics = analyzer.extract_keywords(test_markdown, test_keywords)
    
    print("\nМетрики ключевых слов:\n")
    print(f"Общая плотность ключевых слов: {keyword_metrics['density']:.2f}")
    print(f"Prominence (заметность): {keyword_metrics['prominence']:.2f}")
    print(f"Coverage (покрытие): {keyword_metrics['coverage']:.2f}")
    
    print("\nЧастота ключевых слов:")
    for keyword, frequency in keyword_metrics['frequency'].items():
        print(f"{keyword}: {frequency}")


def demo_seo_advisor():
    """Демонстрация улучшенного SEOAdvisor."""
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ ИНТЕГРИРОВАННОГО SEO ADVISOR")
    print("=" * 80)
    
    # Создаем экземпляр SEOAdvisor
    advisor = SEOAdvisor(industry='finance')
    
    # Тестовый текст
    test_text = """
    # Руководство по инвестированию в криптовалюты в 2025 году
    
    По данным Bloomberg, рынок криптовалют в 2025 году демонстрирует стабильный рост.
    В этой статье я расскажу о перспективных направлениях инвестирования в цифровые активы.
    
    ## Основные тренды 2025 года
    
    * Институциональные инвесторы продолжают увеличивать свое присутствие
    * Регулирование становится более прозрачным
    * DeFi-проекты развиваются и привлекают новых пользователей
    
    ## Стратегии управления рисками
    
    Инвестиции в криптовалюты сопряжены с высокими рисками. Рекомендуется:
    
    1. Диверсифицировать портфель
    2. Инвестировать только свободные средства
    3. Изучать технологию блокчейн и фундаментальные аспекты проектов
    
    Я лично инвестирую в криптовалюты с 2017 года и применяю долгосрочную стратегию.
    
    Источники:
    - [Отчет Bloomberg](https://www.bloomberg.com/)
    - [Исследование MIT](https://www.mit.edu/)
    - [Данные CoinMarketCap](https://coinmarketcap.com/)
    
    ПРЕДУПРЕЖДЕНИЕ: Этот материал не является финансовой консультацией.
    Всегда проводите собственное исследование перед инвестированием.
    
    Последнее обновление: 1 апреля 2025 года
    Автор: Финансовый аналитик с сертификацией CFA
    """
    
    # Ключевые слова
    keywords = ["криптовалюты", "инвестирование", "блокчейн", "DeFi", "риски"]
    
    # Анализ контента
    print("\nВыполняется полный SEO-анализ...")
    report = advisor.analyze_content(test_text, keywords)
    
    # Выводим результаты
    print("\nSEO-отчет:\n")
    print(f"Прогнозируемая позиция: {report.predicted_position:.2f}")
    print(f"Общая оценка E-E-A-T: {report.content_metrics.get('overall_eeat_score', 0):.2f}")
    
    print("\nМетрики контента:")
    for metric, value in report.content_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
    
    print("\nАнализ ключевых слов:")
    print(f"Общая плотность: {report.keyword_analysis['density']:.2f}")
    
    print("\nОценки ключевых факторов:")
    for factor, score in report.feature_scores.items():
        print(f"{factor}: {score:.2f}")
    
    print("\nОценка качества контента:")
    print(f"Общее качество: {report.content_quality.content_scores.get('overall_quality', 0):.2f}")
    
    print("\nСильные стороны:")
    for strength in report.content_quality.strengths[:3]:  # Показываем первые 3
        print(f"- {strength}")
    
    print("\nСлабые стороны:")
    for weakness in report.content_quality.weaknesses[:3]:  # Показываем первые 3
        print(f"- {weakness}")
    
    print("\nТоп-3 рекомендации:")
    priority_count = 0
    for priority in report.priorities:
        if priority_count >= 3:
            break
        print(f"- {priority['task']}")
        priority_count += 1


if __name__ == "__main__":
    try:
        # Выполняем демонстрации
        demo_text_processor()
        demo_eeat_analyzer()
        demo_enhanced_eeat_analyzer()
        demo_content_analyzer()
        demo_seo_advisor()
        
        print("\n" + "=" * 80)
        print("Демонстрация успешно завершена!")
        print("=" * 80)
    except Exception as e:
        import traceback
        print(f"\nОшибка при выполнении демонстрации: {e}")
        traceback.print_exc()
