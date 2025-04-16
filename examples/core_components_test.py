#!/usr/bin/env python3
"""
Комплексный тест ядра SEO AI Models.
Тестирует улучшенные компоненты системы без использования парсера.
"""

import sys
import os
import json
from pprint import pprint
from datetime import datetime

# Добавляем родительскую директорию в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from seo_ai_models.models.seo_advisor.enhanced_advisor import EnhancedSEOAdvisor
from seo_ai_models.common.utils.metrics_consistency import MetricsConsistencyChecker

def test_enhanced_seo_advisor():
    """Тестирование улучшенного SEO Advisor."""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕННОГО SEO ADVISOR")
    print("="*80)
    
    # Создаем экземпляр улучшенного SEO Advisor
    enhanced_advisor = EnhancedSEOAdvisor(industry="digital_marketing")
    
    # Тестовый контент на русском языке
    test_content = """
    # Руководство по SEO оптимизации
    
    Поисковая оптимизация необходима для современных веб-сайтов.
    Рассмотрим основные аспекты:
    
    ## Оптимизация на странице
    
    Контент должен быть релевантным и полезным для пользователей.
    Ключевые слова следует использовать естественным образом.
    
    ## Техническое SEO
    
    Обеспечьте быструю загрузку вашего сайта.
    Адаптивность для мобильных устройств критически важна для рейтинга.
    """
    
    # Тестируем анализ контента
    print("\nАнализ контента с помощью EnhancedSEOAdvisor:")
    result = enhanced_advisor.analyze_content(
        content=test_content,
        target_keywords=["seo", "оптимизация", "контент"]
    )
    
    # Форматированный вывод основных результатов
    print(f"\n1. ОСНОВНЫЕ МЕТРИКИ КОНТЕНТА:")
    print(f"   - Количество слов: {result.content_metrics.get('word_count')}")
    print(f"   - Количество предложений: {result.content_metrics.get('sentence_count')}")
    print(f"   - Средняя длина предложения: {result.content_metrics.get('avg_sentence_length')}")
    print(f"   - Читаемость текста: {result.content_metrics.get('readability')}")
    
    print(f"\n2. СЕМАНТИЧЕСКИЕ МЕТРИКИ:")
    print(f"   - Семантическая плотность: {result.content_metrics.get('semantic_density')}")
    print(f"   - Семантическое покрытие: {result.content_metrics.get('semantic_coverage')}")
    print(f"   - Тематическая связность: {result.content_metrics.get('topical_coherence')}")
    print(f"   - Контекстуальная релевантность: {result.content_metrics.get('contextual_relevance')}")
    
    print(f"\n3. E-E-A-T МЕТРИКИ:")
    print(f"   - Опыт (Experience): {result.content_metrics.get('expertise_score')}")
    print(f"   - Экспертиза (Expertise): {result.content_metrics.get('expertise_score')}")
    print(f"   - Авторитетность (Authority): {result.content_metrics.get('authority_score')}")
    print(f"   - Надежность (Trustworthiness): {result.content_metrics.get('trust_score')}")
    print(f"   - Общая оценка E-E-A-T: {result.content_metrics.get('overall_eeat_score')}")
    
    print(f"\n4. АНАЛИЗ КЛЮЧЕВЫХ СЛОВ:")
    print(f"   - Плотность ключевых слов: {result.keyword_analysis.get('density')}")
    print(f"   - Распределение ключевых слов:")
    for kw, value in result.keyword_analysis.get('distribution', {}).items():
        print(f"     * {kw}: {value}")
    
    print(f"\n5. ПРЕДСКАЗАНИЕ ПОЗИЦИИ: {result.predicted_position}")
    print(f"   Вероятности попадания в:")
    for pos, prob in result.position_probabilities.items():
        print(f"     * {pos}: {prob*100:.1f}%")
    
    print(f"\n6. ТОП-5 ПРИОРИТЕТНЫХ РЕКОМЕНДАЦИЙ:")
    for i, task in enumerate(result.priorities[:5], 1):
        print(f"   {i}. {task['task']}")
        print(f"      - Влияние: {task['impact']:.2f}, Усилия: {task['effort']:.2f}, Приоритет: {task['priority_score']:.2f}")
    
    return result

def test_metrics_consistency_checker():
    """Тестирование проверки согласованности метрик."""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ METRICS CONSISTENCY CHECKER")
    print("="*80)
    
    # Создаем экземпляр проверки согласованности
    consistency_checker = MetricsConsistencyChecker()
    
    # Создаем набор противоречивых метрик для тестирования
    test_metrics = {
        'keyword_density': 0.9,  # Очень высокая плотность ключевых слов
        'readability': 0.3,      # Низкая читабельность
        'keyword_stuffing': 0.8, # Высокий показатель переспама ключевыми словами
        'user_engagement': 0.7,  # Но высокий показатель вовлеченности (что противоречит)
        'bounce_rate': 0.2       # И низкий показатель отказов (что также противоречит)
    }
    
    print("\nИсходные метрики:")
    for key, value in test_metrics.items():
        print(f"   - {key}: {value}")
    
    # Тестируем проверку согласованности
    consistent_metrics = consistency_checker.check_and_fix(test_metrics)
    
    print("\nСкорректированные метрики:")
    for key, value in consistent_metrics.items():
        if key in test_metrics and test_metrics[key] != value:
            print(f"   - {key}: {value} (было: {test_metrics[key]}) 🔄")
        elif key not in test_metrics:
            print(f"   - {key}: {value} (добавлено) ➕")
        else:
            print(f"   - {key}: {value}")
    
    return consistent_metrics

def test_rank_predictor(content, keywords):
    """Тестирование предиктора ранжирования."""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ RANK PREDICTOR")
    print("="*80)
    
    from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor
    from seo_ai_models.models.seo_advisor.predictors.improved_rank_predictor import ImprovedRankPredictor
    
    # Создаем экземпляры предикторов
    basic_predictor = CalibratedRankPredictor()
    improved_predictor = ImprovedRankPredictor()
    
    # Создаем базовый набор метрик для предсказания
    metrics = {
        'keyword_density': 0.03,
        'content_length': 0.7,
        'readability_score': 0.8,
        'authority_score': 0.5,
        'expertise_score': 0.6,
        'trust_score': 0.7,
        'semantic_depth': 0.65,
        'keyword_prominence': 0.75,
        'meta_tags_score': 0.9,
        'header_structure_score': 0.8,
        'multimedia_score': 0.5,
        'internal_linking_score': 0.6
    }
    
    # Тестируем базовый предиктор (без параметра industry)
    print("\nРезультаты базового предиктора:")
    try:
        position_basic = basic_predictor.predict_position(metrics)
        print(f"   - Предсказанная позиция: {position_basic}")
    except Exception as e:
        print(f"   - Ошибка при использовании базового предиктора: {e}")
    
    # Тестируем улучшенный предиктор (без параметра industry)
    print("\nРезультаты улучшенного предиктора:")
    try:
        position_improved = improved_predictor.predict_position(metrics)
        print(f"   - Предсказанная позиция: {position_improved}")
        
        # Получаем вероятности для разных позиций (если метод существует)
        if hasattr(improved_predictor, 'get_position_probabilities'):
            probabilities = improved_predictor.get_position_probabilities(metrics)
            print("   - Вероятности попадания в:")
            for pos, prob in probabilities.items():
                print(f"     * {pos}: {prob*100:.1f}%")
    except Exception as e:
        print(f"   - Ошибка при использовании улучшенного предиктора: {e}")
    
    return {"base_position": position_basic if 'position_basic' in locals() else None, 
            "improved_position": position_improved if 'position_improved' in locals() else None}

def run_all_tests():
    """Запуск всех тестов."""
    print("\n" + "*"*80)
    print("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ЯДРА SEO AI MODELS")
    print("*"*80)
    print(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Тестовый контент для всех тестов на русском языке
    test_content = """
    # Руководство по SEO оптимизации
    
    Поисковая оптимизация необходима для современных веб-сайтов.
    Рассмотрим основные аспекты:
    
    ## Оптимизация на странице
    
    Контент должен быть релевантным и полезным для пользователей.
    Ключевые слова следует использовать естественным образом.
    
    ## Техническое SEO
    
    Обеспечьте быструю загрузку вашего сайта.
    Адаптивность для мобильных устройств критически важна для рейтинга.
    """
    
    # Тестируем проверку согласованности метрик
    test_metrics_consistency_checker()
    
    # Тестируем предиктор ранжирования
    test_rank_predictor(test_content, ["seo", "оптимизация", "контент"])
    
    # Тестируем улучшенный SEO Advisor (комплексный тест)
    test_enhanced_seo_advisor()
    
    print("\n" + "*"*80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("*"*80)

if __name__ == "__main__":
    run_all_tests()
