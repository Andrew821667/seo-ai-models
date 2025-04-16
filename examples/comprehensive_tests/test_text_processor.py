"""
Комплексный тест текстового процессора и проверки согласованности метрик.
"""
import sys
import os
from datetime import datetime

# Добавляем путь к проекту - работает и в Jupyter/Colab, и при прямом запуске
current_dir = os.path.dirname(os.path.abspath(__file__ if '__file__' in globals() else '.'))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from seo_ai_models.common.utils.text_processing import TextProcessor
from seo_ai_models.common.utils.metrics_consistency import MetricsConsistencyChecker

def test_text_processor_and_consistency(test_text):
    """Тестирование TextProcessor и MetricsConsistencyChecker."""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ TEXT PROCESSOR И METRICS CONSISTENCY CHECKER")
    print("="*80)
    
    # Тестируем TextProcessor
    print("\n1. ТЕСТИРОВАНИЕ TEXT PROCESSOR")
    processor = TextProcessor()
    
    # Анализируем структуру текста
    structure = processor.analyze_text_structure(test_text)
    print("\nСтруктура текста:")
    for key, value in structure.items():
        print(f"   - {key}: {value}")
    
    # Расчет читабельности
    readability = processor.calculate_readability(test_text)
    print("\nЧитабельность текста:")
    for key, value in readability.items():
        print(f"   - {key}: {value}")
    
    # Извлечение заголовков
    headers = processor.extract_headers(test_text)
    print("\nЗаголовки в тексте:")
    for h in headers[:5]:  # Показываем только первые 5 заголовков
        print(f"   - {h}")
    if len(headers) > 5:
        print(f"   ... и еще {len(headers) - 5} заголовков")
    
    # Тестируем MetricsConsistencyChecker
    print("\n2. ТЕСТИРОВАНИЕ METRICS CONSISTENCY CHECKER")
    checker = MetricsConsistencyChecker()
    
    # Создаем тестовые метрики с некоторыми противоречиями
    test_metrics = {
        'keyword_density': 0.9,  # Очень высокая плотность ключевых слов
        'readability': 0.3,      # Низкая читабельность
        'keyword_stuffing': 0.8, # Высокий показатель переспама
        'user_engagement': 0.7,  # Но высокий показатель вовлеченности
        'bounce_rate': 0.2       # И низкий показатель отказов
    }
    
    print("\nИсходные метрики:")
    for key, value in test_metrics.items():
        print(f"   - {key}: {value}")
    
    # Проверяем и исправляем противоречия
    consistent_metrics = checker.check_and_fix(test_metrics)
    
    print("\nСкорректированные метрики:")
    for key, value in consistent_metrics.items():
        if key in test_metrics and test_metrics[key] != value:
            print(f"   - {key}: {value} (было: {test_metrics[key]}) 🔄")
        elif key not in test_metrics:
            print(f"   - {key}: {value} (добавлено) ➕")
        else:
            print(f"   - {key}: {value}")
    
    return {
        "text_structure": structure,
        "readability": readability,
        "headers": headers,
        "consistent_metrics": consistent_metrics
    }

if __name__ == "__main__":
    # Тестовый текст (при запуске скрипта напрямую)
    test_text = """
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
    
    # Запускаем тест с тестовым текстом
    test_text_processor_and_consistency(test_text)
