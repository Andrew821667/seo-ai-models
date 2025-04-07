#!/usr/bin/env python3
"""
Демонстрационный скрипт для тестирования проверки консистентности метрик.
"""

import sys
import os
import json
from pprint import pprint

# Добавляем родительскую директорию в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seo_ai_models.common.utils.metrics_consistency import MetricsConsistencyChecker

def main():
    """Основная функция скрипта."""
    print("Демонстрация проверки консистентности метрик")
    
    # Создаем набор противоречивых метрик
    inconsistent_metrics = {
        'readability': 0.9,              # Высокая читабельность
        'avg_sentence_length': 40.2,     # Но очень длинные предложения (противоречие)
        'word_count': 250,               # Небольшой текст
        'semantic_depth': 0.8,           # Но высокая семантическая глубина (противоречие для короткого текста)
        'headers_count': 0,              # Нет заголовков
        'header_score': 0.7,             # Но высокий score заголовков (противоречие)
        'structure_score': 0.9,          # Высокая оценка структуры
        'lists_count': 0,                # Но нет списков (противоречие для высокого structure_score)
        'keyword_density': 0.15,         # Очень высокая плотность ключевых слов (нереалистично)
        'multimedia_score': 0.8,         # Высокая оценка мультимедиа
        'has_images': False              # Но нет изображений (противоречие)
    }
    
    # Инициализируем проверку консистентности
    checker = MetricsConsistencyChecker()
    
    # Проверяем и исправляем метрики
    print("\nИсходные метрики (с противоречиями):")
    pprint(inconsistent_metrics)
    
    # Применяем проверку консистентности
    fixed_metrics = checker.check_and_fix(inconsistent_metrics)
    
    print("\nИсправленные метрики (без противоречий):")
    pprint(fixed_metrics)
    
    # Выводим изменения
    print("\nВнесенные изменения:")
    for key in inconsistent_metrics:
        if inconsistent_metrics[key] != fixed_metrics[key]:
            print(f"- {key}: {inconsistent_metrics[key]} -> {fixed_metrics[key]}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
