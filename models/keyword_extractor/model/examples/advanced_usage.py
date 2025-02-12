# model/examples/advanced_usage.py

import torch
from pathlib import Path
import json
from datetime import datetime

from model.model import KeywordExtractorModel
from model.config.model_config import KeywordModelConfig
from model.utils.visualization import KeywordVisualizer
from model.utils.analysis import ErrorAnalyzer
from model.monitoring.performance import PerformanceMonitor

def analyze_with_visualization():
    """Пример анализа с визуализацией результатов"""
    # Инициализация
    config = KeywordModelConfig()
    model = KeywordExtractorModel(config)
    visualizer = KeywordVisualizer(save_dir="visualizations")
    monitor = PerformanceMonitor()
    
    # Тестовые тексты
    texts = [
        """
        Data science is the study of data to extract meaningful insights. 
        It combines statistics, programming, and domain expertise to analyze 
        complex datasets and solve business problems.
        """,
        """
        Deep learning has revolutionized artificial intelligence. Neural 
        networks can now achieve human-level performance in many tasks 
        including image recognition and natural language processing.
        """
    ]
    
    # Замер производительности
    start_time = monitor.start_batch()
    
    # Извлечение ключевых слов
    keywords = model.extract_keywords(texts)
    
    # Сбор метрик производительности
    performance = monitor.end_batch(start_time, len(texts))
    
    # Визуализация результатов
    visualizer.plot_keyword_distribution(
        keywords,
        filename="keyword_distribution.png"
    )
    
    # Создание отчета
    report = {
        'timestamp': datetime.now().isoformat(),
        'performance': performance,
        'results': keywords
    }
    
    # Сохранение отчета
    with open("analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    return keywords, performance

def error_analysis_example():
    """Пример анализа ошибок и качества извлечения"""
    config = KeywordModelConfig()
    model = KeywordExtractorModel(config)
    analyzer = ErrorAnalyzer()
    
    # Тексты и ожидаемые ключевые слова
    test_data = [
        {
            'text': "Python programming basics and advanced techniques",
            'expected': ['python', 'programming', 'techniques']
        },
        {
            'text': "Machine learning algorithms and applications",
            'expected': ['machine learning', 'algorithms', 'applications']
        }
    ]
    
    # Получение предсказаний
    predictions = []
    for item in test_data:
        keywords = model.extract_keywords([item['text']])
        predictions.append(keywords[0])
    
    # Анализ ошибок
    error_analysis = analyzer.analyze_predictions(
        predictions=predictions,
        targets=[item['expected'] for item in test_data],
        texts=[item['text'] for item in test_data]
    )
    
    # Вывод результатов анализа
    print("\nError Analysis Results:")
    print("1. Error Types:")
    for error_type, stats in error_analysis['error_types'].items():
        print(f"- {error_type}: {stats['percentage']:.1f}%")
        
    print("\n2. Length Analysis:")
    for length, metrics in error_analysis['length_analysis'].items():
        print(f"- Length {length}: avg precision = {metrics['avg_precision']:.2f}")
        
    return error_analysis

def custom_threshold_analysis():
    """Пример анализа влияния порога уверенности"""
    config = KeywordModelConfig()
    model = KeywordExtractorModel(config)
    
    text = """
    Natural Language Processing (NLP) is a branch of artificial intelligence 
    that helps computers understand, interpret and manipulate human language. 
    This technology enables computers to process human language in the form 
    of text or voice data and 'understand' its full meaning.
    """
    
    # Проверка разных порогов
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for threshold in thresholds:
        keywords = model.extract_keywords(
            [text],
            threshold=threshold
        )
        results[threshold] = keywords[0]
    
    # Анализ результатов
    print("\nThreshold Analysis:")
    for threshold, keywords in results.items():
        print(f"\nThreshold: {threshold}")
        print(f"Found {len(keywords)} keywords:")
        for kw in keywords:
            print(f"- {kw['keyword']} (score: {kw['score']:.2f})")
            
    return results

def main():
    print("1. Analysis with Visualization:")
    keywords, performance = analyze_with_visualization()
    print(f"Processed texts in {performance['batch_processing_time']:.2f} seconds")
    
    print("\n2. Error Analysis:")
    error_analysis = error_analysis_example()
    
    print("\n3. Threshold Analysis:")
    threshold_results = custom_threshold_analysis()

if __name__ == "__main__":
    main()
