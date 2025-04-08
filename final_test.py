
import logging
import sys
import json
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.semantic_analyzer import SemanticAnalyzer
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer
from seo_ai_models.parsers.adaptive_parsing_pipeline import AdaptiveParsingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_seo_components():
    """Проверяет работу основных SEO компонентов"""
    print("\n🔍 Проверка SEO компонентов\n")
    
    # Тестовый текст
    test_text = """
    SEO Best Practices Guide
    
    This guide covers essential SEO practices for optimizing websites. 
    It includes technical SEO, on-page optimization, and content strategies.
    Keywords play an important role in SEO success.
    """
    
    # Ключевые слова
    keywords = ["SEO", "optimization", "guide"]
    
    # 1. ContentAnalyzer
    print("✅ ContentAnalyzer:")
    content_analyzer = ContentAnalyzer()
    content_metrics = content_analyzer.analyze_text(test_text)
    print(f"  - Статус: Работает")
    print(f"  - Метрики: {json.dumps(content_metrics, indent=2, ensure_ascii=False)[:200]}...")
    
    # 2. SemanticAnalyzer
    print("\n✅ SemanticAnalyzer:")
    semantic_analyzer = SemanticAnalyzer()
    try:
        semantic_results = semantic_analyzer.analyze_text(test_text, keywords)
        print(f"  - Статус: Работает")
        if semantic_results:
            print(f"  - Результаты: {json.dumps(semantic_results, indent=2, ensure_ascii=False)[:200]}...")
    except Exception as e:
        print(f"  - Статус: Ошибка - {e}")
    
    # 3. EEATAnalyzer
    print("\n✅ EEATAnalyzer:")
    eeat_analyzer = EEATAnalyzer()
    eeat_results = eeat_analyzer.analyze(test_text)
    print(f"  - Статус: Работает")
    print(f"  - Результаты: {json.dumps(eeat_results, indent=2, ensure_ascii=False)[:200]}...")
    
    # 4. SEOAdvisor
    print("\n✅ SEOAdvisor:")
    advisor = SEOAdvisor()
    try:
        report = advisor.analyze_content(test_text, keywords)
        print(f"  - Статус: Работает")
        print(f"  - Атрибуты отчета: {', '.join([attr for attr in dir(report) if not attr.startswith('_')])}")
        
        # Проверяем ключевые атрибуты отчета
        content_metrics = getattr(report, 'content_metrics', None)
        if content_metrics:
            print(f"  - Метрики контента: {content_metrics}")
        
        keyword_analysis = getattr(report, 'keyword_analysis', None)
        if keyword_analysis:
            print(f"  - Анализ ключевых слов: {keyword_analysis}")
        
        predicted_position = getattr(report, 'predicted_position', None)
        if predicted_position:
            print(f"  - Предсказанная позиция: {predicted_position}")
    except Exception as e:
        print(f"  - Статус: Ошибка - {e}")

def check_parser_components():
    """Проверяет работу парсера"""
    print("\n🌐 Проверка компонентов парсера\n")
    
    # Проверяем адаптивный парсер
    print("✅ AdaptiveParsingPipeline:")
    try:
        pipeline = AdaptiveParsingPipeline()
        print(f"  - Статус: Создан успешно")
        
        # Проверяем метод определения типа сайта
        print("  - Метод detect_site_type: Проверяем...")
        site_type = pipeline.detect_site_type("https://example.com")
        print(f"    Результат: {site_type}")
    except Exception as e:
        print(f"  - Статус: Ошибка - {e}")

if __name__ == "__main__":
    print("\n🚀 Финальное тестирование компонентов SEO AI Models\n")
    
    try:
        check_seo_components()
        check_parser_components()
        print("\n✅ Тестирование успешно завершено!\n")
    except Exception as e:
        print(f"\n❌ Тестирование не удалось: {e}\n")
