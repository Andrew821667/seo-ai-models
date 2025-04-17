
# Простой пример использования унифицированного парсера

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.parsers.unified.site_analyzer import SiteAnalyzer
import json

def main():
    print("Демонстрация унифицированного парсера SEO AI Models")
    print("-" * 60)
    
    # Парсинг URL
    parser = UnifiedParser()
    result = parser.parse_url("https://example.com")
    
    print("
Результаты парсинга URL:")
    print(f"Заголовок: {result.get('page_data', {}).get('structure', {}).get('title', '')}")
    print(f"Количество слов: {result.get('page_data', {}).get('content', {}).get('word_count', 0)}")
    
    # Анализ URL с SEO Advisor
    analyzer = SiteAnalyzer()
    analysis = analyzer.analyze_url("https://example.com")
    
    print("
Результаты SEO-анализа:")
    print(f"Предсказанная позиция: {analysis.get('seo_analysis', {}).get('predicted_position', 0):.1f}")
    
    strengths = analysis.get('seo_analysis', {}).get('content_quality', {}).get('strengths', [])
    if strengths:
        print("
Сильные стороны:")
        for strength in strengths[:3]:
            print(f"- {strength}")
    
    # Сохранение результатов
    with open('unified_parser_demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("
Результаты сохранены в файл: unified_parser_demo_results.json")

if __name__ == "__main__":
    main()
