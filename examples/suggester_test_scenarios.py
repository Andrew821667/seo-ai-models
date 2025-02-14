
from models.seo_advisor.suggester import Suggester
import json

def print_analysis_results(analysis):
    """Красивый вывод результатов анализа"""
    print("\n" + "="*50)
    print(f"Текущая позиция: {analysis['current_position']:.2f}")
    
    print("\n--- Анализ факторов ---")
    for feature, data in analysis['score_analysis'].items():
        print(f"{feature}:")
        print(f"  Текущее значение: {data['current_value']}")
        print(f"  Скор: {data['score']:.2f}")
        print(f"  Влияние: {data['impact_percentage']:.1f}%")
        print(f"  Статус: {data['status']}")
    
    print("\n--- Приоритетные задачи ---")
    for task in analysis['priority_tasks']:
        print(f"- {task['title']} (Приоритет: {task['priority']})")
        print(f"  {task['description']}")
        print(f"  Влияние на результат: {task['impact']:.1f}%")
    
    print("\n--- Рекомендации конкурентов ---")
    for insight in analysis['competitor_insights']:
        print(f"- {insight}")

def test_scenarios():
    scenarios = [
        {
            'name': 'Блог с недостаточным контентом',
            'industry': 'blog',
            'features': {
                'keyword_density': 0.015,
                'content_length': 500,
                'readability_score': 65,
                'meta_tags_score': 0.6,
                'header_structure_score': 0.7,
                'backlinks_count': 25,
                'multimedia_score': 0.4,
                'internal_linking_score': 0.5
            }
        },
        {
            'name': 'Научный блог с низкой читабельностью',
            'industry': 'scientific_blog',
            'features': {
                'keyword_density': 0.012,
                'content_length': 1200,
                'readability_score': 40,
                'meta_tags_score': 0.5,
                'header_structure_score': 0.6,
                'backlinks_count': 35,
                'multimedia_score': 0.3,
                'internal_linking_score': 0.4
            }
        },
        {
            'name': 'E-commerce с избыточным контентом',
            'industry': 'ecommerce',
            'features': {
                'keyword_density': 0.04,
                'content_length': 3000,
                'readability_score': 55,
                'meta_tags_score': 0.8,
                'header_structure_score': 0.9,
                'backlinks_count': 150,
                'multimedia_score': 0.7,
                'internal_linking_score': 0.6
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n=== {scenario['name']} ===")
        suggester = Suggester(industry=scenario['industry'])
        analysis = suggester.analyze_content(scenario['features'])
        
        print_analysis_results(analysis)

# Запускаем тесты
test_scenarios()
