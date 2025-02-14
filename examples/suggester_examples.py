
from models.seo_advisor.suggester import Suggester

def test_suggester():
    # Тест 1: Блог с недостаточным контентом
    blog_suggester = Suggester(industry='blog')
    blog_features = {
        'keyword_density': 0.015,
        'content_length': 500,
        'readability_score': 65
    }
    
    analysis = blog_suggester.analyze_content(blog_features)
    print("=== Анализ блога ===")
    print(f"Текущая позиция: {analysis['current_position']}")
    print("\nАнализ факторов:")
    for feature, score in analysis['score_analysis'].items():
        print(f"{feature}: {score}")
    
    print("\nРекомендации:")
    for rec in analysis['base_recommendations'].get('content_length', []):
        print(f"- {rec}")
    
    print("\nИнсайты конкурентов:")
    for insight in analysis['competitor_insights']:
        print(f"- {insight}")

# Запуск тестов
test_suggester()
