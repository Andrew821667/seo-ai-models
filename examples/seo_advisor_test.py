
import sys
sys.path.append('/content/seo-ai-models')

from common.config.advisor_config import AdvisorConfig, ModelConfig
from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor
from models.seo_advisor.suggester import Suggester
from models.seo_advisor.content_analyzer import ContentAnalyzer
from models.seo_advisor.rank_predictor import RankPredictor

import torch
import logging

def test_seo_components():
    print("=== Тестирование компонентов SEO Advisor ===")
    
    # 1. Конфигурация
    advisor_config = AdvisorConfig(
        model_path='/tmp/model',
        cache_dir='/tmp/cache',
        industry='blog'
    )
    
    model_config = ModelConfig()
    
    # 2. Инициализация компонентов
    try:
        advisor = SEOAdvisor(advisor_config)
        rank_predictor = RankPredictor(model_config)
        improved_rank_predictor = ImprovedRankPredictor(industry='blog')
        suggester = Suggester(industry='blog')
        content_analyzer = ContentAnalyzer(model_config)
        
        print("✅ Все компоненты успешно инициализированы")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # 3. Тестирование SEO Advisor
    print("\n--- Тестирование SEO Advisor ---")
    content = {
        "title": "Как улучшить SEO блога",
        "body": "В этой статье мы рассмотрим ключевые стратегии для повышения SEO-эффективности блога."
    }
    recommendations = advisor.get_recommendations(content)
    print("Рекомендации:", recommendations)
    
    # 4. Тестирование ImprovedRankPredictor
    print("\n--- Тестирование ImprovedRankPredictor ---")
    blog_features = {
        'keyword_density': 0.015,
        'content_length': 1500,
        'readability_score': 65,
        'meta_tags_score': 0.6,
        'header_structure_score': 0.7,
        'backlinks_count': 25,
        'multimedia_score': 0.4,
        'internal_linking_score': 0.5
    }
    
    rank_prediction = improved_rank_predictor.predict_position(blog_features)
    print("Позиция в рейтинге:", rank_prediction['position'])
    print("Рекомендации:", improved_rank_predictor.generate_recommendations(blog_features))
    
    # 5. Тестирование Suggester
    print("\n--- Тестирование Suggester ---")
    suggester_analysis = suggester.analyze_content(blog_features)
    print("Текущая позиция:", suggester_analysis['current_position'])
    
    # 6. Тестирование ContentAnalyzer
    print("\n--- Тестирование ContentAnalyzer ---")
    sample_text = "SEO - это комплекс мероприятий по повышению видимости сайта в поисковых системах."
    
    # Отключаем логирование для чистоты вывода
    logging.getLogger('transformers').setLevel(logging.ERROR)
    
    try:
        analysis_result = content_analyzer(sample_text)
        print("Результаты анализа контента:", analysis_result.keys())
        
        # Извлечение ключевых слов
        keywords = content_analyzer.extract_keywords(sample_text)
        print("Ключевые слова:", keywords)
    except Exception as e:
        print(f"❌ Ошибка в ContentAnalyzer: {e}")
    
    # 7. Тестирование RankPredictor
    print("\n--- Тестирование RankPredictor ---")
    try:
        # Создаем случайные признаки
        features = torch.rand(10, model_config.content_dim)
        rank_result = rank_predictor(features)
        print("Результат RankPredictor:", rank_result.keys())
    except Exception as e:
        print(f"❌ Ошибка в RankPredictor: {e}")

if __name__ == "__main__":
    test_seo_components()
