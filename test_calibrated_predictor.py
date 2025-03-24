
from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor
from models.seo_advisor.calibrated_rank_predictor import CalibratedRankPredictor

def test_predictors():
    """Сравнение предсказаний старого и нового предикторов"""
    # Тестовые случаи
    test_cases = [
        {
            'name': 'Плохой контент',
            'features': {
                'keyword_density': 0.005,
                'content_length': 300,
                'readability_score': 30,
                'meta_tags_score': 0.3,
                'header_structure_score': 0.2,
                'multimedia_score': 0.1,
                'internal_linking_score': 0.1,
                'topic_relevance': 0.4,
                'semantic_depth': 0.3,
                'engagement_potential': 0.2
            }
        },
        {
            'name': 'Хороший контент',
            'features': {
                'keyword_density': 0.02,
                'content_length': 1500,
                'readability_score': 70,
                'meta_tags_score': 0.7,
                'header_structure_score': 0.7,
                'multimedia_score': 0.6,
                'internal_linking_score': 0.6,
                'topic_relevance': 0.8,
                'semantic_depth': 0.7,
                'engagement_potential': 0.7
            }
        }
    ]
    
    # Отрасли для тестирования
    industries = ['default', 'electronics', 'health', 'finance']
    
    print("===== СРАВНЕНИЕ ПРЕДИКТОРОВ =====")
    print("{:<15} {:<15} {:<20} {:<20} {:<15}".format(
        "Тип контента", "Отрасль", "Старый предиктор", "Новый предиктор", "Разница"
    ))
    
    for case in test_cases:
        for industry in industries:
            old_predictor = ImprovedRankPredictor(industry=industry)
            new_predictor = CalibratedRankPredictor(industry=industry)
            
            old_prediction = old_predictor.predict_position(case['features'])
            new_prediction = new_predictor.predict_position(case['features'])
            
            print("{:<15} {:<15} {:<20.1f} {:<20.1f} {:<15.1f}".format(
                case['name'],
                industry,
                old_prediction['position'],
                new_prediction['position'],
                old_prediction['position'] - new_prediction['position']
            ))
            
            # Для хорошего контента показываем вероятности
            if case['name'] == 'Хороший контент':
                if 'probability' in new_prediction:
                    print(f"  Вероятности для {industry}:")
                    for k, v in new_prediction['probability'].items():
                        print(f"    {k}: {v*100:.1f}%")

# Запуск теста
if __name__ == "__main__":
    test_predictors()
