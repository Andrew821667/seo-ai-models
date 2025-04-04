import sys
import os
from pprint import pprint

# Добавляем корневую директорию в путь поиска модулей
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seo_ai_models.models.seo_advisor.predictors.calibrated_rank_predictor import CalibratedRankPredictor

def test_calibrated_rank_predictor():
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕННОГО CALIBRATEDRANKPREDICTOR")
    print("=" * 80)
    
    # Создаем предикторы для разных отраслей
    predictor_finance = CalibratedRankPredictor(industry="finance")
    predictor_blog = CalibratedRankPredictor(industry="blog")
    predictor_health = CalibratedRankPredictor(industry="health")
    
    # Тестовые наборы признаков
    features_excellent = {
        'keyword_density': 0.025,        # Оптимальная плотность
        'content_length': 2500,          # Отличная длина
        'readability_score': 85,         # Высокая читабельность
        'meta_tags_score': 0.95,         # Отличные мета-теги
        'header_structure_score': 0.95,  # Отличная структура
        'multimedia_score': 0.90,        # Хорошее использование мультимедиа
        'internal_linking_score': 0.85,  # Хорошее внутреннее связывание
        'topic_relevance': 0.95,         # Высокая релевантность
        'semantic_depth': 0.90,          # Высокая семантическая глубина
        'engagement_potential': 0.85,    # Высокий потенциал вовлечения
        'expertise_score': 0.95,         # Высокая экспертиза
        'authority_score': 0.90,         # Высокая авторитетность
        'trust_score': 0.95,             # Высокое доверие
        'overall_eeat_score': 0.95       # Высокая E-E-A-T оценка
    }
    
    features_good = {
        'keyword_density': 0.02,        # Хорошая плотность
        'content_length': 1500,         # Хорошая длина
        'readability_score': 75,        # Хорошая читабельность
        'meta_tags_score': 0.80,        # Хорошие мета-теги
        'header_structure_score': 0.70, # Хорошая структура
        'multimedia_score': 0.60,       # Среднее использование мультимедиа
        'internal_linking_score': 0.65, # Среднее внутреннее связывание
        'topic_relevance': 0.75,        # Хорошая релевантность
        'semantic_depth': 0.70,         # Хорошая семантическая глубина
        'engagement_potential': 0.65,   # Средний потенциал вовлечения
        'expertise_score': 0.70,        # Хорошая экспертиза
        'authority_score': 0.65,        # Средняя авторитетность
        'trust_score': 0.75,            # Хорошее доверие
        'overall_eeat_score': 0.70      # Хорошая E-E-A-T оценка
    }
    
    features_poor = {
        'keyword_density': 0.01,        # Низкая плотность
        'content_length': 500,          # Короткий контент
        'readability_score': 50,        # Средняя читабельность
        'meta_tags_score': 0.30,        # Плохие мета-теги
        'header_structure_score': 0.25, # Плохая структура
        'multimedia_score': 0.20,       # Слабое использование мультимедиа
        'internal_linking_score': 0.15, # Слабое внутреннее связывание
        'topic_relevance': 0.35,        # Низкая релевантность
        'semantic_depth': 0.30,         # Низкая семантическая глубина
        'engagement_potential': 0.25,   # Низкий потенциал вовлечения
        'expertise_score': 0.30,        # Низкая экспертиза
        'authority_score': 0.25,        # Низкая авторитетность
        'trust_score': 0.20,            # Низкое доверие
        'overall_eeat_score': 0.25      # Низкая E-E-A-T оценка
    }
    
    # ТЕСТ 1: Влияние отрасли
    print("\n>> ТЕСТ 1: Влияние отрасли при одинаковых показателях")
    print("\nОтличный контент:")
    for predictor, industry in [(predictor_finance, "Finance (YMYL)"), 
                               (predictor_health, "Health (YMYL)"), 
                               (predictor_blog, "Blog (не YMYL)")]:
        result = predictor.predict_position(features_excellent)
        print(f"{industry} - Позиция: {result['position']}, Top 3: {result['probability']['top3']:.2f}, Competition Factor: {result['competition_factor']}")
    
    # ТЕСТ 2: Влияние качества контента в одной отрасли
    print("\n>> ТЕСТ 2: Влияние качества контента (финансовая отрасль)")
    for features, quality in [(features_excellent, "Отличный"), 
                             (features_good, "Хороший"), 
                             (features_poor, "Слабый")]:
        result = predictor_finance.predict_position(features)
        print(f"{quality} контент - Позиция: {result['position']}, Top 10: {result['probability']['top10']:.2f}, Score: {result['total_score']:.2f}")
    
    # ТЕСТ 3: Влияние E-E-A-T в YMYL и не-YMYL отраслях
    print("\n>> ТЕСТ 3: Влияние E-E-A-T факторов")
    
    # Создаем копию тестовых данных с высокими и низкими E-E-A-T
    features_high_eeat = features_good.copy()
    features_high_eeat.update({
        'expertise_score': 0.95,
        'authority_score': 0.90,
        'trust_score': 0.95,
        'overall_eeat_score': 0.95
    })
    
    features_low_eeat = features_good.copy()
    features_low_eeat.update({
        'expertise_score': 0.30,
        'authority_score': 0.25,
        'trust_score': 0.20,
        'overall_eeat_score': 0.25
    })
    
    # Проверяем разницу для YMYL отрасли
    high_eeat_finance = predictor_finance.predict_position(features_high_eeat)
    low_eeat_finance = predictor_finance.predict_position(features_low_eeat)
    
    print(f"Финансы (YMYL) - Высокий E-E-A-T - Позиция: {high_eeat_finance['position']}")
    print(f"Финансы (YMYL) - Низкий E-E-A-T - Позиция: {low_eeat_finance['position']}")
    
    # Проверяем разницу для не-YMYL отрасли
    high_eeat_blog = predictor_blog.predict_position(features_high_eeat)
    low_eeat_blog = predictor_blog.predict_position(features_low_eeat)
    
    print(f"Блог (не YMYL) - Высокий E-E-A-T - Позиция: {high_eeat_blog['position']}")
    print(f"Блог (не YMYL) - Низкий E-E-A-T - Позиция: {low_eeat_blog['position']}")
    
    # ТЕСТ 4: Детальный анализ результата
    print("\n>> ТЕСТ 4: Детальный анализ результата (хороший контент, финансы)")
    result = predictor_finance.predict_position(features_good)
    
    print("\nПозиция и вероятности:")
    print(f"Предсказанная позиция: {result['position']}")
    for range_name, prob in result['probability'].items():
        print(f"Вероятность {range_name}: {prob:.2f}")
    
    print("\nТоп-5 факторов по взвешенной важности:")
    sorted_factors = sorted(
        result['weighted_scores'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for factor, score in sorted_factors[:5]:
        norm_score = result['feature_scores'][factor]
        print(f"{factor}: нормализованный скор = {norm_score:.2f}, взвешенный скор = {score:.3f}")

if __name__ == "__main__":
    test_calibrated_rank_predictor()
