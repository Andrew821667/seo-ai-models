
from models.seo_advisor.rank_predictor import RankPredictor

# Создаем тестовые данные
test_features = {
    'keyword_density': 0.02,
    'content_length': 0.8,
    'readability_score': 0.75,
    'meta_tags_score': 1.0,
    'header_structure_score': 0.9
}

# Инициализируем предиктор
predictor = RankPredictor(input_size=5)

# Тестируем прогноз позиции
position = predictor.predict_position(test_features)
print(f"Прогноз позиции: {position:.2f}")

# Тестируем оценку силы страницы
strengths = predictor.evaluate_page_strength(test_features)
print("\nОценка факторов:")
for factor, strength in strengths.items():
    print(f"{factor}: {strength}")
