
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.eeat_analyzer import EEATAnalyzer
from models.seo_advisor.calibrated_rank_predictor import CalibratedRankPredictor

# Тестовый контент
test_content = """
# Инвестирование в акции: руководство
Автор: Иван Петров, финансовый аналитик

## Введение
Многие люди не инвестируют из-за недостатка знаний.

## Что такое акции
Акции - это ценные бумаги, представляющие долю владения в компании.

## Методология
Данные получены из открытых источников.
"""

keywords = ["инвестиции", "акции"]

# 1. Проверка E-E-A-T анализатора
print("=== 1. ПРОВЕРКА E-E-A-T АНАЛИЗАТОРА ===")
eeat_analyzer = EEATAnalyzer()
eeat_results = eeat_analyzer.analyze(test_content)
print(f"Экспертность: {eeat_results['expertise_score']:.2f}")
print(f"Авторитетность: {eeat_results['authority_score']:.2f}")
print(f"Доверие: {eeat_results['trust_score']:.2f}")
print(f"Общий E-E-A-T: {eeat_results['overall_eeat_score']:.2f}")

# 2. Проверка интеграции в SEOAdvisor
print("\n=== 2. ПРОВЕРКА ИНТЕГРАЦИИ В SEOAdvisor ===")
advisor = SEOAdvisor(industry="finance")
# Отслеживаем вызов analyze_content и перехватываем данные
original_analyze = advisor.analyze_content
def debug_analyze(*args, **kwargs):
    print("Вызов analyze_content")
    result = original_analyze(*args, **kwargs)
    print("Содержимое content_metrics:")
    for key in ['expertise_score', 'authority_score', 'trust_score', 'overall_eeat_score']:
        if key in result.content_metrics:
            print(f"  - {key}: {result.content_metrics[key]}")
        else:
            print(f"  - {key}: ОТСУТСТВУЕТ")
    return result

advisor.analyze_content = debug_analyze

# 3. Проверка передачи в предиктор
print("\n=== 3. ПРОВЕРКА ПЕРЕДАЧИ В ПРЕДИКТОР ===")
predictor = CalibratedRankPredictor(industry="finance")
original_predict = predictor.predict_position
def debug_predict(features, *args, **kwargs):
    print("Вызов predict_position")
    print("Содержимое features:")
    for key in ['expertise_score', 'authority_score', 'trust_score', 'overall_eeat_score']:
        if key in features:
            print(f"  - {key}: {features[key]}")
        else:
            print(f"  - {key}: ОТСУТСТВУЕТ")
    
    print("\nОсновные параметры:")
    print(f"  - Отрасль: {predictor.industry}")
    print(f"  - YMYL: {predictor.ymyl_industries.get(predictor.industry, False)}")
    
    result = original_predict(features, *args, **kwargs)
    print("\nРезультаты предсказания:")
    print(f"  - Позиция: {result['position']}")
    if 'eeat_adjustment' in result:
        print(f"  - E-E-A-T корректировка: {result['eeat_adjustment']}")
    if 'adjusted_score' in result:
        print(f"  - Скорректированный скор: {result['adjusted_score']}")
    if 'total_score' in result:
        print(f"  - Базовый скор: {result['total_score']}")
    return result

predictor.predict_position = debug_predict
advisor.rank_predictor.predict_position = debug_predict

# Запускаем анализ
analysis = advisor.analyze_content(test_content, keywords)
print("\n=== 4. ИТОГОВАЯ ОЦЕНКА ===")
print(f"Предсказанная позиция: {analysis.predicted_position}")
