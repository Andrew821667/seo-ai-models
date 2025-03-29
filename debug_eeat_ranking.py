
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.eeat_analyzer import EEATAnalyzer

# Базовый контент с низким E-E-A-T
basic_content = """
# Инвестирование в акции
Акции - это ценные бумаги, которые позволяют заработать на росте стоимости компаний.

## Как инвестировать
Выберите брокера и купите акции компаний, которые вам нравятся.

## Риски
Стоимость акций может как расти, так и падать.
"""

# Улучшенный контент с высоким E-E-A-T
enhanced_content = """
# Инвестирование в акции: руководство для начинающих
Автор: Иван Петров, финансовый аналитик с 10-летним опытом работы в Morgan Stanley

## Введение
По данным исследования Financial Behavior Institute (2024), более 65% людей не инвестируют из-за недостатка знаний. Данная статья основана на анализе 50+ научных публикаций.

## Что такое акции
Акции - это ценные бумаги, представляющие долю владения в компании. Согласно данным NYSE за 2024 год, капитализация фондового рынка США превышает $45 триллионов.

## Методология
В данной статье использованы данные из исследований Morningstar, Bloomberg, Reuters за период 2020-2024 гг.

## Об авторе
Иван Петров - сертифицированный финансовый аналитик (CFA), выпускник Высшей школы экономики.

*Опубликовано: 15.03.2025*
*Раскрытие информации: автор не владеет акциями компаний, упомянутых в статье.*
"""

keywords = ["инвестиции", "акции"]

# Тестовые сценарии
eeat_analyzer = EEATAnalyzer()

test_cases = [
    {"name": "1. Финансы - Базовый контент", "content": basic_content, "industry": "finance"},
    {"name": "2. Финансы - Улучшенный контент", "content": enhanced_content, "industry": "finance"},
    {"name": "3. Блог - Базовый контент", "content": basic_content, "industry": "blog"},
    {"name": "4. Блог - Улучшенный контент", "content": enhanced_content, "industry": "blog"}
]

for test in test_cases:
    print(f"\n===== ТЕСТ: {test['name']} =====")
    
    # Анализ E-E-A-T
    eeat_results = eeat_analyzer.analyze(test['content'])
    print(f"E-E-A-T метрики:")
    print(f"- Экспертность: {eeat_results['expertise_score']:.2f}")
    print(f"- Авторитетность: {eeat_results['authority_score']:.2f}")
    print(f"- Доверие: {eeat_results['trust_score']:.2f}")
    print(f"- Общий E-E-A-T: {eeat_results['overall_eeat_score']:.2f}")
    
    # Прогнозирование позиции
    advisor = SEOAdvisor(industry=test['industry'])
    original_predict = advisor.rank_predictor.predict_position
    
    def debug_predict(features, *args, **kwargs):
        print("\nПрогнозирование позиции:")
        print("- Входные метрики:")
        for key in ['expertise_score', 'authority_score', 'trust_score', 'overall_eeat_score']:
            if key in features:
                print(f"  * {key}: {features[key]:.2f}")
        
        result = original_predict(features, *args, **kwargs)
        
        print("- Результаты расчета:")
        print(f"  * Базовый контентный скор: {result['total_score']:.2f}")
        print(f"  * Базовая позиция (без E-E-A-T): {result['base_position']:.2f}")
        
        if 'eeat_components' in result:
            print("  * E-E-A-T компоненты:")
            for k, v in result['eeat_components'].items():
                print(f"    - {k}: {v:.2f}")
                
        print(f"  * Общая E-E-A-T корректировка: {result['eeat_adjustment']:.2f}")
        print(f"  * Улучшение позиции благодаря E-E-A-T: {result.get('eeat_position_improvement', 0):.2f}")
        print(f"  * Финальная предсказанная позиция: {result['position']:.2f}")
        
        return result
    
    advisor.rank_predictor.predict_position = debug_predict
    
    analysis = advisor.analyze_content(test['content'], keywords)
