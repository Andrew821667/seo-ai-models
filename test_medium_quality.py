
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.eeat_analyzer import EEATAnalyzer

# Текст среднего качества для категории "Финансы"
finance_medium_text = """
# Как начать инвестировать

Инвестирование - способ приумножить ваши деньги. Рассмотрим основные шаги.

## Выбор брокера
Выберите брокера с низкими комиссиями. Популярные брокеры:
* Брокер А
* Брокер Б
* Брокер В

## Основные типы инвестиций
1. Акции - доли в компаниях
2. Облигации - долговые инструменты
3. ETF - биржевые фонды

## Стратегии
Самые распространенные стратегии:
- Долгосрочное инвестирование
- Трейдинг
- Пассивное инвестирование

## Риски
При инвестировании есть риски потери денег.
"""

# Проведем полный анализ текста
print("===== ТЕСТ РАНЖИРОВАНИЯ ТЕКСТА СРЕДНЕЙ СЛОЖНОСТИ =====\n")

# YMYL тематики
for industry in ["finance", "health"]:
    advisor = SEOAdvisor(industry=industry)
    eeat_analyzer = EEATAnalyzer()
    
    # E-E-A-T анализ
    eeat_results = eeat_analyzer.analyze(finance_medium_text)
    
    # SEO анализ
    keywords = ["инвестиции", "брокер", "акции"]
    seo_results = advisor.analyze_content(finance_medium_text, keywords)
    
    print(f"Отрасль: {industry.upper()}")
    print(f"E-E-A-T метрики:")
    print(f"- Экспертность: {eeat_results['expertise_score']:.2f}")
    print(f"- Авторитетность: {eeat_results['authority_score']:.2f}")
    print(f"- Доверие: {eeat_results['trust_score']:.2f}")
    print(f"- Общий E-E-A-T: {eeat_results['overall_eeat_score']:.2f}")
    print(f"\nРанжирование:")
    print(f"- Предсказанная позиция: {seo_results.predicted_position:.2f}")
    
    # Дополнительные метрики
    print(f"\nРекомендации:")
    for i, strength in enumerate(seo_results.content_quality.strengths[:3]):
        print(f"- {strength}")
        
    print(f"\nСлабые стороны:")
    for i, weakness in enumerate(seo_results.content_quality.weaknesses[:3]):
        print(f"- {weakness}")
    
    print("\n" + "="*50 + "\n")
    
# Не-YMYL тематика для сравнения
advisor = SEOAdvisor(industry="blog")
seo_results = advisor.analyze_content(finance_medium_text, keywords)
print(f"Отрасль: BLOG (не-YMYL)")
print(f"- Предсказанная позиция: {seo_results.predicted_position:.2f}")
