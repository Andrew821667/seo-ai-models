
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.eeat_analyzer import EEATAnalyzer

def test_seo_ranking(content, industry, keywords, level_name):
    """Тестирование SEO-ранжирования для контента"""
    advisor = SEOAdvisor(industry=industry)
    eeat_analyzer = EEATAnalyzer()
    
    # E-E-A-T анализ
    eeat_results = eeat_analyzer.analyze(content)
    
    # SEO анализ
    seo_results = advisor.analyze_content(content, keywords)
    
    print(f"{industry.upper()} - {level_name}:")
    print(f"E-E-A-T: {eeat_results['overall_eeat_score']:.2f}, Позиция: {seo_results.predicted_position:.2f}")
    return {
        'eeat': eeat_results['overall_eeat_score'],
        'position': seo_results.predicted_position
    }

# Различные уровни E-E-A-T контента
levels = {
    "Очень низкий E-E-A-T": """
        # Инвестирование
        Акции это способ заработать. Купите акции.
    """,
    
    "Низкий E-E-A-T": """
        # Инвестирование в акции
        Акции - это ценные бумаги, которые позволяют заработать на росте стоимости компаний.

        ## Как инвестировать
        Выберите брокера и купите акции компаний, которые вам нравятся.

        ## Риски
        Стоимость акций может как расти, так и падать.
    """,
    
    "Средний E-E-A-T": """
        # Инвестирование в акции
        Автор: Иван Петров, финансовый аналитик

        ## Введение
        Многие люди не инвестируют из-за недостатка знаний.

        ## Что такое акции
        Акции - это ценные бумаги, представляющие долю владения в компании.

        ## Как начать инвестировать
        1. Определите свои цели
        2. Выберите брокера
        3. Создайте диверсифицированный портфель

        ## Риски инвестирования
        Инвестирование всегда сопряжено с рисками. Статистика показывает, что в долгосрочной 
        перспективе рынок акций растет на 7-10% в год.

        ## Об авторе
        Иван Петров работает финансовым аналитиком и пишет статьи об инвестициях.

        *Опубликовано: март 2025*
    """,
    
    "Высокий E-E-A-T": """
        # Инвестирование в акции: руководство для начинающих
        Автор: Иван Петров, финансовый аналитик с 10-летним опытом работы в Morgan Stanley

        ## Введение
        По данным исследования Financial Behavior Institute (2024), более 65% людей не инвестируют 
        из-за недостатка знаний. Данная статья основана на анализе 50+ научных публикаций и 
        практическом опыте инвестирования.

        ## Что такое акции
        Акции - это ценные бумаги, представляющие долю владения в компании. Согласно данным NYSE 
        за 2024 год, капитализация фондового рынка США превышает $45 триллионов.

        ## Как начать инвестировать
        1. Определите свои финансовые цели и горизонт инвестирования
        2. Выберите надежного брокера с лицензией SEC
        3. Создайте диверсифицированный портфель, соответствующий вашему риск-профилю
        4. Регулярно пересматривайте и ребалансируйте ваш портфель

        ## Методология
        В данной статье использованы данные из исследований Morningstar, Bloomberg, Reuters 
        за период 2020-2024 гг. Все статистические данные проверены и актуальны на 15.03.2025.

        ## Об авторе
        Иван Петров - сертифицированный финансовый аналитик (CFA), выпускник Высшей школы экономики, 
        автор книги "Инвестиции для начинающих" (2023).

        *Опубликовано: 15.03.2025*
        *Раскрытие информации: автор не владеет акциями компаний, упомянутых в статье.*
    """
}

# Тестируемые отрасли
industries = ["finance", "health", "blog", "travel"]
keywords = ["инвестиции", "акции", "финансы"]

# Проведем тестирование
print("===== ИТОГОВОЕ ТЕСТИРОВАНИЕ ВЛИЯНИЯ E-E-A-T НА РАНЖИРОВАНИЕ =====\n")

for industry in industries:
    print(f"Отрасль: {industry.upper()}")
    results = []
    
    for level_name, content in levels.items():
        result = test_seo_ranking(content, industry, keywords, level_name)
        results.append((level_name, result))
    
    # Вычислим разницу между худшим и лучшим контентом
    worst = results[0][1]['position']
    best = results[-1][1]['position']
    diff = worst - best
    
    print(f"Разница между худшим и лучшим контентом: {diff:.2f} позиций\n")

print("===== КРОСС-ОТРАСЛЕВОЕ СРАВНЕНИЕ =====\n")
print("Влияние высокого E-E-A-T на различные отрасли:")

ymyl_results = []
non_ymyl_results = []

for industry in industries:
    for level_name, content in levels.items():
        if level_name == "Высокий E-E-A-T":
            result = test_seo_ranking(content, industry, keywords, industry)
            if industry in ["finance", "health"]:
                ymyl_results.append((industry, result))
            else:
                non_ymyl_results.append((industry, result))

print("\nYMYL отрасли:")
for industry, result in ymyl_results:
    print(f"{industry}: E-E-A-T {result['eeat']:.2f} → Позиция {result['position']:.2f}")

print("\nНе-YMYL отрасли:")
for industry, result in non_ymyl_results:
    print(f"{industry}: E-E-A-T {result['eeat']:.2f} → Позиция {result['position']:.2f}")
