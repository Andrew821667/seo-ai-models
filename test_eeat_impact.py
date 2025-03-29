
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.eeat_analyzer import EEATAnalyzer

# Базовый финансовый контент с низким E-E-A-T
basic_finance_content = """
# Инвестирование в акции
Акции - это ценные бумаги, которые позволяют заработать на росте стоимости компаний.

## Как инвестировать
Выберите брокера и купите акции компаний, которые вам нравятся.

## Риски
Стоимость акций может как расти, так и падать.

## Заключение
Инвестируйте с умом.
"""

# Улучшенный финансовый контент с высоким E-E-A-T
enhanced_finance_content = """
# Инвестирование в акции: руководство для начинающих
Автор: Иван Петров, финансовый аналитик с 10-летним опытом работы в Morgan Stanley

## Введение
По данным исследования Financial Behavior Institute (2024), более 65% людей не инвестируют из-за недостатка знаний. Данная статья основана на анализе 50+ научных публикаций и практическом опыте инвестирования.

## Что такое акции
Акции - это ценные бумаги, представляющие долю владения в компании. Согласно данным NYSE за 2024 год, капитализация фондового рынка США превышает $45 триллионов.

## Как начать инвестировать
1. Определите свои финансовые цели и горизонт инвестирования
2. Выберите надежного брокера с лицензией SEC
3. Создайте диверсифицированный портфель, соответствующий вашему риск-профилю
4. Регулярно пересматривайте и ребалансируйте ваш портфель

## Риски инвестирования
Профессор Гарвардской школы бизнеса Джереми Сигел отмечает: "Инвестирование всегда сопряжено с рисками, но исторически индекс S&P 500 демонстрирует среднегодовую доходность около 10% на протяжении последних 100 лет с учетом дивидендов и инфляции".

## Налоговые аспекты
По данным IRS, долгосрочные инвестиции (более 1 года) облагаются по льготной ставке 0-20%, в зависимости от вашей налоговой категории.

## Заключение
Инвестируйте регулярно, диверсифицируйте портфель и придерживайтесь долгосрочной стратегии.

## Методология
В данной статье использованы данные из исследований Morningstar, Bloomberg, Reuters за период 2020-2024 гг. Все статистические данные проверены и актуальны на 15.03.2025.

## Об авторе
Иван Петров - сертифицированный финансовый аналитик (CFA), выпускник Высшей школы экономики, автор книги "Инвестиции для начинающих" (2023). Имеет 10 лет опыта работы в инвестиционных компаниях и преподает курс "Финансовые рынки" в МГУ.

*Опубликовано: 15.03.2025*
*Последнее обновление: 28.03.2025*
*Раскрытие информации: автор не владеет акциями компаний, упомянутых в статье. Данный материал предоставляется исключительно в образовательных целях и не является инвестиционной рекомендацией.*
"""

# Финансовый контент со средним E-E-A-T
medium_finance_content = """
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
Инвестирование всегда сопряжено с рисками. Статистика показывает, что в долгосрочной перспективе рынок акций растет на 7-10% в год.

## Заключение
Инвестируйте осмотрительно и на долгий срок.

## Об авторе
Иван Петров работает финансовым аналитиком и пишет статьи об инвестициях.

*Опубликовано: март 2025*
"""

# Ключевые слова
finance_keywords = ["инвестиции", "акции", "фондовый рынок"]

# Анализ контента с разным уровнем E-E-A-T
print("===== ТЕСТ ВЛИЯНИЯ E-E-A-T НА ПРЕДСКАЗАНИЕ ПОЗИЦИИ =====")

# Создаем три экземпляра SEO Advisor для finance и не-YMYL отрасли
finance_advisor = SEOAdvisor(industry="finance")  # YMYL
blog_advisor = SEOAdvisor(industry="blog")  # не-YMYL

print("\n1. ВЛИЯНИЕ НА ФИНАНСОВУЮ ТЕМАТИКУ (YMYL)")

print("\n- Базовый контент (низкий E-E-A-T):")
eeat_analyzer = EEATAnalyzer()
basic_eeat = eeat_analyzer.analyze(basic_finance_content)
print(f"  Общий E-E-A-T: {basic_eeat['overall_eeat_score']:.2f}")
basic_finance_analysis = finance_advisor.analyze_content(basic_finance_content, finance_keywords)
print(f"  Предсказанная позиция: {basic_finance_analysis.predicted_position:.2f}")

print("\n- Средний контент (средний E-E-A-T):")
medium_eeat = eeat_analyzer.analyze(medium_finance_content)
print(f"  Общий E-E-A-T: {medium_eeat['overall_eeat_score']:.2f}")
medium_finance_analysis = finance_advisor.analyze_content(medium_finance_content, finance_keywords)
print(f"  Предсказанная позиция: {medium_finance_analysis.predicted_position:.2f}")

print("\n- Улучшенный контент (высокий E-E-A-T):")
enhanced_eeat = eeat_analyzer.analyze(enhanced_finance_content)
print(f"  Общий E-E-A-T: {enhanced_eeat['overall_eeat_score']:.2f}")
enhanced_finance_analysis = finance_advisor.analyze_content(enhanced_finance_content, finance_keywords)
print(f"  Предсказанная позиция: {enhanced_finance_analysis.predicted_position:.2f}")

print("\n2. СРАВНЕНИЕ С НЕ-YMYL ТЕМАТИКОЙ (БЛОГ)")

print("\n- Базовый контент (низкий E-E-A-T):")
basic_blog_analysis = blog_advisor.analyze_content(basic_finance_content, finance_keywords)
print(f"  Предсказанная позиция: {basic_blog_analysis.predicted_position:.2f}")

print("\n- Улучшенный контент (высокий E-E-A-T):")
enhanced_blog_analysis = blog_advisor.analyze_content(enhanced_finance_content, finance_keywords)
print(f"  Предсказанная позиция: {enhanced_blog_analysis.predicted_position:.2f}")

print("\n3. РАЗНИЦА В ПОЗИЦИЯХ:")
print(f"Для YMYL: {basic_finance_analysis.predicted_position - enhanced_finance_analysis.predicted_position:.2f} позиций")
print(f"Для не-YMYL: {basic_blog_analysis.predicted_position - enhanced_blog_analysis.predicted_position:.2f} позиций")
