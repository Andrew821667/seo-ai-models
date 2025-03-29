
import sys
sys.path.append('/content/seo-ai-models')

from models.seo_advisor.advisor import SEOAdvisor
from models.seo_advisor.eeat_analyzer import EEATAnalyzer

# Тестовый финансовый контент
test_finance_content = """
# Инвестирование в акции: руководство для начинающих
Автор: Иван Петров, финансовый аналитик с 10-летним опытом

## Введение
По данным исследования Morgan Stanley (2024), более 65% людей не инвестируют из-за недостатка знаний.

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

## Методология
Данные для статьи получены из отчетов аналитических агентств и официальной статистики фондового рынка за 2023-2024 годы.

## Об авторе
Иван Петров - сертифицированный финансовый аналитик (CFA), выпускник Высшей школы экономики, автор книги "Инвестиции для начинающих".

*Опубликовано: 15.03.2025*
*Раскрытие информации: автор не владеет акциями компаний, упомянутых в статье.*
"""

# Сначала проверим только E-E-A-T метрики без интеграции с SEOAdvisor
print("ТЕСТ УЛУЧШЕННОГО EEAT АНАЛИЗАТОРА")
eeat_analyzer = EEATAnalyzer()
eeat_results = eeat_analyzer.analyze(test_finance_content)
print(f"Экспертность: {eeat_results['expertise_score']:.2f}")
print(f"Авторитетность: {eeat_results['authority_score']:.2f}")
print(f"Доверие: {eeat_results['trust_score']:.2f}")
print(f"Структурный скор: {eeat_results['structural_score']:.2f}")
print(f"Общий E-E-A-T: {eeat_results['overall_eeat_score']:.2f}")
print(f"Рекомендации: {eeat_results['recommendations']}")

# Теперь проверим с полным SEOAdvisor
print("\nТЕСТ ИНТЕГРАЦИИ С SEO ADVISOR")
finance_keywords = ["инвестиции", "акции", "фондовый рынок"]
finance_advisor = SEOAdvisor(industry="finance")
try:
    finance_analysis = finance_advisor.analyze_content(test_finance_content, finance_keywords)
    print(f"Успешный анализ")
except Exception as e:
    print(f"Ошибка анализа: {e}")
