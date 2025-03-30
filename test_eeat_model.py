
import sys
sys.path.append('/content/seo-ai-models')
from models.seo_advisor.enhanced_eeat_analyzer import EnhancedEEATAnalyzer

# Пример контента для анализа
test_content = """
# Руководство по инвестированию для начинающих

Инвестирование - это способ обеспечить свое финансовое будущее. В этой статье мы рассмотрим основные подходы к инвестированию для тех, кто только начинает свой путь.

## Что такое инвестиции?

Инвестиции - это вложение денег с целью получения дохода или увеличения их стоимости с течением времени. Важно понимать разницу между сбережениями и инвестициями.

## Почему стоит инвестировать?

По данным исследования Morgan Stanley, средняя годовая доходность индекса S&P 500 за последние 30 лет составила около 10%.

## Основные виды инвестиций

1. Акции - доли в компаниях
2. Облигации - долговые ценные бумаги
3. Недвижимость - материальные активы
4. ETF и взаимные фонды - диверсифицированные инвестиции

## Советы от экспертов

Профессор экономики Джереми Сигел в своей книге "Stocks for the Long Run" рекомендует...

Источники:
- Investopedia.com
- SEC.gov
- "The Intelligent Investor" by Benjamin Graham
"""

# Инициализация анализатора
analyzer = EnhancedEEATAnalyzer(model_path='/content/seo-ai-models/models/checkpoints/eeat_best_model.joblib')

# Анализ контента
print("Анализ примера финансового контента (YMYL):")
results = analyzer.analyze(test_content, industry='finance')

# Вывод результатов
print("\nРезультаты анализа E-E-A-T:")
print(f"Expertise Score: {results['expertise_score']:.4f}")
print(f"Authority Score: {results['authority_score']:.4f}")
print(f"Trust Score: {results['trust_score']:.4f}")
print(f"Overall E-E-A-T Score: {results['overall_eeat_score']:.4f}")
print(f"Использование модели МО: {results.get('ml_model_used', 'Нет данных')}")

# Вывод ключевых рекомендаций
print("\nТоп-3 рекомендации:")
for i, rec in enumerate(results.get('recommendations', [])[:3], 1):
    print(f"{i}. {rec}")
