# SEO AI Models

## Описание проекта

Библиотека моделей и инструментов искусственного интеллекта для SEO-оптимизации контента.

## Новая структура проекта

seo_ai_models/
├── api/                  # API для доступа к моделям
├── common/               # Общие компоненты
│   ├── config/           # Конфигурационные файлы
│   └── utils/            # Вспомогательные утилиты
├── data/                 # Данные, включая модели ML
│   └── models/           # Предобученные модели
│       └── eeat/         # Модели для E-E-A-T анализа
├── models/               # Основные модели 
│   ├── dim_reducer/      # Модели понижения размерности
│   ├── keyword_extractor/# Экстракторы ключевых слов
│   └── seo_advisor/      # Основные компоненты SEO Advisor
│       ├── analyzers/    # Анализаторы контента
│       │   └── eeat/     # Компоненты E-E-A-T
│       ├── predictors/   # Предикторы ранжирования
│       └── suggester/    # Генераторы рекомендаций

## Основные компоненты

- **SEOAdvisor**: Основной класс для анализа и оптимизации контента
- **EEATAnalyzer**: Анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
- **EnhancedEEATAnalyzer**: Улучшенный анализатор E-E-A-T с использованием машинного обучения
- **CalibratedRankPredictor**: Предсказание позиций в поисковой выдаче

## Установка

pip install -e .

## Примеры использования

from seo_ai_models.models.seo_advisor import SEOAdvisor

# Инициализация советника для определенной отрасли
advisor = SEOAdvisor(industry='finance')

# Анализ контента
result = advisor.analyze_content(content, target_keywords)

# Работа с отчетом
print(f"Прогнозируемая позиция: {result.predicted_position}")
print(f"E-E-A-T score: {result.content_metrics['overall_eeat_score']}")

Более подробные примеры можно найти в директории `examples/`.

## Расширенное использование E-E-A-T анализатора

from seo_ai_models.models.seo_advisor.analyzers.eeat import EnhancedEEATAnalyzer

# Путь к модели
model_path = "seo_ai_models/data/models/eeat/eeat_best_model.joblib"

# Инициализация анализатора с моделью машинного обучения
analyzer = EnhancedEEATAnalyzer(model_path=model_path)

# Анализ контента для определенной отрасли
result = analyzer.analyze(content, industry='finance')

# Работа с результатами
print(f"Expertise Score: {result['expertise_score']:.4f}")
print(f"Authority Score: {result['authority_score']:.4f}")
print(f"Trust Score: {result['trust_score']:.4f}")
print(f"Overall E-E-A-T Score: {result['overall_eeat_score']:.4f}")

## Контакты

Для вопросов и предложений, пожалуйста, создавайте issues в репозитории проекта.
