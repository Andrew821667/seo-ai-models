# SEO AI Models

Набор алгоритмов и моделей искусственного интеллекта для SEO-анализа и оптимизации контента.

## О проекте

SEO AI Models предоставляет инструменты для анализа и оптимизации контента с точки зрения SEO. Проект включает в себя несколько ключевых компонентов:

- **TextProcessor**: Анализ и обработка текста, включая токенизацию, определение языка, извлечение заголовков и оценку читабельности.
- **ContentAnalyzer**: Комплексный анализ контента с расчетом метрик, анализом ключевых слов и оценкой структуры.
- **EEATAnalyzer**: Анализ Experience, Expertise, Authoritativeness, Trustworthiness контента в соответствии с рекомендациями Google.
- **EnhancedEEATAnalyzer**: Расширенный анализатор E-E-A-T с поддержкой ML-моделей.
- **CalibratedRankPredictor**: Предсказание позиций в выдаче на основе различных факторов.
- **SEOAdvisor**: Интегрированный компонент, объединяющий все вышеперечисленное для полного SEO-анализа.

## Установка

### Требования

- Python 3.8+
- pip

### Установка из исходного кода

```bash
# Клонирование репозитория
git clone https://github.com/Andrew821667/seo-ai-models.git
cd seo-ai-models

# Установка зависимостей
pip install -r requirements.txt

# Установка проекта в режиме разработки
pip install -e .
```

## Использование

### TextProcessor

```python
from seo_ai_models.common.utils.text_processing import TextProcessor

processor = TextProcessor()

# Определение языка
lang = processor.detect_language("Это текст на русском языке")
print(f"Определенный язык: {lang}")  # Вывод: "ru"

# Токенизация текста
tokens = processor.tokenize("Это пример текста для токенизации")
print(tokens)

# Извлечение заголовков из Markdown
headers = processor.extract_headers("# Заголовок статьи\n## Подзаголовок")
print(headers)
```

### ContentAnalyzer

```python
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer

analyzer = ContentAnalyzer()

# Анализ текста
text = "# Заголовок статьи\n\nЭто пример текста для анализа содержимого."
metrics = analyzer.analyze_text(text)
print(metrics)

# Анализ ключевых слов
keywords = ["анализ", "текст", "содержимое"]
keyword_metrics = analyzer.extract_keywords(text, keywords)
print(keyword_metrics)
```

### EEATAnalyzer

```python
from seo_ai_models.models.seo_advisor.analyzers.eeat.eeat_analyzer import EEATAnalyzer

analyzer = EEATAnalyzer()

# Анализ E-E-A-T
text = "# Статья о здоровье\n\nПо данным исследований..."
results = analyzer.analyze(text, industry="health")
print(results)
```

### SEOAdvisor

```python
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor

advisor = SEOAdvisor(industry="blog")

# Комплексный анализ
text = "# Заголовок статьи\n\nСодержимое статьи..."
keywords = ["ключевое слово", "тема"]
report = advisor.analyze_content(text, keywords)
print(report.predicted_position)
print(report.content_metrics)
print(report.content_quality.strengths)
```

## Структура проекта

```
seo_ai_models/
├── common/                  # Общие компоненты
│   ├── config/              # Конфигурации
│   └── utils/               # Утилиты (включая TextProcessor)
├── models/                  # Модели и алгоритмы
│   ├── dim_reducer/         # Редуктор размерности
│   ├── keyword_extractor/   # Извлечение ключевых слов
│   └── seo_advisor/         # SEO-анализатор
│       ├── analyzers/       # Анализаторы контента
│       │   └── eeat/        # Анализаторы E-E-A-T
│       ├── predictors/      # Предикторы ранжирования
│       └── suggester/       # Генератор рекомендаций
└── data/                    # Данные для моделей
    └── models/              # Предобученные модели
```

## Тестирование

Проект включает модульные и интеграционные тесты. Для запуска тестов используйте:

```bash
# Установка зависимостей для тестирования
pip install -r requirements.txt

# Запуск всех тестов
cd /path/to/seo-ai-models
PYTHONPATH=/path/to/seo-ai-models pytest tests/

# Запуск конкретного модульного теста
pytest tests/unit/seo_advisor/analyzers/test_content_analyzer.py
```

## Планы по развитию

В ближайших планах:
- Расширение документации с примерами и кейсами использования
- Добавление поддержки многоязычного анализа
- Интеграция с другими инструментами SEO
- Разработка веб-интерфейса
- Добавление визуализации результатов

## Лицензия

MIT
