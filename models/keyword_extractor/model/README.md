# Keyword Extractor

Модуль для извлечения и анализа ключевых слов из текста с использованием глубокого обучения.

## Возможности

- Автоматическое извлечение ключевых слов из текста
- Анализ трендов и важности ключевых слов
- Оценка релевантности и значимости
- Визуализация результатов
- Пакетная обработка больших объемов текста
- Поддержка различных языков (благодаря XLM-RoBERTa)

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/seo_ai_models.git
cd seo_ai_models

# Установка зависимостей
pip install -r requirements.txt
```

## Быстрый старт

### Использование через CLI

```bash
# Обучение модели
python -m model.cli train \
    --data-dir path/to/data \
    --output-dir path/to/output \
    --epochs 10

# Извлечение ключевых слов
python -m model.cli predict \
    --model-path path/to/model \
    --input path/to/input.txt \
    --output path/to/output.json
```

### Использование в коде

```python
from model.model import KeywordExtractorModel
from model.config.model_config import KeywordModelConfig

# Инициализация модели
config = KeywordModelConfig()
model = KeywordExtractorModel(config)

# Извлечение ключевых слов
texts = ["Your text here"]
keywords = model.extract_keywords(texts)

print(keywords)
```

## Структура проекта

```
model/
├── config/           # Конфигурации
├── model/           # Основные компоненты модели
├── monitoring/      # Мониторинг и логирование
├── utils/          # Вспомогательные функции
└── tests/          # Тесты
```

## Конфигурация

Основные параметры модели можно настроить через конфигурационный файл:

```json
{
    "model": {
        "model_name": "xlm-roberta-base",
        "max_length": 512,
        "hidden_dim": 256
    },
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 10
    }
}
```

## Обучение

### Подготовка данных

Данные для обучения должны быть в формате JSON:

```json
[
    {
        "text": "Your text here",
        "keywords": ["key", "words"],
        "trends": [0.8, 0.6]
    }
]
```

### Запуск обучения

```bash
python -m model.train \
    --config config.json \
    --data-dir data/ \
    --output-dir outputs/
```

## Визуализация

Модуль включает инструменты для визуализации:
- Распределение ключевых слов
- История обучения
- Анализ трендов

```python
from model.utils.visualization import KeywordVisualizer

visualizer = KeywordVisualizer()
visualizer.plot_keyword_distribution(keywords)
```

## Мониторинг

Встроенный мониторинг производительности:
- Использование ресурсов
- Время обработки
- Метрики качества

## Тестирование

```bash
# Запуск всех тестов
pytest tests/

# Запуск конкретной группы тестов
pytest tests/test_model.py
```

## Участие в разработке

См. [CONTRIBUTING.md](CONTRIBUTING.md) для информации о том, как принять участие в проекте.

## Лицензия

Проект распространяется под лицензией MIT. См. [LICENSE](LICENSE) для деталей.
