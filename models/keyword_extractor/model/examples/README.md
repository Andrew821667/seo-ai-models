# Примеры использования Keyword Extractor

В этой директории содержатся примеры использования модуля извлечения ключевых слов.

## Структура

- `basic_usage.py` - базовые примеры использования
- `advanced_usage.py` - продвинутые примеры с визуализацией и анализом

## Базовые примеры

### 1. Простое извлечение ключевых слов

```python
from model.model import KeywordExtractorModel

model = KeywordExtractorModel()
keywords = model.extract_keywords(["Your text here"])
```

### 2. Пакетная обработка

```python
texts = ["First text", "Second text", "Third text"]
results = model.extract_keywords(texts, batch_size=2)
```

### 3. Сохранение и загрузка модели

```python
# Сохранение
model.save_pretrained("saved_model")

# Загрузка
loaded_model = KeywordExtractorModel.from_pretrained("saved_model")
```

## Продвинутые примеры

### 1. Визуализация результатов

```python
from model.utils.visualization import KeywordVisualizer

visualizer = KeywordVisualizer(save_dir="visuals")
visualizer.plot_keyword_distribution(keywords)
```

### 2. Анализ ошибок

```python
from model.utils.analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer()
analysis = analyzer.analyze_predictions(predictions, targets, texts)
```

### 3. Мониторинг производительности

```python
from model.monitoring.performance import PerformanceMonitor

monitor = PerformanceMonitor()
start_time = monitor.start_batch()
# ... обработка ...
metrics = monitor.end_batch(start_time, batch_size)
```

## Запуск примеров

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите базовые примеры:
```bash
python model/examples/basic_usage.py
```

3. Запустите продвинутые примеры:
```bash
python model/examples/advanced_usage.py
```

## Структура выходных данных

### Формат ключевых слов:
```python
{
    'keywords': [
        {
            'keyword': 'example',
            'score': 0.85,
            'trend_score': 0.65
        },
        # ...
    ],
    'processing_time': 0.15
}
```

### Формат анализа:
```python
{
    'error_types': {
        'false_positives': {'count': 5, 'percentage': 10.5},
        'false_negatives': {'count': 3, 'percentage': 6.3}
    },
    'length_analysis': {
        '100': {
            'avg_precision': 0.85,
            'std_precision': 0.12
        }
    }
}
```

## Дополнительные ресурсы

- Полная документация API: `docs/api.md`
- Техническая документация: `docs/architecture.md`
- Руководство по участию: `CONTRIBUTING.md`
