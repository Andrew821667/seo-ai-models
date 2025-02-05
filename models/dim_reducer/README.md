# Dimension Reducer

Модуль для эффективного сжатия размерности и анализа SEO-характеристик текста.

## Основные компоненты

### Модель (model.py)
- Сжатие размерности с помощью нейронной сети
- SEO-специфичный слой внимания
- Оценка важности характеристик

### Обработка данных (features.py)
- Извлечение SEO-характеристик из текста
- Анализ HTML-структуры
- TF-IDF анализ и выделение ключевых слов

### Загрузка данных (data_loader.py)
- Настраиваемый DataLoader для SEO-данных
- Кэширование признаков
- Поддержка различных форматов файлов

### Обучение (trainer.py)
- Гибкая конфигурация процесса обучения
- Валидация модели
- Сохранение чекпоинтов

### Инференс (inference.py)
- Применение обученной модели
- Пакетная обработка
- Анализ важности признаков

## CLI-интерфейс

### Обучение модели
```bash
python -m models.dim_reducer.cli train \
    configs/dim_reducer.yml \
    data/train_data.csv \
    --val-data data/val_data.csv \
    --output-dir models/checkpoints
```

### Применение модели
```bash
python -m models.dim_reducer.cli predict \
    models/checkpoints/final_model.pt \
    data/test_data.csv \
    --output-dir predictions
```

### Анализ текста
```bash
python -m models.dim_reducer.cli analyze-text \
    models/checkpoints/final_model.pt \
    "Текст для анализа"
```

## Конфигурация

Пример конфигурационного файла `dim_reducer.yml`:
```yaml
# Архитектура модели
input_dim: 768
hidden_dim: 512
latent_dim: 256
num_attention_heads: 8
dropout_rate: 0.2

# Параметры обучения
batch_size: 32
num_epochs: 10
learning_rate: 1e-4
weight_decay: 0.01

# Параметры данных
max_features: 100
max_length: 512
model_name: 'bert-base-uncased'
```

## Использование в коде

```python
from models.dim_reducer.model import DimensionReducer
from models.dim_reducer.inference import DimReducerInference
from common.config.dim_reducer_config import DimReducerConfig

# Инициализация
config = DimReducerConfig()
model = DimensionReducer(config)

# Инференс
inference = DimReducerInference('path/to/model.pt', config)
results = inference.process_text("Текст для анализа")
```

## Зависимости

- PyTorch >= 2.0.0
- transformers >= 4.30.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.0.0
- beautifulsoup4 >= 4.9.0
- click >= 8.0.0

## Тестирование

```bash
# Запуск всех тестов
pytest tests/unit/test_dim_reducer_*.py

# Запуск конкретного теста
pytest tests/unit/test_dim_reducer_model.py
```
