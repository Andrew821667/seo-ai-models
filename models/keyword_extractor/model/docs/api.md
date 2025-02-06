# API Documentation

## Основные компоненты

### KeywordExtractorModel

Основная модель для извлечения ключевых слов.

```python
class KeywordExtractorModel(nn.Module):
    """
    Параметры:
        config (KeywordModelConfig): Конфигурация модели
        cache_dir (Optional[Path]): Директория для кэширования
    
    Методы:
        extract_keywords(
            texts: Union[str, List[str]],
            threshold: float = 0.5
        ) -> List[Dict[str, Union[str, float]]]
        
        save_pretrained(path: Union[str, Path]) -> None
        
        @classmethod
        from_pretrained(
            model_dir: Union[str, Path],
            cache_dir: Optional[Path] = None
        ) -> 'KeywordExtractorModel'
    """
```

#### Пример использования:

```python
from model.model import KeywordExtractorModel
from model.config import KeywordModelConfig

config = KeywordModelConfig(
    model_name="xlm-roberta-base",
    max_length=512
)

model = KeywordExtractorModel(config)
results = model.extract_keywords(
    ["Your text here"],
    threshold=0.5
)
```

### KeywordProcessor

Процессор для обработки текстовых данных.

```python
class KeywordProcessor:
    """
    Параметры:
        config (KeywordModelConfig): Конфигурация модели
        cache_dir (Optional[Path]): Директория для кэширования
        
    Методы:
        encode_texts(
            texts: Union[str, List[str]]
        ) -> Dict[str, torch.Tensor]
        
        decode_keywords(
            token_ids: torch.Tensor,
            scores: Optional[torch.Tensor] = None
        ) -> Union[List[str], List[Dict[str, Union[str, float]]]]
    """
```

## Конфигурация

### KeywordModelConfig

```python
class KeywordModelConfig(BaseModel):
    """
    Параметры:
        model_name (str): Название базовой модели
        max_length (int): Максимальная длина последовательности
        input_dim (int): Размерность входных данных
        hidden_dim (int): Размерность скрытого слоя
        dropout_rate (float): Вероятность dropout
        num_heads (int): Количество голов внимания
    """
```

### KeywordTrainingConfig

```python
class KeywordTrainingConfig(BaseModel):
    """
    Параметры:
        learning_rate (float): Скорость обучения
        batch_size (int): Размер батча
        num_epochs (int): Количество эпох
        warmup_steps (int): Шаги прогрева
        max_grad_norm (float): Ограничение градиентов
    """
```

## Обучение

### KeywordExtractorTrainer

```python
class KeywordExtractorTrainer:
    """
    Параметры:
        model (KeywordExtractorModel): Модель для обучения
        config (KeywordTrainingConfig): Конфигурация обучения
        device (str): Устройство для обучения
        
    Методы:
        train_step(
            batch: Dict[str, torch.Tensor]
        ) -> Dict[str, float]
        
        validate(
            val_dataloader
        ) -> Dict[str, float]
        
        train(
            train_dataloader,
            val_dataloader,
            save_dir: Optional[Path] = None
        ) -> None
    """
```

## Утилиты

### DataUtils

```python
def load_dataset(
    data_path: Union[str, Path],
    processor: KeywordProcessor,
    max_samples: Optional[int] = None,
    validation_split: float = 0.2
) -> Tuple[KeywordDataset, KeywordDataset]:
    """Загрузка данных"""

def create_dataloaders(
    train_dataset: KeywordDataset,
    val_dataset: KeywordDataset,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Создание загрузчиков данных"""
```

### Metrics

```python
class KeywordMetrics:
    """
    Методы:
        calculate_metrics(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Dict[str, float]
        
        calculate_trend_metrics(
            predictions: torch.Tensor,
            targets: torch.Tensor
        ) -> Dict[str, float]
    """
```

## CLI Интерфейс

### Команды

```bash
# Обучение
python -m model.cli train --help

Options:
  --config PATH           Путь к конфигурации
  --data-dir PATH        Директория с данными  [required]
  --output-dir PATH      Директория для результатов  [required]
  --epochs INTEGER       Количество эпох
  --batch-size INTEGER   Размер батча
  --device TEXT          Устройство (cuda/cpu)

# Предсказание
python -m model.cli predict --help

Options:
  --model-path PATH      Путь к модели  [required]
  --input PATH          Входной файл/директория  [required]
  --output PATH         Выходной файл  [required]
  --threshold FLOAT     Порог уверенности
```

## Форматы данных

### Входные данные

```json
{
    "texts": [
        "First document text",
        "Second document text"
    ],
    "metadata": {
        "source": "source_name",
        "date": "2025-02-06"
    }
}
```

### Выходные данные

```json
{
    "predictions": [
        {
            "keywords": [
                {
                    "keyword": "document",
                    "score": 0.85,
                    "trend_score": 0.65
                }
            ],
            "processing_time": 0.15
        }
    ],
    "metadata": {
        "model_version": "1.0.0",
        "timestamp": "2025-02-06T10:00:00"
    }
}
```

## Мониторинг

### PerformanceMonitor

```python
class PerformanceMonitor:
    """
    Методы:
        start_batch() -> float
        
        end_batch(
            start_time: float,
            batch_size: int
        ) -> Dict[str, float]
        
        export_performance_report(
            output_file: Optional[Union[str, Path]] = None
        ) -> Dict[str, Any]
    """
```

## Обработка ошибок

Модуль использует следующие типы исключений:

```python
class KeywordExtractorError(Exception):
    """Базовое исключение"""
    pass

class ModelConfigError(KeywordExtractorError):
    """Ошибка конфигурации"""
    pass

class DataProcessingError(KeywordExtractorError):
    """Ошибка обработки данных"""
    pass

class PredictionError(KeywordExtractorError):
    """Ошибка при предсказании"""
    pass
```

## Расширение функциональности

### Добавление новых компонентов

1. Создайте новый класс, наследующий базовые интерфейсы
2. Реализуйте необходимые методы
3. Зарегистрируйте компонент в фабрике

Пример:

```python
from model.model.base import BaseComponent

class CustomComponent(BaseComponent):
    def __init__(self, config):
        super().__init__(config)
        
    def process(self, inputs):
        # Реализация
        pass
        
# Регистрация
COMPONENTS.register("custom", CustomComponent)
```

### Поддержка новых моделей

Для добавления поддержки новой базовой модели:

1. Добавьте конфигурацию
2. Реализуйте адаптер
3. Обновите фабрику моделей

## Интеграция

### REST API

Модуль можно интегрировать через REST API:

```python
from fastapi import FastAPI
from model.api import create_app

app = create_app()

@app.post("/extract_keywords")
async def extract_keywords(text: str):
    keywords = model.extract_keywords([text])
    return {"keywords": keywords}
```

### Batch Processing

Для пакетной обработки используйте:

```python
from model.batch import BatchProcessor

processor = BatchProcessor(
    model_path="path/to/model",
    batch_size=32,
    num_workers=4
)

results = processor.process_directory("path/to/data")
