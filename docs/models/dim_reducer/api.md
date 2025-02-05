# DimensionReducer API Documentation

## REST API Endpoints

### Analyze Text
`POST /api/v1/dim-reducer/analyze`

Анализ текста с извлечением SEO-характеристик и их сжатием.

#### Request
```json
{
    "text": "Your text content here",
    "html": "<html>Optional HTML markup</html>",
    "return_importance": true
}
```

#### Response
```json
{
    "latent_features": [0.1, 0.2, ...],
    "feature_importance": [0.8, 0.5, ...],
    "reconstruction_error": 0.023
}
```

### Batch Analysis
`POST /api/v1/dim-reducer/batch`

Пакетный анализ нескольких текстов.

#### Request
```json
{
    "texts": [
        "First text content",
        "Second text content"
    ],
    "html_texts": [
        "<html>First HTML</html>",
        "<html>Second HTML</html>"
    ]
}
```

#### Response
```json
[
    {
        "latent_features": [...],
        "feature_importance": [...],
        "reconstruction_error": 0.021
    },
    {
        "latent_features": [...],
        "feature_importance": [...],
        "reconstruction_error": 0.019
    }
]
```

### Model Info
`GET /api/v1/dim-reducer/info`

Получение информации о модели.

#### Response
```json
{
    "config": {
        "input_dim": 768,
        "latent_dim": 256,
        ...
    },
    "device": "cuda",
    "input_dim": 768,
    "latent_dim": 256
}
```

## Python API

### Basic Usage

```python
from models.dim_reducer import DimensionReducer, DimReducerConfig

# Initialize
config = DimReducerConfig()
model = DimensionReducer(config)

# Process text
results = model.process_text("Your text here")
```

### Batch Processing

```python
from models.dim_reducer.inference import DimReducerInference

# Initialize inference
inference = DimReducerInference('path/to/model.pt', config)

# Batch process
results = inference.batch_process([
    "First text",
    "Second text"
])
```

### Training

```python
from models.dim_reducer.trainer import DimReducerTrainer
from models.dim_reducer.data_loader import SEODataLoader

# Setup data
data_loader = SEODataLoader(batch_size=32)
train_dataset, val_dataset = data_loader.create_datasets(
    train_path='train.csv',
    val_path='val.csv'
)

# Train model
trainer = DimReducerTrainer(model, config)
trainer.train_epoch(train_loader, epoch=0, val_loader=val_loader)
```

## Configuration

### Model Configuration
```python
config = DimReducerConfig(
    input_dim=768,
    hidden_dim=512,
    latent_dim=256,
    num_attention_heads=8,
    dropout_rate=0.2
)
```

### Training Configuration
```python
config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10,
    weight_decay=0.01
)
```

## Error Handling

### HTTP Status Codes
- 200: Успешное выполнение
- 400: Некорректный запрос
- 422: Ошибка валидации данных
- 500: Внутренняя ошибка сервера

### Error Response Format
```json
{
    "detail": "Error message description"
}
```
