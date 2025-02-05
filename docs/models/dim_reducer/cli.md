# DimensionReducer CLI Guide

## Installation

```bash
# From project root
pip install -e .
```

## Basic Commands

### Training

Train a new model:
```bash
python -m models.dim_reducer.cli train \
    configs/dim_reducer.yml \
    data/train_data.csv \
    --val-data data/val_data.csv \
    --output-dir models/checkpoints \
    --device cuda
```

#### Options:
- `config_path`: Путь к файлу конфигурации (YAML)
- `train_data`: Путь к обучающим данным
- `--val-data`: Путь к валидационным данным (опционально)
- `--output-dir`: Директория для сохранения модели
- `--device`: Устройство для обучения (cuda/cpu)

### Prediction

Применение обученной модели:
```bash
python -m models.dim_reducer.cli predict \
    models/checkpoints/final_model.pt \
    data/test_data.csv \
    --config-path configs/dim_reducer.yml \
    --output-dir predictions \
    --batch-size 32
```

#### Options:
- `model_path`: Путь к обученной модели
- `input_data`: Путь к входным данным
- `--config-path`: Путь к конфигурации (опционально)
- `--output-dir`: Директория для результатов
- `--batch-size`: Размер батча

### Text Analysis

Анализ отдельного текста:
```bash
python -m models.dim_reducer.cli analyze-text \
    models/checkpoints/final_model.pt \
    "Your text for analysis" \
    --config-path configs/dim_reducer.yml
```

## Configuration

### Example Configuration (dim_reducer.yml)

```yaml
# Model Architecture
input_dim: 768
hidden_dim: 512
latent_dim: 256
num_attention_heads: 8
dropout_rate: 0.2

# Training
batch_size: 32
num_epochs: 10
learning_rate: 1e-4
weight_decay: 0.01

# Data Processing
max_features: 100
max_length: 512
model_name: 'bert-base-uncased'
```

## Data Format

### Training Data CSV Format
```csv
text,html,label
"Sample text 1","<html>...</html>",1
"Sample text 2","<html>...</html>",0
```

### Prediction Input Format
```csv
text,html
"Test text 1","<html>...</html>"
"Test text 2","<html>...</html>"
```

## Output Format

### Prediction Output (JSON)
```json
{
    "latent_features": [...],
    "feature_importance": [...],
    "reconstruction_error": 0.023
}
```

## Examples

### Training with Custom Config
```bash
# Create config
cat > config.yml << EOF
input_dim: 512
latent_dim: 128
batch_size: 64
num_epochs: 20
EOF

# Train model
python -m models.dim_reducer.cli train config.yml data/train.csv
```

### Batch Processing
```bash
# Process multiple files
for file in data/*.csv; do
    python -m models.dim_reducer.cli predict \
        model.pt "$file" \
        --output-dir "predictions/$(basename "$file" .csv)"
done
```

## Troubleshooting

### Common Issues

1. CUDA Out of Memory
```bash
# Reduce batch size
python -m models.dim_reducer.cli predict \
    model.pt data.csv \
    --batch-size 16
```

2. CPU Performance
```bash
# Increase number of workers
export NUM_WORKERS=4
python -m models.dim_reducer.cli predict ...
```
