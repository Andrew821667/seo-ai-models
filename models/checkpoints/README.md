# Создаём README для модели
cat > models/checkpoints/README.md << 'EOL'
# Dimension Reducer Model

## Характеристики модели
- Входная размерность: 768 (BERT)
- Выходная размерность: 128
- Корреляция: 0.95
- Early Stopping: сработал на 18 эпохе

## Метрики
- Train Loss: 0.7798
- Val Loss: 0.9707

## Улучшения
- Добавлен BatchNorm
- Добавлен Dropout
- Реализован Early Stopping

## История обучения
- Количество эпох: 28
- Хранится история обучения и валидации

## Использование модели

```python
from models.dim_reducer.model import ImprovedDimReducer

# Загрузка модели
model = ImprovedDimReducer(input_dim=768, output_dim=128)
model.load_model('models/checkpoints/dim_reducer_20250212_201812.pt')

# Использование
encoded = model.encode(input_data)
```
EOL

# Добавляем README в Git
!git add models/checkpoints/README.md
!git commit -m "Add model documentation"
!git push origin main
