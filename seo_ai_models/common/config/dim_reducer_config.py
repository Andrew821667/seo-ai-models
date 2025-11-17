from typing import Optional
from .base_config import BaseConfig


class DimReducerConfig(BaseConfig):
    """Конфигурация модели сжатия размерности"""

    # Размерности
    input_dim: int = 768
    hidden_dim: int = 512
    latent_dim: int = 256

    # Параметры архитектуры
    num_encoder_layers: int = 2
    num_attention_heads: int = 8
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "leaky_relu"

    # Параметры обучения
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 0.01

    # Параметры предобработки
    max_length: int = 512
    model_name: str = "bert-base-uncased"

    class Config:
        """Настройки Pydantic модели"""

        arbitrary_types_allowed = True
