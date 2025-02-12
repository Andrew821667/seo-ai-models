from typing import Optional
from .....config.base_config import BaseConfig

class KeywordModelConfig(BaseConfig):
    """Конфигурация модели для извлечения ключевых слов"""
    # Базовые параметры
    model_name: str = "bert-base-uncased"
    input_dim: int = 768
    hidden_dim: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.1
    max_length: int = 512
    num_labels: int = 2
    
    # Дополнительные параметры
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 час
    
    # Параметры обучения
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 0.01
    
    class Config:
        """Настройки Pydantic модели"""
        arbitrary_types_allowed = True
