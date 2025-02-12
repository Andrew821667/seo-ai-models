from dataclasses import dataclass

@dataclass
class KeywordModelConfig:
    """Конфигурация модели для извлечения ключевых слов"""
    model_name: str = "bert-base-uncased"
    input_dim: int = 768
    hidden_dim: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.1
    max_length: int = 512
    num_labels: int = 2
