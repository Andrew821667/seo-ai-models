from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Конфигурация SEO Advisor"""
    input_dim: int = 768
    hidden_dim: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.1
    num_suggestions: int = 5
