from dataclasses import dataclass

@dataclass
class DimReducerConfig:
    """Конфигурация модели DimensionReducer"""
    input_dim: int = 768
    hidden_dim: int = 512
    latent_dim: int = 256
    num_encoder_layers: int = 2
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "leaky_relu"
    num_attention_heads: int = 8
