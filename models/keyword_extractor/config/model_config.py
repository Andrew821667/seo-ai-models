# models/keyword_extractor/config/model_config.py

from pydantic import BaseModel
from typing import Optional

class KeywordModelConfig(BaseModel):
    """Конфигурация модели извлечения ключевых слов"""
    model_name: str = "xlm-roberta-base"
    max_length: int = 512
    input_dim: int = 768
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    num_heads: int = 8
    num_layers: int = 2
    use_batch_norm: bool = True

class KeywordTrainingConfig(BaseModel):
    """Конфигурация процесса обучения"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    validation_split: float = 0.2
    early_stopping_patience: int = 3

class KeywordInferenceConfig(BaseModel):
    """Конфигурация инференса"""
    threshold: float = 0.5
    max_keywords: int = 20
    min_keyword_length: int = 3
    use_cache: bool = True
    cache_ttl: int = 3600

class KeywordExtractorConfig(BaseModel):
    """Основная конфигурация модели"""
    model: KeywordModelConfig = KeywordModelConfig()
    training: KeywordTrainingConfig = KeywordTrainingConfig()
    inference: KeywordInferenceConfig = KeywordInferenceConfig()
    device: str = "cuda"
    debug_mode: bool = False
    logging_level: str = "INFO"
