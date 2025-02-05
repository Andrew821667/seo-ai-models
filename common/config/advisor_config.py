from typing import Optional
from .base_config import BaseConfig

class ModelConfig(BaseConfig):
    """Конфигурация нейронной сети"""
    content_dim: int = 768
    metrics_dim: int = 6
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    model_name: str = 'bert-base-multilingual-uncased'

class TrainingConfig(BaseConfig):
    """Конфигурация обучения"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

class InferenceConfig(BaseConfig):
    """Конфигурация инференса"""
    max_length: int = 512
    num_suggestions: int = 5
    confidence_threshold: float = 0.5
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 час

class CacheConfig(BaseConfig):
    """Конфигурация кэширования"""
    redis_url: str = 'redis://localhost:6379'
    prefix: str = 'seo_advisor:'
    default_ttl: int = 3600

class MonitoringConfig(BaseConfig):
    """Конфигурация мониторинга"""
    enable_metrics: bool = True
    prometheus_port: int = 9090
    log_performance: bool = True
    alert_threshold: float = 0.9

class AdvisorConfig(BaseConfig):
    """Основная конфигурация SEO Advisor"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    cache: CacheConfig = CacheConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Дополнительные параметры
    debug_mode: bool = False
    log_file: Optional[str] = None
    metrics_file: Optional[str] = None
