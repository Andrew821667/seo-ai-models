
from pydantic import BaseModel
from typing import Optional

class AdvisorConfig(BaseModel):
    """Configuration for SEO Advisor"""
    model_path: str
    cache_dir: Optional[str] = None
    max_length: int = 512
    batch_size: int = 32
    device: str = "cpu"

class ModelConfig(BaseModel):
    """Configuration for SEO Models"""
    content_dim: int = 768
    hidden_dim: int = 512
    dropout: float = 0.1
    num_heads: int = 8
    max_length: int = 512
    model_name: str = 'bert-base-unchecked'
