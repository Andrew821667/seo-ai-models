from pydantic import BaseModel

class ModelConfig(BaseModel):
    # Основные параметры модели
    content_dim: int = 768  # Размерность выхода BERT
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    device: str = "cpu"
    
    class Config:
        arbitrary_types_allowed = True
