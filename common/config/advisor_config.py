from pydantic import BaseModel

class AdvisorConfig(BaseModel):
    model_path: str
    cache_dir: str
    max_length: int = 512
    batch_size: int = 32
