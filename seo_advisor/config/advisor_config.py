from pydantic import BaseModel

class AdvisorConfig(BaseModel):
    """Configuration for SEO Advisor"""
    model_path: str
    cache_dir: str | None = None
    max_length: int = 512
    batch_size: int = 32
    device: str = "cpu"
