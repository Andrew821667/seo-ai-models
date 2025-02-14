
from pydantic import BaseModel

class SEOAdvisorConfig(BaseModel):
    # Параметры контента
    min_content_length: int = 1000
    min_keyword_density: float = 0.01
    max_keyword_density: float = 0.03
    
    # Параметры ранжирования
    ranking_factors_count: int = 50
    min_confidence_score: float = 0.7
    
    # Параметры рекомендаций
    max_suggestions: int = 10
    priority_threshold_high: float = 0.8
    priority_threshold_medium: float = 0.5
