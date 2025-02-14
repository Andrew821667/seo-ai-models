
from pydantic import BaseModel
from typing import List

class RankPredictorConfig(BaseModel):
    input_size: int = 50
    hidden_size: int = 128
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    ranking_factors: List[str] = [
        'keyword_density',
        'content_length',
        'readability_score',
        'meta_tags_score',
        'header_structure_score'
    ]

class SEOAdvisorConfig(BaseModel):
    # Параметры контента
    min_content_length: int = 1000
    min_keyword_density: float = 0.01
    max_keyword_density: float = 0.03
    
    # Параметры ранжирования
    rank_predictor: RankPredictorConfig = RankPredictorConfig()
    min_confidence_score: float = 0.7
    
    # Параметры рекомендаций
    max_suggestions: int = 10
    priority_threshold_high: float = 0.8
    priority_threshold_medium: float = 0.5
