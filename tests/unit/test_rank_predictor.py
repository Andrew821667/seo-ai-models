import pytest
from seo_ai_models.models.seo_advisor.predictors.rank_predictor import RankPredictor

def test_rank_predictor_initialization():
    predictor = RankPredictor()
    assert predictor.initialized == True

def test_rank_prediction():
    predictor = RankPredictor()
    content_analysis = {
        "readability_score": 0.8,
        "keyword_density": 0.02,
        "content_length": 1000
    }
    score = predictor.predict(content_analysis)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
