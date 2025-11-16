import pytest
from datetime import datetime
from seo_ai_models.models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor, IndustryThresholds, IndustryWeights

@pytest.fixture
def default_predictor():
    return ImprovedRankPredictor()

@pytest.fixture
def test_features():
    return {
        'keyword_density': 0.018,
        'content_length': 1800,
        'readability_score': 55,
        'meta_tags_score': 0.85,
        'header_structure_score': 0.8,
        'backlinks_count': 65,
        'multimedia_score': 0.6,
        'internal_linking_score': 0.7
    }

def test_predictor_initialization():
    """Test initialization with different industries"""
    industries = ['default', 'blog', 'scientific_blog', 'ecommerce']
    for industry in industries:
        predictor = ImprovedRankPredictor(industry=industry)
        assert predictor.industry == industry
        assert isinstance(predictor.thresholds, IndustryThresholds)
        assert isinstance(predictor.weights, IndustryWeights)
        assert predictor.history == []

def test_predict_position(default_predictor, test_features):
    """Test position prediction functionality"""
    prediction = default_predictor.predict_position(test_features)
    
    # Check prediction structure
    assert isinstance(prediction, dict)
    assert all(key in prediction for key in ['position', 'total_score', 'feature_scores', 'weighted_scores'])
    
    # Check value ranges
    assert 1 <= prediction['position'] <= 100
    assert 0 <= prediction['total_score'] <= 1
    
    # Check history update
    assert len(default_predictor.history) == 1
    assert isinstance(default_predictor.history[0]['timestamp'], datetime)

def test_generate_recommendations(default_predictor, test_features):
    """Test recommendations generation"""
    recommendations = default_predictor.generate_recommendations(test_features)
    
    assert isinstance(recommendations, dict)
    assert len(recommendations) <= 3  # Проверяем, что выдается не более 3 рекомендаций
    
    for feature, recs in recommendations.items():
        assert isinstance(recs, list)
        assert len(recs) > 0
        assert all(isinstance(rec, str) for rec in recs)

def test_industry_specific_predictions():
    """Test predictions for different industries"""
    test_features = {
        'keyword_density': 0.018,
        'content_length': 1800,
        'readability_score': 55,
        'meta_tags_score': 0.85,
        'header_structure_score': 0.8,
        'backlinks_count': 65,
        'multimedia_score': 0.6,
        'internal_linking_score': 0.7
    }
    
    results = {}
    for industry in ['default', 'blog', 'scientific_blog', 'ecommerce']:
        predictor = ImprovedRankPredictor(industry=industry)
        prediction = predictor.predict_position(test_features)
        results[industry] = prediction['position']
    
    # Проверяем, что разные индустрии дают разные результаты
    positions = list(results.values())
    assert len(set(positions)) > 1  # Должны быть разные значения для разных индустрий

def test_history_analytics(default_predictor, test_features):
    """Test history analytics functionality"""
    # Сделаем несколько предсказаний
    for _ in range(3):
        default_predictor.predict_position(test_features)
    
    analytics = default_predictor.get_history_analytics()
    
    assert isinstance(analytics, dict)
    assert all(key in analytics for key in [
        'avg_position', 'min_position', 'max_position', 
        'avg_score', 'predictions_count'
    ])
    assert analytics['predictions_count'] == 3
