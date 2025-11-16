
import pytest
from seo_ai_models.models.seo_advisor.suggester.suggester import Suggester

@pytest.fixture
def default_suggester():
    return Suggester()

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

def test_suggester_initialization():
    """Test initialization with different industries"""
    industries = ['default', 'blog', 'scientific_blog', 'ecommerce']
    for industry in industries:
        suggester = Suggester(industry=industry)
        assert suggester.industry == industry
        assert suggester.rank_predictor is not None

def test_analyze_content(default_suggester, test_features):
    """Test content analysis functionality"""
    analysis = default_suggester.analyze_content(test_features)
    
    # Check analysis structure
    assert isinstance(analysis, dict)
    assert all(key in analysis for key in [
        'current_position',
        'score_analysis',
        'priority_tasks',
        'base_recommendations',
        'competitor_insights'
    ])
    
    # Check position range
    assert 1 <= analysis['current_position'] <= 100
    
    # Check detailed analysis
    assert isinstance(analysis['score_analysis'], dict)
    for feature, details in analysis['score_analysis'].items():
        assert all(key in details for key in [
            'current_value',
            'score',
            'weighted_score',
            'impact_percentage',
            'status',
            'thresholds'
        ])
        assert 0 <= details['score'] <= 1
        assert 0 <= details['impact_percentage'] <= 100

def test_priority_tasks(default_suggester, test_features):
    """Test priority tasks generation"""
    analysis = default_suggester.analyze_content(test_features)
    tasks = analysis['priority_tasks']
    
    assert isinstance(tasks, list)
    for task in tasks:
        assert all(key in task for key in [
            'feature',
            'title',
            'description',
            'priority',
            'impact'
        ])
        assert task['priority'] in ['high', 'medium', 'low']
        assert 0 <= task['impact'] <= 100

def test_competitor_insights():
    """Test competitor insights for different industries"""
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
    
    for industry in ['default', 'blog', 'scientific_blog', 'ecommerce']:
        suggester = Suggester(industry=industry)
        analysis = suggester.analyze_content(test_features)
        insights = analysis['competitor_insights']
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)

def test_feature_status(default_suggester):
    """Test feature status determination"""
    # Test content length status
    thresholds = {'low': 650, 'high': 2400, 'optimal_min': 950, 'optimal_max': 2600}
    
    assert default_suggester._get_feature_status('content_length', 500, thresholds) == 'critical'
    assert default_suggester._get_feature_status('content_length', 1500, thresholds) == 'optimal'
    assert default_suggester._get_feature_status('content_length', 3000, thresholds) == 'excessive'
    
    # Test other features status
    thresholds = {'low': 0.01, 'high': 0.03}
    assert default_suggester._get_feature_status('keyword_density', 0.005, thresholds) == 'below_threshold'
    assert default_suggester._get_feature_status('keyword_density', 0.02, thresholds) == 'optimal'
    assert default_suggester._get_feature_status('keyword_density', 0.04, thresholds) == 'above_threshold'
