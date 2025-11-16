import pytest
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer
from seo_ai_models.models.seo_advisor.predictors.rank_predictor import RankPredictor

class TestSEOAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return ContentAnalyzer()
        
    @pytest.fixture
    def predictor(self):
        return RankPredictor()
    
    def test_content_analysis(self, analyzer):
        content = "Test content for SEO analysis"
        result = analyzer.analyze(content)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['readability_score', 'keyword_density', 'content_length'])
        
    def test_rank_prediction(self, predictor):
        analysis = {
            'readability_score': 0.8,
            'keyword_density': 0.02,
            'content_length': 1000
        }
        score = predictor.predict(analysis)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
