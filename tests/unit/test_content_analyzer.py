import pytest
from seo_ai_models.models.seo_advisor.analyzers.content_analyzer import ContentAnalyzer

def test_content_analyzer_initialization():
    analyzer = ContentAnalyzer()
    assert analyzer.initialized == True

def test_content_analysis():
    analyzer = ContentAnalyzer()
    content = "This is a test content for SEO analysis"
    result = analyzer.analyze(content)
    
    assert isinstance(result, dict)
    assert "readability_score" in result
    assert "keyword_density" in result
    assert "content_length" in result
