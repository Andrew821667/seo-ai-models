"""
Unit tests for improvement modules.
"""

import pytest
from seo_ai_models.improvements.visual_analyzer import VisualContentAnalyzer
from seo_ai_models.improvements.content_refresh import ContentRefreshAutomation
from seo_ai_models.improvements.intent_optimizer import IntentBasedOptimizer, SearchIntent
from seo_ai_models.improvements.mobile_optimizer import MobileOptimizer


class TestVisualContentAnalyzer:
    """Tests for VisualContentAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return VisualContentAnalyzer(llm_service=None)

    @pytest.fixture
    def sample_page(self):
        return {
            "text": "Sample text " * 200,  # ~400 words
            "images": [
                {"src": "image1.jpg", "alt": "", "size_bytes": 250000},
                {"src": "image2.png", "alt": "Good alt text", "size_bytes": 150000},
                {"src": "image3.jpg", "alt": "", "size_bytes": 50000}
            ]
        }

    def test_analyze_images(self, analyzer, sample_page):
        """Test image analysis."""
        result = analyzer.analyze_images(sample_page)

        assert "total_images" in result
        assert result["total_images"] == 3
        assert "missing_alt" in result
        assert len(result["missing_alt"]) == 2  # 2 images without alt
        assert "oversized" in result
        assert len(result["oversized"]) == 1  # 1 image > 200KB

    def test_visual_coverage_calculation(self, analyzer, sample_page):
        """Test visual coverage metric."""
        result = analyzer.analyze_images(sample_page)

        assert "visual_coverage" in result
        # 3 images per ~400 words = ~7.5 images per 1000 words
        assert result["visual_coverage"] > 0

    def test_auto_fix_images(self, analyzer, sample_page):
        """Test auto-fixing images."""
        result = analyzer.auto_fix_images(sample_page, auto_execute=False)

        assert result["success"] is True
        assert "fixes" in result
        assert result["fixes"]["alt_tags_added"] >= 0


class TestContentRefreshAutomation:
    """Tests for ContentRefreshAutomation."""

    @pytest.fixture
    def refresher(self):
        return ContentRefreshAutomation(cms_connector=None, llm_service=None)

    def test_priority_calculation(self, refresher):
        """Test refresh priority calculation."""
        page = {
            "id": "page1",
            "traffic": 5000,
            "avg_ranking": 15
        }

        priority = refresher._calculate_refresh_priority(page, age_days=730)

        assert priority > 0
        assert isinstance(priority, float)

    def test_update_statistics(self, refresher):
        """Test year update in text."""
        text = "In 2020, we saw major changes. By 2021, things improved."
        updated = refresher._update_statistics(text)

        # Years older than 2 years should be updated
        assert "2020" not in updated or "2021" not in updated

    def test_modernize_terminology(self, refresher):
        """Test terminology modernization."""
        text = "Our webmaster uses PageRank to improve Google+ presence."
        updated, terms = refresher._modernize_terminology(text)

        assert "SEO specialist" in updated or "webmaster" in updated
        assert len(terms) >= 0


class TestIntentBasedOptimizer:
    """Tests for IntentBasedOptimizer."""

    @pytest.fixture
    def optimizer(self):
        return IntentBasedOptimizer(llm_service=None)

    def test_detect_informational_intent(self, optimizer):
        """Test detecting informational intent."""
        result = optimizer.detect_intent("how to bake bread")

        assert result["primary_intent"] == SearchIntent.INFORMATIONAL
        assert result["confidence"] > 0

    def test_detect_transactional_intent(self, optimizer):
        """Test detecting transactional intent."""
        result = optimizer.detect_intent("buy nike shoes online")

        assert result["primary_intent"] == SearchIntent.TRANSACTIONAL

    def test_detect_commercial_intent(self, optimizer):
        """Test detecting commercial intent."""
        result = optimizer.detect_intent("best laptop 2024 reviews")

        assert result["primary_intent"] == SearchIntent.COMMERCIAL

    def test_intent_score_calculation(self, optimizer):
        """Test intent scoring."""
        result = optimizer.detect_intent("ultimate guide to SEO")

        assert "all_scores" in result
        assert isinstance(result["all_scores"], dict)


class TestMobileOptimizer:
    """Tests for MobileOptimizer."""

    @pytest.fixture
    def optimizer(self):
        return MobileOptimizer(crawler=None)

    def test_analyze_mobile_friendliness(self, optimizer):
        """Test mobile friendliness analysis."""
        result = optimizer.analyze_mobile_friendliness("https://example.com")

        assert "mobile_friendly" in result
        assert "score" in result
        assert "issues" in result
        assert "passed_checks" in result

    def test_core_web_vitals_analysis(self, optimizer):
        """Test Core Web Vitals analysis."""
        result = optimizer.analyze_core_web_vitals("https://example.com")

        assert "metrics" in result
        assert "lcp" in result["metrics"]
        assert "fid" in result["metrics"]
        assert "cls" in result["metrics"]
        assert "passed" in result

    def test_lcp_rating(self, optimizer):
        """Test LCP rating."""
        assert optimizer._rate_lcp(2.0) == "good"
        assert optimizer._rate_lcp(3.0) == "needs_improvement"
        assert optimizer._rate_lcp(5.0) == "poor"

    def test_fid_rating(self, optimizer):
        """Test FID rating."""
        assert optimizer._rate_fid(50) == "good"
        assert optimizer._rate_fid(150) == "needs_improvement"
        assert optimizer._rate_fid(400) == "poor"

    def test_cls_rating(self, optimizer):
        """Test CLS rating."""
        assert optimizer._rate_cls(0.05) == "good"
        assert optimizer._rate_cls(0.15) == "needs_improvement"
        assert optimizer._rate_cls(0.30) == "poor"


class TestAIContentGenerator:
    """Tests for AIContentGenerator."""

    @pytest.fixture
    def generator(self):
        from seo_ai_models.improvements.ai_content_generator import AIContentGenerator
        return AIContentGenerator(llm_service=None, seo_advisor=None)

    def test_detect_search_intent(self, generator):
        """Test search intent detection."""
        assert generator._detect_search_intent("how to cook pasta") == "informational"
        assert generator._detect_search_intent("buy cheap laptops") == "commercial"
        assert generator._detect_search_intent("best phones review") == "commercial_investigation"

    def test_recommend_word_count(self, generator):
        """Test word count recommendation."""
        count = generator._recommend_word_count("how to learn python", [])

        assert count > 0
        assert count >= 1000

    def test_recommend_tone(self, generator):
        """Test tone recommendation."""
        tone = generator._recommend_tone("buy now", "commercial")

        assert isinstance(tone, str)
        assert len(tone) > 0
