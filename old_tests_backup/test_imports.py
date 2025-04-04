def test_imports():
    try:
        # Импорты из common
        from common.config.advisor_config import AdvisorConfig
        
        # Импорты из models
        from models.dim_reducer.model import DimensionReducer
        from models.keyword_extractor.model.processor import KeywordProcessor
        
        # Импорты из seo_advisor
        from seo_advisor.config.advisor_config import AdvisorConfig as SeoAdvisorConfig
        from seo_advisor.core.models.content_analyzer import ContentAnalyzer
        from seo_advisor.core.models.rank_predictor import RankPredictor
        
        assert True, "Импорты работают"
    except ImportError as e:
        assert False, f"Ошибка импорта: {e}"
