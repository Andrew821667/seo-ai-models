"""
Тестовый скрипт для компонентов SERP-анализа LLM-поисковиков.

Скрипт проверяет основные классы и методы пакета serp_analysis.
"""

import unittest
import logging
from unittest.mock import MagicMock, patch

# Настраиваем логирование
logging.basicConfig(level=logging.ERROR)

# Патчим зависимости
@patch('seo_ai_models.models.llm_integration.serp_analysis.llm_serp_analyzer.CitabilityScorer')
@patch('seo_ai_models.models.llm_integration.serp_analysis.citation_analyzer.CitabilityScorer')
class TestSerpAnalysis(unittest.TestCase):
    """Тесты для компонентов SERP-анализа."""
    
    def setUp(self):
        """Настройка теста."""
        # Создаем моки для базовых сервисов
        self.llm_service = MagicMock()
        self.prompt_generator = MagicMock()
        
        # Настраиваем поведение llm_service
        self.llm_service.query = MagicMock(return_value={
            "text": '{"answer": "This is a test answer", "sources": ["source1", "source2"], "sections": [{"title": "Section 1", "content": "Content 1"}]}',
            "provider": "openai",
            "model": "gpt-4",
            "tokens": {"total": 100},
            "cost": 0.1
        })
        
        # Создаем мок для CitabilityScorer
        self.citability_scorer = MagicMock()
        self.citability_scorer.score_citability = MagicMock(return_value={
            "citability_score": 7,
            "factor_scores": {"Информативность": 8, "Уникальность": 6},
            "citability_analysis": "Хороший анализ",
            "factor_analysis": {"Информативность": "Анализ информативности"},
            "suggested_improvements": {"Информативность": ["Улучшение 1"]},
            "tokens": {"total": 150},
            "cost": 0.15
        })
    
    def test_llm_serp_analyzer(self, mock_citability_scorer1, mock_citability_scorer2):
        """Тест LLMSerpAnalyzer."""
        from seo_ai_models.models.llm_integration.serp_analysis.llm_serp_analyzer import LLMSerpAnalyzer
        
        # Настраиваем мок для CitabilityScorer
        mock_citability_scorer1.return_value = self.citability_scorer
        
        # Создаем экземпляр LLMSerpAnalyzer
        serp_analyzer = LLMSerpAnalyzer(self.llm_service, self.prompt_generator)
        
        # Проверяем инициализацию
        self.assertEqual(serp_analyzer.llm_service, self.llm_service)
        self.assertEqual(serp_analyzer.prompt_generator, self.prompt_generator)
        
        # Тестируем метод check_content_citation
        engine_data = {
            "answer": "This is a test answer containing test content phrase.",
            "sources": ["source with some keywords from test content"]
        }
        content = "Test content phrase with specific keywords."
        
        # Патчим _check_content_citation чтобы он возвращал True
        with patch.object(serp_analyzer, '_check_content_citation', return_value=True):
            with patch.object(serp_analyzer, '_evaluate_relevance', return_value=0.7):
                with patch.object(serp_analyzer, '_determine_citation_position', return_value={"is_cited": True}):
                    # Вызываем метод analyze_serp
                    result = serp_analyzer.analyze_serp(
                        query="test query",
                        content="test content",
                        llm_engines=["perplexity"],
                        num_samples=1,
                        budget=10.0
                    )
                    
                    # Проверяем результат
                    self.assertEqual(result["query"], "test query")
                    self.assertIn("perplexity", result["engines"])
                    self.assertEqual(result["samples"], 1)
                    self.assertIn("citation_rate", result)
                    self.assertIn("visibility_score", result)
                    self.assertIn("engines_results", result)
    
    def test_citation_analyzer(self, mock_citability_scorer1, mock_citability_scorer2):
        """Тест CitationAnalyzer."""
        from seo_ai_models.models.llm_integration.serp_analysis.citation_analyzer import CitationAnalyzer
        
        # Настраиваем мок для CitabilityScorer
        mock_citability_scorer2.return_value = self.citability_scorer
        
        # Мокаем LLMSerpAnalyzer
        with patch('seo_ai_models.models.llm_integration.serp_analysis.citation_analyzer.LLMSerpAnalyzer') as mock_serp_analyzer_class:
            mock_serp_analyzer = MagicMock()
            mock_serp_analyzer_class.return_value = mock_serp_analyzer
            
            # Настраиваем поведение mock_serp_analyzer.analyze_serp
            mock_serp_analyzer.analyze_serp.return_value = {
                "query": "test query",
                "citation_rate": 0.5,
                "visibility_score": 0.7,
                "engines_results": {},
                "tokens": {"total": 100},
                "cost": 0.1
            }
            
            # Создаем экземпляр CitationAnalyzer
            citation_analyzer = CitationAnalyzer(self.llm_service, self.prompt_generator)
            
            # Проверяем инициализацию
            self.assertEqual(citation_analyzer.llm_service, self.llm_service)
            self.assertEqual(citation_analyzer.prompt_generator, self.prompt_generator)
            
            # Патчим _analyze_citation_factors
            with patch.object(citation_analyzer, '_analyze_citation_factors', return_value={}):
                with patch.object(citation_analyzer, '_save_citation_data'):
                    # Вызываем метод analyze_citation
                    result = citation_analyzer.analyze_citation(
                        content="test content",
                        queries=["query1", "query2"],
                        llm_engines=["perplexity"],
                        num_samples=1,
                        budget=10.0
                    )
                    
                    # Проверяем результат
                    self.assertEqual(result["queries_count"], 2)
                    self.assertIn("citation_rate", result)
                    self.assertIn("visibility_score", result)
                    self.assertIn("citation_factors", result)
                    self.assertIn("queries_results", result)
                    self.assertIn("tokens", result)
                    self.assertIn("cost", result)
                    self.assertIn("timestamp", result)
    
    def test_competitor_tracking(self, mock_citability_scorer1, mock_citability_scorer2):
        """Тест CompetitorTracking."""
        from seo_ai_models.models.llm_integration.serp_analysis.competitor_tracking import CompetitorTracking
        
        # Мокаем LLMSerpAnalyzer
        with patch('seo_ai_models.models.llm_integration.serp_analysis.competitor_tracking.LLMSerpAnalyzer') as mock_serp_analyzer_class:
            mock_serp_analyzer = MagicMock()
            mock_serp_analyzer_class.return_value = mock_serp_analyzer
            
            # Настраиваем поведение mock_serp_analyzer.analyze_serp
            mock_serp_analyzer.analyze_serp.return_value = {
                "query": "test query",
                "citation_rate": 0.5,
                "visibility_score": 0.7,
                "engines_results": {},
                "engines": ["perplexity"],
                "tokens": {"total": 100},
                "cost": 0.1
            }
            
            # Создаем экземпляр CompetitorTracking
            competitor_tracking = CompetitorTracking(self.llm_service, self.prompt_generator)
            
            # Проверяем инициализацию
            self.assertEqual(competitor_tracking.llm_service, self.llm_service)
            self.assertEqual(competitor_tracking.prompt_generator, self.prompt_generator)
            
            # Патчим _analyze_competitor_strategies и _save_competitor_data
            with patch.object(competitor_tracking, '_analyze_competitor_strategies', return_value={"top_competitor": {}, "recommendations": []}):
                with patch.object(competitor_tracking, '_save_competitor_data'):
                    with patch.object(competitor_tracking, '_rank_competitors', return_value=[]):
                        # Подготавливаем данные о конкурентах
                        competitors = [
                            {"id": "comp1", "name": "Competitor 1", "content": "competitor 1 content"},
                            {"id": "comp2", "name": "Competitor 2", "content": "competitor 2 content"}
                        ]
                        
                        # Вызываем метод track_competitors
                        result = competitor_tracking.track_competitors(
                            query="test query",
                            competitors=competitors,
                            our_content="our content",
                            llm_engines=["perplexity"],
                            num_samples=1,
                            budget=10.0
                        )
                        
                        # Проверяем результат
                        self.assertEqual(result["query"], "test query")
                        self.assertEqual(result["competitors_count"], 2)
                        self.assertTrue(result["has_our_content"])
                        self.assertIn("ranking", result)
                        self.assertIn("competitor_strategies", result)
                        self.assertIn("competitors_results", result)
                        self.assertIn("our_result", result)
                        self.assertIn("tokens", result)
                        self.assertIn("cost", result)
                        self.assertIn("timestamp", result)
    
    def test_industry_benchmarker(self, mock_citability_scorer1, mock_citability_scorer2):
        """Тест IndustryBenchmarker."""
        from seo_ai_models.models.llm_integration.serp_analysis.industry_benchmarker import IndustryBenchmarker
        
        # Мокаем LLMSerpAnalyzer
        with patch('seo_ai_models.models.llm_integration.serp_analysis.industry_benchmarker.LLMSerpAnalyzer') as mock_serp_analyzer_class:
            mock_serp_analyzer = MagicMock()
            mock_serp_analyzer_class.return_value = mock_serp_analyzer
            
            # Настраиваем поведение mock_serp_analyzer.analyze_serp
            mock_serp_analyzer.analyze_serp.return_value = {
                "query": "test query",
                "citation_rate": 0.5,
                "visibility_score": 0.7,
                "engines_results": {},
                "engines": ["perplexity"],
                "tokens": {"total": 100},
                "cost": 0.1
            }
            
            # Создаем экземпляр IndustryBenchmarker
            industry_benchmarker = IndustryBenchmarker(self.llm_service, self.prompt_generator)
            
            # Проверяем инициализацию
            self.assertEqual(industry_benchmarker.llm_service, self.llm_service)
            self.assertEqual(industry_benchmarker.prompt_generator, self.prompt_generator)
            
            # Тестируем метод get_industry_benchmarks
            result = industry_benchmarker.get_industry_benchmarks("technology", "how_to_guide")
            
            # Проверяем результат
            self.assertEqual(result["industry"], "technology")
            self.assertEqual(result["content_type"], "how_to_guide")
            self.assertIn("benchmarks", result)
            self.assertIn("last_updated", result)
            
            # Патчим методы для analyze_industry_benchmarks
            with patch.object(industry_benchmarker, '_analyze_content_metrics', return_value={}):
                with patch.object(industry_benchmarker, '_generate_benchmark_analysis', return_value={}):
                    with patch.object(industry_benchmarker, '_generate_benchmark_recommendations', return_value=[]):
                        # Вызываем метод analyze_industry_benchmarks
                        result = industry_benchmarker.analyze_industry_benchmarks(
                            content="test content",
                            queries=["query1", "query2"],
                            industry="technology",
                            content_type="how_to_guide",
                            llm_engines=["perplexity"],
                            num_samples=1,
                            budget=10.0
                        )
                        
                        # Проверяем результат
                        self.assertEqual(result["industry"], "technology")
                        self.assertEqual(result["content_type"], "how_to_guide")
                        self.assertEqual(result["queries_count"], 2)
                        self.assertIn("citation_rate", result)
                        self.assertIn("visibility_score", result)
                        self.assertIn("benchmarks", result)

if __name__ == '__main__':
    unittest.main()
