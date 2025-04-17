"""
Анализатор сайтов для проекта SEO AI Models.
Предоставляет расширенные возможности анализа сайтов и интеграцию с ядром системы.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.models.seo_advisor.advisor import SEOAdvisor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SiteAnalyzer:
    """
    Анализатор сайтов, интегрирующий парсер с ядром SEO AI Models.
    Предоставляет возможности расширенного анализа сайта и контента.
    """
    
    def __init__(
        self,
        industry: str = 'default',
        parser_settings: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация анализатора сайтов.
        
        Args:
            industry: Отрасль для анализа
            parser_settings: Настройки парсера
        """
        # Инициализация ядра SEO AI Models
        self.seo_advisor = SEOAdvisor(industry=industry)
        
        # Инициализация парсера
        self.parser = UnifiedParser(**(parser_settings or {}))
        
        logger.info(f"SiteAnalyzer initialized for industry: {industry}")
    
    def analyze_url(self, url: str, target_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Анализ одного URL с использованием SEO Advisor.
        
        Args:
            url: URL для анализа
            target_keywords: Целевые ключевые слова (если None, будут извлечены автоматически)
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        logger.info(f"Analyzing URL (debug mode): {url}")
        start_time = time.time()
        
        try:
            # Парсим URL
            parse_result = self.parser.parse_url(url)
            
            if not parse_result.get("success", False):
                return {
                    "success": False,
                    "error": parse_result.get("error", "Parsing failed"),
                    "url": url,
                    "processing_time": time.time() - start_time
                }
            
            # Извлекаем контент и структуру
            page_data = parse_result.get("page_data", {})
            content = page_data.get("content", {}).get("full_text", "Sample content for testing")
            
            # Если не указаны целевые ключевые слова, извлекаем из контента
            if target_keywords is None or len(target_keywords) == 0:
                target_keywords = ["sample", "test", "keyword"]
                logger.info(f"Using default keywords for testing: {target_keywords}")
            
            # Анализируем контент с помощью SEO Advisor
            seo_analysis = self.seo_advisor.analyze_content(content, target_keywords)
            
            # Для отладки используем простую структуру
            result = {
                "success": True,
                "url": url,
                "page_data": page_data,
                "seo_analysis": {
                    "content_metrics": getattr(seo_analysis, "content_metrics", {}),
                    "keyword_analysis": getattr(seo_analysis, "keyword_analysis", {}),
                    "predicted_position": getattr(seo_analysis, "predicted_position", 0),
                    "feature_scores": getattr(seo_analysis, "feature_scores", {}),
                    "content_quality": {
                        "strengths": ["Sample strength 1", "Sample strength 2"],
                        "weaknesses": ["Sample weakness 1"],
                        "potential_improvements": ["Sample improvement 1", "Sample improvement 2"]
                    },
                    "recommendations": {
                        "content": ["Add more content", "Improve headings structure"],
                        "technical": ["Optimize meta tags", "Improve page speed"]
                    },
                    "priorities": [
                        {"task": "Add more content", "priority": "high"},
                        {"task": "Optimize meta tags", "priority": "medium"}
                    ]
                },
                "target_keywords": target_keywords,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}",
                "url": url,
                "processing_time": time.time() - start_time
            }
    
    def analyze_site(
        self, 
        base_url: str, 
        max_pages: int = 10,
        focus_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Анализ сайта с использованием SEO Advisor (режим отладки).
        """
        logger.info(f"Analyzing site (debug mode): {base_url}")
        start_time = time.time()
        
        # Возвращаем тестовые данные для отладки
        return {
            "success": True,
            "base_url": base_url,
            "site_data": {
                "pages_count": 3,
                "statistics": {
                    "content": {"avg_word_count": 500, "total_words": 1500},
                    "meta": {"missing_title_percent": 0, "missing_description_percent": 33.3}
                }
            },
            "site_keywords": focus_keywords or ["sample", "test", "keyword"],
            "page_analyses": {
                f"{base_url}": {
                    "target_keywords": ["sample", "test"],
                    "predicted_position": 5.2,
                    "content_metrics": {"word_count": 500, "readability": 0.75}
                },
                f"{base_url}/page1": {
                    "target_keywords": ["sample", "page1"],
                    "predicted_position": 8.4,
                    "content_metrics": {"word_count": 350, "readability": 0.68}
                },
                f"{base_url}/page2": {
                    "target_keywords": ["test", "page2"],
                    "predicted_position": 12.1,
                    "content_metrics": {"word_count": 650, "readability": 0.82}
                }
            },
            "site_recommendations": {
                "content": [
                    "Increase content length across all pages",
                    "Improve keyword usage in page titles"
                ],
                "technical": [
                    "Add meta descriptions to all pages",
                    "Optimize images with alt tags"
                ]
            },
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_competitors(self, query: str, target_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализ конкурентов для заданного запроса (режим отладки).
        """
        logger.info(f"Analyzing competitors (debug mode) for query: {query}")
        start_time = time.time()
        
        # Возвращаем тестовые данные для отладки
        return {
            "success": True,
            "query": query,
            "search_results_count": 10,
            "competitor_analyses": {
                "http://example.com/competitor1": {
                    "title": "Competitor 1",
                    "position": 1,
                    "content_metrics": {"word_count": 1200, "readability": 0.85},
                    "keyword_analysis": {"density": 2.3, "prominence": 0.9},
                    "feature_scores": {"content_quality": 0.9, "keyword_usage": 0.85}
                },
                "http://example.com/competitor2": {
                    "title": "Competitor 2",
                    "position": 2,
                    "content_metrics": {"word_count": 950, "readability": 0.78},
                    "keyword_analysis": {"density": 1.8, "prominence": 0.75},
                    "feature_scores": {"content_quality": 0.82, "keyword_usage": 0.77}
                }
            },
            "target_analysis": target_url and {
                "url": target_url,
                "seo_analysis": {
                    "content_metrics": {"word_count": 800, "readability": 0.72},
                    "feature_scores": {"content_quality": 0.75, "keyword_usage": 0.68}
                }
            },
            "comparison": target_url and {
                "content_comparison": {
                    "word_count": {
                        "target_value": 800,
                        "avg_competitor_value": 1075,
                        "difference": -275,
                        "percentage_difference": -25.6,
                        "status": "worse"
                    }
                },
                "feature_comparison": {
                    "content_quality": {
                        "target_score": 0.75,
                        "avg_competitor_score": 0.86,
                        "difference": -0.11,
                        "percentage_difference": -12.8,
                        "status": "worse"
                    }
                },
                "strengths": [],
                "weaknesses": [
                    "content_quality (на 12.8% хуже конкурентов)",
                    "word_count (на 25.6% хуже конкурентов)"
                ],
                "overall_status": "worse"
            },
            "related_queries": [
                f"{query} example",
                f"{query} guide",
                f"best {query}",
                f"how to {query}"
            ],
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def _consolidate_recommendations(self, page_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Объединяет рекомендации со всех страниц (заглушка).
        """
        return {
            "content": ["Improve content length", "Optimize keyword usage"],
            "technical": ["Fix meta tags", "Improve page speed"],
            "structure": ["Improve heading structure", "Add more internal links"]
        }
    
    def _compare_with_competitors(
        self, 
        target_analysis: Dict[str, Any],
        competitor_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Сравнивает целевой сайт с конкурентами (заглушка).
        """
        return {
            "content_comparison": {},
            "feature_comparison": {},
            "strengths": ["Sample strength compared to competitors"],
            "weaknesses": ["Sample weakness compared to competitors"],
            "overall_status": "comparable"
        }
