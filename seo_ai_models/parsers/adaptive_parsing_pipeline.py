"""
Адаптивный парсинг-конвейер для проекта SEO AI Models.
Автоматически определяет тип сайта (обычный или SPA) и использует соответствующий краулер.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union

from seo_ai_models.parsers.crawlers.web_crawler import WebCrawler
from seo_ai_models.parsers.crawlers.spa_crawler import SPACrawler
from seo_ai_models.parsers.extractors.content_extractor import ContentExtractor
from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.utils.request_utils import fetch_url, fetch_url_with_javascript_sync
from seo_ai_models.parsers.utils.spa_detector import SPADetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveParsingPipeline:
    """
    Адаптивный конвейер парсинга, определяющий тип сайта и выбирающий соответствующий метод парсинга.
    """
    
    def __init__(
        self,
        user_agent: str = "SEOAIModels AdaptiveParsingPipeline/1.0",
        respect_robots: bool = True,
        delay: float = 1.0,
        max_pages: int = 100,
        wait_for_idle: int = 2000,  # мс
        wait_for_timeout: int = 10000,  # мс
        headless: bool = True,
        browser_type: str = "chromium",
        force_spa_mode: bool = False,
    ):
        """
        Инициализация AdaptiveParsingPipeline.

        Args:
            user_agent: User-Agent для запросов
            respect_robots: Уважать ли robots.txt
            delay: Задержка между запросами в секундах
            max_pages: Максимальное количество страниц для сканирования
            wait_for_idle: Время ожидания в мс после событий 'networkidle'
            wait_for_timeout: Максимальное время ожидания в мс
            headless: Запускать ли браузер в режиме headless для SPA
            browser_type: Тип браузера ('chromium', 'firefox', 'webkit')
            force_spa_mode: Всегда использовать режим SPA независимо от детектирования
        """
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.delay = delay
        self.max_pages = max_pages
        self.wait_for_idle = wait_for_idle
        self.wait_for_timeout = wait_for_timeout
        self.headless = headless
        self.browser_type = browser_type
        self.force_spa_mode = force_spa_mode
        
        # Инициализация различных компонентов
        self.spa_detector = SPADetector()
        self.meta_extractor = MetaExtractor()
        
        logger.info("Adaptive Parsing Pipeline initialized")
        
    def detect_site_type(self, url: str) -> Dict[str, Any]:
        """
        Определяет тип сайта (обычный или SPA).
        
        Args:
            url: URL для проверки
            
        Returns:
            Dict[str, Any]: Информация о типе сайта
        """
        if self.force_spa_mode:
            logger.info(f"Forced SPA mode for {url}")
            return {
                "is_spa": True,
                "confidence": 1.0,
                "detection_method": "forced"
            }
            
        logger.info(f"Detecting site type for {url}")
        
        # Сначала пробуем обычный запрос
        html_content, _, error = fetch_url(url)
        
        if error or not html_content:
            logger.warning(f"Could not fetch {url} with standard method: {error}")
            
            # Если обычный запрос не сработал, попробуем с JavaScript рендерингом
            html_content, _, error = fetch_url_with_javascript_sync(
                url, 
                headless=self.headless,
                wait_for_idle=self.wait_for_idle,
                wait_for_timeout=self.wait_for_timeout
            )
            
            if error or not html_content:
                logger.error(f"Could not fetch {url} with any method: {error}")
                return {
                    "is_spa": False,
                    "confidence": 0.0,
                    "error": error
                }
                
            # Если JavaScript рендеринг сработал, а обычный запрос нет, скорее всего это SPA
            return {
                "is_spa": True,
                "confidence": 0.9,
                "detection_method": "request_comparison"
            }
            
        # Анализируем HTML на признаки SPA
        spa_analysis = self.spa_detector.analyze_html(html_content)
        
        # Если обнаружены явные признаки SPA, проверяем разницу между обычным и JavaScript рендерингом
        if spa_analysis["is_spa"]:
            # Получаем контент с JavaScript рендерингом
            js_html_content, _, js_error = fetch_url_with_javascript_sync(
                url, 
                headless=self.headless,
                wait_for_idle=self.wait_for_idle,
                wait_for_timeout=self.wait_for_timeout
            )
            
            if js_error or not js_html_content:
                logger.warning(f"Could not fetch {url} with JavaScript rendering: {js_error}")
                
                # Если JavaScript рендеринг не сработал, но обнаружены признаки SPA,
                # все равно считаем сайт SPA, но с меньшей уверенностью
                spa_analysis["confidence"] *= 0.8
                
            else:
                # Сравниваем длину контента
                standard_length = len(html_content)
                js_length = len(js_html_content)
                
                # Если JavaScript рендеринг дал значительно больше контента, подтверждаем SPA
                if js_length > standard_length * 1.2:  # На 20% больше контента
                    spa_analysis["confidence"] = min(1.0, spa_analysis["confidence"] + 0.2)
                    spa_analysis["content_difference"] = {
                        "standard_length": standard_length,
                        "js_length": js_length,
                        "difference_percent": (js_length - standard_length) / standard_length * 100
                    }
                    
        spa_analysis["detection_method"] = "html_analysis"
        return spa_analysis
        
    def analyze_url(self, url: str, detect_type: bool = True) -> Dict[str, Any]:
        """
        Анализирует URL и возвращает результаты.
        
        Args:
            url: URL для анализа
            detect_type: Определять ли тип сайта автоматически
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        result = {
            "url": url,
            "success": False,
            "content": None,
            "metadata": None,
            "error": None
        }
        
        try:
            # Определение типа сайта
            if detect_type:
                site_type = self.detect_site_type(url)
                result["site_type"] = site_type
                is_spa = site_type.get("is_spa", False)
            else:
                is_spa = self.force_spa_mode
                result["site_type"] = {"is_spa": is_spa, "detection_method": "predefined"}
                
            logger.info(f"Analyzing URL {url} as {'SPA' if is_spa else 'standard'} site")
                
            # Выбор соответствующего экстрактора
            if is_spa:
                content_extractor = SPAContentExtractor(
                    wait_for_idle=self.wait_for_idle,
                    wait_for_timeout=self.wait_for_timeout,
                    headless=self.headless,
                    browser_type=self.browser_type
                )
                result["content"] = content_extractor.extract_content_from_url(url)
            else:
                content_extractor = ContentExtractor()
                html_content, _, error = fetch_url(url)
                
                if error or not html_content:
                    result["error"] = f"Failed to fetch content: {error}"
                    return result
                    
                result["content"] = content_extractor.extract_content(html_content, url)
                
            # Извлечение метаданных
            meta_result = self.meta_extractor.extract_from_url(url)
            result["metadata"] = meta_result
            
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {str(e)}")
            result["error"] = str(e)
            
        return result
        
    def crawl_site(self, base_url: str, detect_type: bool = True, **crawler_options) -> Dict[str, Any]:
        """
        Сканирует сайт и возвращает результаты.
        
        Args:
            base_url: Начальный URL для сканирования
            detect_type: Определять ли тип сайта автоматически
            **crawler_options: Дополнительные опции для краулера
            
        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        result = {
            "base_url": base_url,
            "success": False,
            "crawled_urls": [],
            "content": {},
            "metadata": {},
            "error": None
        }
        
        try:
            # Определение типа сайта
            if detect_type:
                site_type = self.detect_site_type(base_url)
                result["site_type"] = site_type
                is_spa = site_type.get("is_spa", False)
            else:
                is_spa = self.force_spa_mode
                result["site_type"] = {"is_spa": is_spa, "detection_method": "predefined"}
                
            logger.info(f"Crawling site {base_url} as {'SPA' if is_spa else 'standard'} site")
            
            # Установка опций краулера
            crawler_options = {
                "base_url": base_url,
                "max_pages": crawler_options.get("max_pages", self.max_pages),
                "delay": crawler_options.get("delay", self.delay),
                "respect_robots": crawler_options.get("respect_robots", self.respect_robots),
                "user_agent": crawler_options.get("user_agent", self.user_agent),
            }
            
            # Выбор соответствующего краулера
            if is_spa:
                # Добавляем опции для SPA краулера
                spa_options = {
                    "wait_for_idle": crawler_options.get("wait_for_idle", self.wait_for_idle),
                    "wait_for_timeout": crawler_options.get("wait_for_timeout", self.wait_for_timeout),
                    "headless": crawler_options.get("headless", self.headless),
                    "browser_type": crawler_options.get("browser_type", self.browser_type),
                }
                crawler_options.update(spa_options)
                
                crawler = SPACrawler(**crawler_options)
                crawl_result = crawler.run_crawler()
            else:
                crawler = WebCrawler(**crawler_options)
                crawl_result = crawler.crawl()
                
            result["crawled_urls"] = crawl_result["crawled_urls"]
            result["found_urls"] = crawl_result["found_urls"]
            result["failed_urls"] = crawl_result["failed_urls"]
            
            # Анализ каждого URL из результатов краулинга
            for url in result["crawled_urls"]:
                try:
                    url_analysis = self.analyze_url(url, detect_type=False)
                    
                    if url_analysis["success"]:
                        result["content"][url] = url_analysis["content"]
                        result["metadata"][url] = url_analysis["metadata"]
                        
                except Exception as e:
                    logger.error(f"Error analyzing crawled URL {url}: {str(e)}")
                    
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error crawling site {base_url}: {str(e)}")
            result["error"] = str(e)
            
        return result
        
    def analyze_multiple_urls(self, urls: List[str], detect_type: bool = True) -> Dict[str, Any]:
        """
        Анализирует несколько URL и возвращает результаты.
        
        Args:
            urls: Список URL для анализа
            detect_type: Определять ли тип сайта автоматически
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        results = {}
        
        for url in urls:
            try:
                logger.info(f"Analyzing URL: {url}")
                url_result = self.analyze_url(url, detect_type)
                results[url] = url_result
                
            except Exception as e:
                logger.error(f"Error analyzing URL {url}: {str(e)}")
                results[url] = {
                    "url": url,
                    "success": False,
                    "error": str(e)
                }
                
        return results
