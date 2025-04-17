"""
Усовершенствованный унифицированный парсер для проекта SEO AI Models.
Объединяет все новые компоненты и обеспечивает расширенную функциональность:
- Поддержка SPA-сайтов через Playwright
- Многопоточный параллельный парсинг
- Интеграция с API поисковых систем
- Расширенный семантический анализ с NLP
"""

import logging
import time
import re
import json
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse, urljoin
from datetime import datetime
import traceback

from seo_ai_models.parsers.unified.parser_interface import ParserInterface
from seo_ai_models.parsers.unified.parser_result import (
    ParserResult, PageData, SiteData, SearchResultData,
    TextContent, PageStructure, MetaData
)
from seo_ai_models.common.utils.enhanced_text_processor import EnhancedTextProcessor

# Импортируем компоненты парсинга
from seo_ai_models.parsers.unified.crawlers.web_crawler import WebCrawler
from seo_ai_models.parsers.unified.crawlers.enhanced_spa_crawler import EnhancedSPACrawler
from seo_ai_models.parsers.unified.extractors.content_extractor import ContentExtractor
from seo_ai_models.parsers.unified.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.unified.analyzers.search_api_integration import SearchAPIIntegration
from seo_ai_models.parsers.unified.analyzers.enhanced_semantic_analyzer import EnhancedSemanticAnalyzer
from seo_ai_models.parsers.unified.utils.parallel_parsing import ParallelParser
from seo_ai_models.parsers.unified.utils.request_utils import fetch_url, fetch_url_with_javascript_sync

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedParser(ParserInterface):
    """
    Унифицированный парсер, объединяющий все возможности парсинга в единый интерфейс.
    Предоставляет расширенные возможности для анализа сайтов любого типа.
    """
    
    def __init__(
        self,
        user_agent: str = "SEOAIModels UnifiedParser/2.0",
        respect_robots: bool = True,
        delay: float = 1.0,
        max_pages: int = 100,
        search_engine: str = "google",
        spa_settings: Optional[Dict[str, Any]] = None,
        auto_detect_spa: bool = True,
        force_spa_mode: bool = False,
        extract_readability: bool = True,
        extract_semantic: bool = True,
        parallel_parsing: bool = True,
        max_workers: int = 5,
        search_api_keys: Optional[List[str]] = None
    ):
        """
        Инициализация унифицированного парсера.
        
        Args:
            user_agent: User-Agent для запросов
            respect_robots: Уважать ли robots.txt
            delay: Задержка между запросами в секундах
            max_pages: Максимальное количество страниц для сканирования
            search_engine: Поисковая система для использования
            spa_settings: Настройки для SPA-парсера
            auto_detect_spa: Автоматически определять SPA-сайты
            force_spa_mode: Всегда использовать режим SPA
            extract_readability: Извлекать метрики читаемости
            extract_semantic: Извлекать семантические метрики
            parallel_parsing: Использовать многопоточный парсинг
            max_workers: Максимальное количество потоков для параллельного парсинга
            search_api_keys: API-ключи для поисковых систем
        """
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.delay = delay
        self.max_pages = max_pages
        self.search_engine = search_engine
        self.auto_detect_spa = auto_detect_spa
        self.force_spa_mode = force_spa_mode
        self.extract_readability = extract_readability
        self.extract_semantic = extract_semantic
        self.parallel_parsing = parallel_parsing
        self.max_workers = max_workers
        
        # Настройки SPA по умолчанию
        self.spa_settings = spa_settings or {
            "wait_for_idle": 2000,  # мс
            "wait_for_timeout": 10000,  # мс
            "headless": True,
            "browser_type": "chromium",
            "intercept_ajax": True
        }
        
        # Инициализация компонентов
        try:
            # Инициализация текстового процессора
            self.text_processor = EnhancedTextProcessor()
            
            # Базовые экстракторы
            self.content_extractor = ContentExtractor()
            self.meta_extractor = MetaExtractor()
            
            # Продвинутые компоненты
            if extract_semantic:
                self.semantic_analyzer = EnhancedSemanticAnalyzer()
            
            # Поисковый API
            self.search_api = SearchAPIIntegration(
                api_keys=search_api_keys,
                api_provider="serpapi" if search_api_keys else "custom"
            )
            
            # Параллельный парсер
            if self.parallel_parsing:
                self.parallel_parser = ParallelParser(
                    max_workers=self.max_workers,
                    rate_limit=self.delay,
                    respect_robots=self.respect_robots
                )
                
            logger.info("UnifiedParser initialized with all components")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            logger.warning("UnifiedParser will operate in limited mode")
    
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
            
        if not self.auto_detect_spa:
            logger.info(f"Auto-detection disabled, using standard mode for {url}")
            return {
                "is_spa": False,
                "confidence": 1.0,
                "detection_method": "predefined"
            }
            
        logger.info(f"Detecting site type for {url}")
        
        try:
            # Сначала пробуем обычный запрос
            html_content, status_code, error = fetch_url(url)
            
            if error or not html_content:
                logger.warning(f"Could not fetch {url} with standard method: {error}")
                
                # Если обычный запрос не сработал, попробуем с JavaScript рендерингом
                html_content, status_code, error = fetch_url_with_javascript_sync(
                    url, 
                    headless=self.spa_settings.get("headless", True),
                    wait_for_idle=self.spa_settings.get("wait_for_idle", 2000),
                    wait_for_timeout=self.spa_settings.get("wait_for_timeout", 10000)
                )
                
                if error or not html_content:
                    logger.error(f"Could not fetch {url} with any method: {error}")
                    return {
                        "is_spa": False,
                        "confidence": 0.0,
                        "error": str(error)
                    }
                    
                # Если JavaScript рендеринг сработал, а обычный запрос нет, скорее всего это SPA
                return {
                    "is_spa": True,
                    "confidence": 0.9,
                    "detection_method": "request_comparison"
                }
                
            # Анализируем HTML на признаки SPA
            # Проверяем наличие признаков SPA
            spa_indicators = [
                'ng-app',
                'ng-controller',
                'data-reactroot',
                'react-app',
                'vue-app',
                'nuxt',
                'ember-app',
                'backbone',
                '<script src="[^"]*angular\.js"></script>',
                '<script src="[^"]*react\.js"></script>',
                '<script src="[^"]*vue\.js"></script>'
            ]
            
            spa_score = 0
            for indicator in spa_indicators:
                if re.search(indicator, html_content, re.IGNORECASE):
                    spa_score += 1
                    
            confidence = min(1.0, spa_score / (len(spa_indicators) / 2))
            is_spa = confidence > 0.3
            
            return {
                "is_spa": is_spa,
                "confidence": confidence,
                "detection_method": "html_analysis"
            }
            
        except Exception as e:
            logger.error(f"Error detecting site type for {url}: {str(e)}")
            return {
                "is_spa": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def parse_url(self, url: str, **options) -> Dict[str, Any]:
        """
        Парсинг одного URL, извлечение контента и метаданных.
        
        Args:
            url: URL для парсинга
            **options: Дополнительные опции парсинга
            
        Returns:
            Dict[str, Any]: Результат парсинга в формате, совместимом с ядром
        """
        logger.info(f"Parsing URL: {url}")
        start_time = time.time()
        
        try:
            # Определяем, нужно ли анализировать как SPA
            force_spa = options.get("force_spa", self.force_spa_mode)
            auto_detect = options.get("auto_detect_spa", self.auto_detect_spa)
            
            if force_spa:
                is_spa = True
                site_type_info = {"is_spa": True, "detection_method": "forced"}
            elif auto_detect:
                site_type_info = self.detect_site_type(url)
                is_spa = site_type_info.get("is_spa", False)
            else:
                is_spa = False
                site_type_info = {"is_spa": False, "detection_method": "predefined"}
                
            # Извлечение контента и метаданных
            if is_spa:
                logger.info(f"Parsing {url} as SPA")
                # Используем SPA подход
                html_content, status_code, error = fetch_url_with_javascript_sync(
                    url, 
                    headless=self.spa_settings.get("headless", True),
                    wait_for_idle=self.spa_settings.get("wait_for_idle", 2000),
                    wait_for_timeout=self.spa_settings.get("wait_for_timeout", 10000),
                    user_agent=self.user_agent
                )
            else:
                logger.info(f"Parsing {url} as standard site")
                # Используем стандартный подход
                html_content, status_code, error = fetch_url(url)
                
            if not html_content or status_code != 200:
                processing_time = time.time() - start_time
                return ParserResult.create_error(
                    f"Failed to fetch URL: {error}",
                    processing_time
                ).to_dict()
                
            # Извлекаем контент и метаданные
            content_data = self.content_extractor.extract_content(html_content, url)
            meta_data = self.meta_extractor.extract_meta_information(html_content, url)
            
            # Создаем структурированный результат
            page_data = self._create_page_data(url, html_content, content_data, meta_data)
            
            # Добавляем информацию о типе сайта
            page_data.html_stats["site_type"] = site_type_info
            
            # Добавляем семантический анализ, если требуется
            if self.extract_semantic and hasattr(self, 'semantic_analyzer'):
                text = page_data.content.full_text
                semantic_analysis = self.semantic_analyzer.analyze_text(text)
                
                # Добавляем результаты семантического анализа в performance
                page_data.performance["semantic_analysis"] = {
                    "semantic_density": semantic_analysis.get("semantic_density", 0),
                    "semantic_coverage": semantic_analysis.get("semantic_coverage", 0),
                    "topical_coherence": semantic_analysis.get("topical_coherence", 0),
                    "contextual_relevance": semantic_analysis.get("contextual_relevance", 0),
                    "keywords": semantic_analysis.get("keywords", {})
                }
            
            # Рассчитываем время обработки
            processing_time = time.time() - start_time
            
            # Создаем и возвращаем результат
            return ParserResult.create_page_result(page_data, processing_time).to_dict()
            
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {str(e)}")
            traceback.print_exc()
            processing_time = time.time() - start_time
            return ParserResult.create_error(
                f"Error parsing URL: {str(e)}",
                processing_time
            ).to_dict()
    
    def parse_html(self, html: str, base_url: Optional[str] = None, **options) -> Dict[str, Any]:
        """
        Парсинг HTML-контента.
        
        Args:
            html: HTML-контент для парсинга
            base_url: Базовый URL для относительных ссылок
            **options: Дополнительные опции парсинга
            
        Returns:
            Dict[str, Any]: Результат парсинга в формате, совместимом с ядром
        """
        logger.info(f"Parsing HTML content{' for ' + base_url if base_url else ''}")
        start_time = time.time()
        
        try:
            # Используем базовый URL или заглушку
            url = base_url or "http://example.com"
            
            # Извлечение контента и метаданных
            content_data = self.content_extractor.extract_content(html, url)
            meta_data = self.meta_extractor.extract_meta_information(html, url)
            
            # Создаем структурированный результат
            page_data = self._create_page_data(url, html, content_data, meta_data)
            
            # Добавляем семантический анализ, если требуется
            if self.extract_semantic and hasattr(self, 'semantic_analyzer'):
                text = page_data.content.full_text
                semantic_analysis = self.semantic_analyzer.analyze_text(text)
                
                # Добавляем результаты семантического анализа
                page_data.performance["semantic_analysis"] = {
                    "semantic_density": semantic_analysis.get("semantic_density", 0),
                    "semantic_coverage": semantic_analysis.get("semantic_coverage", 0),
                    "topical_coherence": semantic_analysis.get("topical_coherence", 0),
                    "contextual_relevance": semantic_analysis.get("contextual_relevance", 0),
                    "keywords": semantic_analysis.get("keywords", {})
                }
            
            # Рассчитываем время обработки
            processing_time = time.time() - start_time
            
            # Создаем и возвращаем результат
            return ParserResult.create_page_result(page_data, processing_time).to_dict()
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            traceback.print_exc()
            processing_time = time.time() - start_time
            return ParserResult.create_error(
                f"Error parsing HTML: {str(e)}",
                processing_time
            ).to_dict()
    
    def crawl_site(self, base_url: str, **options) -> Dict[str, Any]:
        """
        Сканирование сайта и парсинг его страниц.
        
        Args:
            base_url: Начальный URL для сканирования
            **options: Дополнительные опции сканирования
            
        Returns:
            Dict[str, Any]: Результаты сканирования и парсинга
        """
        logger.info(f"Crawling site: {base_url}")
        start_time = time.time()
        
        try:
            # Извлекаем параметры
            max_pages = options.get("max_pages", self.max_pages)
            delay = options.get("delay", self.delay)
            respect_robots = options.get("respect_robots", self.respect_robots)
            user_agent = options.get("user_agent", self.user_agent)
            use_parallel = options.get("parallel_parsing", self.parallel_parsing)
            
            # Определяем тип сайта
            force_spa = options.get("force_spa", self.force_spa_mode)
            auto_detect = options.get("auto_detect_spa", self.auto_detect_spa)
            
            if force_spa:
                is_spa = True
                site_type_info = {"is_spa": True, "detection_method": "forced"}
            elif auto_detect:
                site_type_info = self.detect_site_type(base_url)
                is_spa = site_type_info.get("is_spa", False)
            else:
                is_spa = False
                site_type_info = {"is_spa": False, "detection_method": "predefined"}
                
            # Выбираем подходящий краулер
            if is_spa:
                logger.info(f"Crawling {base_url} as SPA")
                crawler = EnhancedSPACrawler(
                    base_url=base_url,
                    max_pages=max_pages,
                    delay=delay,
                    respect_robots=respect_robots,
                    user_agent=user_agent,
                    headless=self.spa_settings.get("headless", True),
                    wait_for_idle=self.spa_settings.get("wait_for_idle", 2000),
                    wait_for_timeout=self.spa_settings.get("wait_for_timeout", 10000),
                    browser_type=self.spa_settings.get("browser_type", "chromium")
                )
            else:
                logger.info(f"Crawling {base_url} as standard site")
                crawler = WebCrawler(
                    base_url=base_url,
                    max_pages=max_pages,
                    delay=delay,
                    respect_robots=respect_robots,
                    user_agent=user_agent
                )
                
            # Выполняем сканирование
            crawl_result = crawler.crawl()
                
            # Создаем структуру данных сайта
            site_data = SiteData(
                base_url=base_url,
                crawl_info={
                    "start_time": start_time,
                    "end_time": time.time(),
                    "urls_found": len(crawl_result.get("found_urls", [])),
                    "urls_crawled": len(crawl_result.get("crawled_urls", [])),
                    "urls_failed": len(crawl_result.get("failed_urls", [])),
                    "site_type": site_type_info
                }
            )
            
            # Парсим каждый URL (последовательно или параллельно)
            crawled_urls = crawl_result.get("crawled_urls", [])
            
            if use_parallel and hasattr(self, 'parallel_parser'):
                logger.info(f"Using parallel parsing for {len(crawled_urls)} URLs")
                
                # Функция для парсинга одного URL
                def _parse_single_url(url, **kwargs):
                    page_result = self.parse_url(url, force_spa=is_spa, auto_detect=False)
                    return url, page_result
                
                # Параллельный парсинг
                parallel_results = self.parallel_parser.parse_urls(
                    crawled_urls,
                    _parse_single_url
                )
                
                # Обработка результатов
                for url, page_result in parallel_results.get("results", {}).items():
                    self._add_page_to_site_data(site_data, url, page_result)
            else:
                logger.info(f"Using sequential parsing for {len(crawled_urls)} URLs")
                
                # Последовательный парсинг
                for url in crawled_urls:
                    page_result = self.parse_url(url, force_spa=is_spa, auto_detect=False)
                    self._add_page_to_site_data(site_data, url, page_result)
                    
            # Добавляем статистику и структуру сайта
            site_data.statistics = self._calculate_site_statistics(site_data.pages)
            site_data.site_structure = self._build_site_structure(site_data.pages)
            
            # Рассчитываем время обработки
            processing_time = time.time() - start_time
            
            # Создаем и возвращаем результат
            return ParserResult.create_site_result(site_data, processing_time).to_dict()
            
        except Exception as e:
            logger.error(f"Error crawling site {base_url}: {str(e)}")
            traceback.print_exc()
            processing_time = time.time() - start_time
            return ParserResult.create_error(
                f"Error crawling site: {str(e)}",
                processing_time
            ).to_dict()
    
    def parse_search_results(self, query: str, **options) -> Dict[str, Any]:
        """
        Парсинг результатов поиска по запросу.
        
        Args:
            query: Поисковый запрос
            **options: Дополнительные опции парсинга
            
        Returns:
            Dict[str, Any]: Результаты парсинга поисковой выдачи
        """
        logger.info(f"Parsing search results for query: {query}")
        start_time = time.time()
        
        try:
            # Извлекаем параметры
            search_engine = options.get("search_engine", self.search_engine)
            results_count = options.get("results_count", 10)
            analyze_content = options.get("analyze_content", True)
            
            # Используем поисковый API, если доступен
            if hasattr(self, 'search_api'):
                logger.info(f"Using search API for query: {query}")
                # Дополнительные параметры для поиска
                search_params = {
                    "country": options.get("country", "us"),
                    "language": options.get("language", "en"),
                    "results_count": results_count
                }
                
                # Получаем результаты через API
                api_results = self.search_api.search(query, **search_params)
                
                # Формируем структуру поисковых данных
                search_data = SearchResultData(
                    query=query,
                    results=api_results.get("results", []),
                    related_queries=api_results.get("related_queries", []),
                    search_features=api_results.get("search_features", {}),
                    search_engine=search_engine,
                    timestamp=datetime.now()
                )
            else:
                # Используем базовую имитацию поисковых результатов
                logger.info(f"Using simulated search results for: {query}")
                search_data = self._generate_simulated_search_results(query, results_count)
            
            # Если нужно анализировать контент результатов
            if analyze_content:
                # Ограничиваем количество результатов для анализа
                analysis_limit = min(3, len(search_data.results))
                
                for i in range(analysis_limit):
                    result = search_data.results[i]
                    url = result.get("url")
                    
                    if url:
                        try:
                            # Парсим URL для получения подробной информации
                            page_result = self.parse_url(url)
                            
                            # Добавляем подробную информацию к результату
                            if page_result.get("success", False) and "page_data" in page_result:
                                search_data.results[i]["detailed_analysis"] = page_result["page_data"]
                                
                        except Exception as e:
                            logger.error(f"Error analyzing search result {url}: {str(e)}")
                            continue
            
            # Рассчитываем время обработки
            processing_time = time.time() - start_time
            
            # Создаем и возвращаем результат
            return ParserResult.create_search_result(search_data, processing_time).to_dict()
            
        except Exception as e:
            logger.error(f"Error parsing search results for query {query}: {str(e)}")
            traceback.print_exc()
            processing_time = time.time() - start_time
            return ParserResult.create_error(
                f"Error parsing search results: {str(e)}",
                processing_time
            ).to_dict()
    
    def _create_page_data(self, url: str, html_content: str, content_data: Dict[str, Any], meta_data: Dict[str, Any]) -> PageData:
        """
        Создает структурированные данные страницы из результатов извлечения.
        
        Args:
            url: URL страницы
            html_content: HTML-контент
            content_data: Данные контента от экстрактора
            meta_data: Метаданные от экстрактора
            
        Returns:
            PageData: Структурированные данные страницы
        """
        # Извлекаем текстовый контент
        full_text = content_data.get("content", {}).get("all_text", "")
        paragraphs = content_data.get("paragraphs", [])
        
        # Разбиваем текст на предложения, если их нет
        sentences = content_data.get("content", {}).get("sentences", [])
        if not sentences and full_text:
            sentences = re.split(r'[.!?]+', full_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Подсчитываем слова и символы
        word_count = content_data.get("content", {}).get("word_count", 0)
        if not word_count and full_text:
            word_count = len(full_text.split())
            
        char_count = content_data.get("content", {}).get("char_count", 0)
        if not char_count and full_text:
            char_count = len(full_text)
        
        # Извлекаем ключевые слова
        keywords = {}
        if self.extract_semantic and hasattr(self, 'text_processor'):
            try:
                # Используем текстовый процессор для извлечения ключевых слов
                processed_data = self.text_processor.process_html_content(html_content)
                keywords = processed_data.get("keywords", {})
            except Exception as e:
                logger.warning(f"Error extracting keywords: {str(e)}")
        
        # Определяем язык
        language = None
        if full_text and hasattr(self, 'text_processor'):
            try:
                self.text_processor.detect_language(full_text)
                language = self.text_processor.language
            except Exception as e:
                logger.warning(f"Error detecting language: {str(e)}")
        
        # Создаем объект TextContent
        text_content = TextContent(
            full_text=full_text,
            paragraphs=paragraphs,
            sentences=sentences,
            word_count=word_count,
            char_count=char_count,
            keywords=keywords,
            language=language
        )
        
        # Создаем структуру страницы
        title = content_data.get("title", "")
        headings = content_data.get("headings", {})
        
        # Приведение headings к нужному формату
        formatted_headings = {f"h{i}": [] for i in range(1, 7)}
        for level, texts in headings.items():
            if level.lower().startswith('h') and level[1:].isdigit():
                formatted_headings[level.lower()] = texts
        
        # Извлечение ссылок
        links = meta_data.get("links", {})
        
        # Извлечение изображений
        images = []
        for img in meta_data.get("images", []):
            images.append({
                "src": img.get("src", ""),
                "alt": img.get("alt", ""),
                "width": img.get("width", ""),
                "height": img.get("height", "")
            })
        
        # Создаем объект PageStructure
        page_structure = PageStructure(
            title=title,
            headings=formatted_headings,
            links=links,
            images=images,
            tables=[],  # Пока оставляем пустыми
            lists=content_data.get("lists", [])
        )
        
        # Создаем объект MetaData
        meta_data_obj = MetaData(
            title=meta_data.get("meta_tags", {}).get("title", ""),
            description=meta_data.get("meta_tags", {}).get("description", ""),
            keywords=meta_data.get("meta_tags", {}).get("keywords", ""),
            canonical=meta_data.get("meta_tags", {}).get("canonical", None),
            robots=meta_data.get("meta_tags", {}).get("robots", None),
            og_tags=meta_data.get("meta_tags", {}).get("og", {}),
            twitter_tags=meta_data.get("meta_tags", {}).get("twitter", {}),
            schema_org=meta_data.get("schema_org", []),
            additional_meta=meta_data.get("meta_tags", {}).get("other", {})
        )
        
        # Создаем статистику HTML
        html_stats = {
            "html_size": len(html_content) if html_content else 0,
            "text_ratio": char_count / len(html_content) if html_content and len(html_content) > 0 else 0,
            "headings_count": sum(len(texts) for texts in formatted_headings.values()),
            "links_count": len(links.get("internal", [])) + len(links.get("external", [])),
            "images_count": len(images)
        }
        
        # Добавляем метрики читаемости, если нужно
        performance = {}
        if self.extract_readability and hasattr(self, 'text_processor'):
            try:
                # Используем текстовый процессор для расчета читаемости
                processed_data = self.text_processor.process_html_content(html_content)
                readability = processed_data.get("readability", {})
                performance["readability"] = readability
            except Exception as e:
                logger.warning(f"Error calculating readability: {str(e)}")
        
        # Создаем объект PageData
        page_data = PageData(
            url=url,
            content=text_content,
            structure=page_structure,
            metadata=meta_data_obj,
            html_stats=html_stats,
            performance=performance
        )
        
        return page_data
    
    def _add_page_to_site_data(self, site_data: SiteData, url: str, page_result: Dict[str, Any]) -> None:
        """
        Добавляет данные страницы в данные сайта.
        
        Args:
            site_data: Данные сайта
            url: URL страницы
            page_result: Результат парсинга страницы
        """
        if not page_result.get("success", False) or "page_data" not in page_result:
            return
            
        # Преобразуем из словаря в объект
        page_data_dict = page_result["page_data"]
        
        # Создаем и добавляем объект PageData
        content = TextContent(
            full_text=page_data_dict["content"]["full_text"],
            paragraphs=page_data_dict["content"].get("paragraphs", []),
            sentences=page_data_dict["content"].get("sentences", []),
            word_count=page_data_dict["content"].get("word_count", 0),
            char_count=page_data_dict["content"].get("char_count", 0),
            keywords=page_data_dict["content"].get("keywords", {}),
            language=page_data_dict["content"].get("language")
        )
        
        structure = PageStructure(
            title=page_data_dict["structure"]["title"],
            headings=page_data_dict["structure"].get("headings", {}),
            links=page_data_dict["structure"].get("links", {"internal": [], "external": []}),
            images=page_data_dict["structure"].get("images", []),
            tables=page_data_dict["structure"].get("tables", []),
            lists=page_data_dict["structure"].get("lists", [])
        )
        
        metadata = MetaData(
            title=page_data_dict["metadata"].get("title", ""),
            description=page_data_dict["metadata"].get("description", ""),
            keywords=page_data_dict["metadata"].get("keywords", ""),
            canonical=page_data_dict["metadata"].get("canonical"),
            robots=page_data_dict["metadata"].get("robots"),
            og_tags=page_data_dict["metadata"].get("og_tags", {}),
            twitter_tags=page_data_dict["metadata"].get("twitter_tags", {}),
            schema_org=page_data_dict["metadata"].get("schema_org", []),
            additional_meta=page_data_dict["metadata"].get("additional_meta", {})
        )
        
        page_data = PageData(
            url=page_data_dict["url"],
            content=content,
            structure=structure,
            metadata=metadata,
            html_stats=page_data_dict.get("html_stats", {}),
            performance=page_data_dict.get("performance", {})
        )
        
        site_data.pages[url] = page_data
    
    def _calculate_site_statistics(self, pages: Dict[str, PageData]) -> Dict[str, Any]:
        """
        Рассчитывает статистику для всего сайта на основе данных страниц.
        
        Args:
            pages: Словарь URL -> PageData
            
        Returns:
            Dict[str, Any]: Статистика по сайту
        """
        if not pages:
            return {}
            
        # Базовые подсчеты
        pages_count = len(pages)
        
        # Средние значения метрик
        avg_word_count = sum(page.content.word_count for page in pages.values()) / pages_count
        avg_char_count = sum(page.content.char_count for page in pages.values()) / pages_count
        
        # Статистика по заголовкам
        headings_stats = {f"h{i}_count": 0 for i in range(1, 7)}
        headings_stats["total_headings"] = 0
        
        for page in pages.values():
            for level, texts in page.structure.headings.items():
                if level.startswith('h') and level[1:].isdigit():
                    headings_stats[level + "_count"] = headings_stats.get(level + "_count", 0) + len(texts)
                    headings_stats["total_headings"] += len(texts)
        
        # Средние значения по заголовкам
        for level in [f"h{i}" for i in range(1, 7)]:
            headings_stats[f"avg_{level}_per_page"] = headings_stats.get(f"{level}_count", 0) / pages_count
        
        headings_stats["avg_headings_per_page"] = headings_stats["total_headings"] / pages_count
        
        # Статистика по метатегам
        meta_stats = {
            "missing_title": 0,
            "missing_description": 0,
            "missing_keywords": 0,
            "has_canonical": 0,
            "has_robots": 0,
            "has_og": 0,
            "has_twitter": 0,
            "has_schema": 0
        }
        
        for page in pages.values():
            if not page.metadata.title:
                meta_stats["missing_title"] += 1
            if not page.metadata.description:
                meta_stats["missing_description"] += 1
            if not page.metadata.keywords:
                meta_stats["missing_keywords"] += 1
            if page.metadata.canonical:
                meta_stats["has_canonical"] += 1
            if page.metadata.robots:
                meta_stats["has_robots"] += 1
            if page.metadata.og_tags:
                meta_stats["has_og"] += 1
            if page.metadata.twitter_tags:
                meta_stats["has_twitter"] += 1
            if page.metadata.schema_org:
                meta_stats["has_schema"] += 1
        
        # Преобразуем в проценты
        meta_stats["missing_title_percent"] = meta_stats["missing_title"] / pages_count * 100
        meta_stats["missing_description_percent"] = meta_stats["missing_description"] / pages_count * 100
        meta_stats["missing_keywords_percent"] = meta_stats["missing_keywords"] / pages_count * 100
        meta_stats["has_canonical_percent"] = meta_stats["has_canonical"] / pages_count * 100
        meta_stats["has_robots_percent"] = meta_stats["has_robots"] / pages_count * 100
        meta_stats["has_og_percent"] = meta_stats["has_og"] / pages_count * 100
        meta_stats["has_twitter_percent"] = meta_stats["has_twitter"] / pages_count * 100
        meta_stats["has_schema_percent"] = meta_stats["has_schema"] / pages_count * 100
        
        # Статистика по ссылкам
        links_stats = {
            "total_internal_links": 0,
            "total_external_links": 0
        }
        
        for page in pages.values():
            links_stats["total_internal_links"] += len(page.structure.links.get("internal", []))
            links_stats["total_external_links"] += len(page.structure.links.get("external", []))
        
        links_stats["avg_internal_links_per_page"] = links_stats["total_internal_links"] / pages_count
        links_stats["avg_external_links_per_page"] = links_stats["total_external_links"] / pages_count
        
        # Собираем в общую статистику
        return {
            "pages_count": pages_count,
            "content": {
                "avg_word_count": avg_word_count,
                "avg_char_count": avg_char_count,
                "total_words": sum(page.content.word_count for page in pages.values()),
                "total_chars": sum(page.content.char_count for page in pages.values())
            },
            "headings": headings_stats,
            "meta": meta_stats,
            "links": links_stats
        }
    
    def _build_site_structure(self, pages: Dict[str, PageData]) -> Dict[str, Any]:
        """
        Строит структуру сайта на основе данных страниц.
        
        Args:
            pages: Словарь URL -> PageData
            
        Returns:
            Dict[str, Any]: Структура сайта в виде дерева
        """
        if not pages:
            return {}
            
        # Разбираем URL на компоненты для построения дерева
        url_paths = {}
        
        for url, page_data in pages.items():
            parsed_url = urlparse(url)
            path = parsed_url.path or "/"
            
            url_paths[path] = {
                "url": url,
                "title": page_data.structure.title
            }
        
        # Строим дерево
        root = {"name": "/", "children": [], "url": "", "title": "Root"}
        
        for path, info in sorted(url_paths.items()):
            # Разбиваем путь на сегменты
            segments = [s for s in path.split("/") if s]
            
            # Начинаем с корня
            current = root
            current_path = ""
            
            for i, segment in enumerate(segments):
                current_path += f"/{segment}"
                
                # Ищем существующий сегмент в дочерних элементах
                child = None
                for child_node in current["children"]:
                    if child_node["name"] == segment:
                        child = child_node
                        break
                
                # Если не нашли, создаем
                if child is None:
                    child = {
                        "name": segment,
                        "path": current_path,
                        "children": [],
                        "url": info["url"] if current_path == path else "",
                        "title": info["title"] if current_path == path else segment
                    }
                    current["children"].append(child)
                
                # Переходим к этому дочернему элементу
                current = child
        
        return root
    
    def _generate_simulated_search_results(self, query: str, results_count: int) -> SearchResultData:
        """
        Генерирует имитацию результатов поиска.
        
        Args:
            query: Поисковый запрос
            results_count: Количество результатов
            
        Returns:
            SearchResultData: Имитация результатов поиска
        """
        results = []
        
        # Создаем заглушку результатов
        for i in range(min(10, results_count)):
            results.append({
                "position": i + 1,
                "title": f"Sample Result {i+1} for {query}",
                "url": f"http://example.com/result{i+1}",
                "snippet": f"Sample snippet containing the query '{query}' and other information for result {i+1}.",
                "displayed_url": f"example.com/result{i+1}",
                "timestamp": time.time()
            })
            
        # Связанные запросы
        related_terms = ["guide", "tutorial", "example", "best", "review", "vs", "alternative"]
        related_queries = [f"{query} {term}" for term in related_terms[:min(7, len(related_terms))]]
        
        return SearchResultData(
            query=query,
            results=results,
            related_queries=related_queries,
            search_features={
                "knowledge_panel": False,
                "top_stories": False,
                "local_pack": False,
                "ads": False
            },
            search_engine="simulated",
            timestamp=datetime.now()
        )
