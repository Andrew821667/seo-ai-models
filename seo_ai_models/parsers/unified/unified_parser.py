"""
Унифицированный парсер для проекта SEO AI Models.
Объединяет функциональность всех парсеров и предоставляет единый интерфейс,
совместимый с ядром системы.
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

# Обновленные импорты из новой структуры
from seo_ai_models.parsers.unified.crawlers.web_crawler import WebCrawler
from seo_ai_models.parsers.unified.extractors.content_extractor import ContentExtractor
from seo_ai_models.parsers.unified.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.unified.utils.request_utils import fetch_url, fetch_url_with_javascript_sync

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedParser(ParserInterface):
    """
    Унифицированный парсер, объединяющий все возможности существующих парсеров.
    Предоставляет интерфейс, совместимый с ядром системы SEO AI Models.
    """
    
    def __init__(
        self,
        user_agent: str = "SEOAIModels UnifiedParser/1.0",
        respect_robots: bool = True,
        delay: float = 1.0,
        max_pages: int = 100,
        search_engine: str = "google",
        spa_settings: Optional[Dict[str, Any]] = None,
        auto_detect_spa: bool = True,
        force_spa_mode: bool = False,
        extract_readability: bool = True,
        extract_semantic: bool = True
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
        
        # Настройки SPA по умолчанию
        self.spa_settings = spa_settings or {
            "wait_for_idle": 2000,  # мс
            "wait_for_timeout": 10000,  # мс
            "headless": True,
            "browser_type": "chromium"
        }
        
        # Инициализация текстового процессора
        self.text_processor = EnhancedTextProcessor()
        
        # Инициализация компонентов
        self.content_extractor = ContentExtractor()
        self.meta_extractor = MetaExtractor()
        
        logger.info("UnifiedParser initialized")
    
    def detect_site_type(self, url: str) -> Dict[str, Any]:
        """
        Определяет тип сайта (обычный или SPA) в упрощенном виде.
        
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
        
        # Упрощенная версия для отладки
        is_spa = False
        
        # Определяем по URL (это упрощение)
        if any(s in url for s in ["reactjs", "angular", "vuejs", "spa", "app."]):
            is_spa = True
            
        return {
            "is_spa": is_spa,
            "confidence": 0.7,
            "detection_method": "url_pattern"
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
                
            # Загружаем HTML-контент
            if is_spa:
                logger.info(f"Loading {url} as SPA")
                html_content, status_code, error = fetch_url_with_javascript_sync(
                    url,
                    headless=self.spa_settings.get("headless", True),
                    wait_for_idle=self.spa_settings.get("wait_for_idle", 2000),
                    wait_for_timeout=self.spa_settings.get("wait_for_timeout", 10000),
                    user_agent=self.user_agent
                )
            else:
                logger.info(f"Loading {url} as standard site")
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
                
            # Выбираем подходящий краулер (сейчас только WebCrawler)
            crawler = WebCrawler(
                base_url=base_url,
                max_pages=max_pages,
                delay=delay,
                respect_robots=respect_robots,
                user_agent=user_agent
            )
            
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
            
            # Парсим каждый URL
            for url in crawl_result.get("crawled_urls", []):
                try:
                    # Используем имеющийся метод для парсинга URL
                    page_result = self.parse_url(url, force_spa=is_spa, auto_detect=False)
                    
                    # Если успешно, добавляем в результаты
                    if page_result.get("success", False) and "page_data" in page_result:
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
                        
                except Exception as e:
                    logger.error(f"Error parsing crawled URL {url}: {str(e)}")
                    continue
                    
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
            
            # Упрощенная имитация результатов поиска
            search_results = []
            related_queries = []
            
            # Генерируем имитацию результатов поиска
            for i in range(1, min(11, results_count + 1)):
                result = {
                    "position": i,
                    "title": f"Sample Result {i} for {query}",
                    "url": f"http://example.com/result{i}",
                    "snippet": f"Sample snippet containing the query '{query}' and other information for result {i}."
                }
                search_results.append(result)
                
            # Генерируем имитацию связанных запросов
            related_terms = ["guide", "tutorial", "examples", "best", "review", "vs", "alternative"]
            for term in related_terms[:5]:
                related_queries.append(f"{query} {term}")
                
            # Создаем структуру данных поиска
            search_data = SearchResultData(
                query=query,
                results=search_results,
                related_queries=related_queries,
                search_features={},
                search_engine=search_engine,
                timestamp=datetime.now()
            )
            
            # Если нужно анализировать контент результатов
            if analyze_content:
                # Ограничиваем количество результатов для анализа
                analysis_limit = min(3, len(search_results))
                
                for i in range(analysis_limit):
                    result = search_results[i]
                    url = result.get("url")
                    
                    if url:
                        try:
                            # Парсим URL для получения подробной информации
                            page_result = self.parse_url(url)
                            
                            # Добавляем подробную информацию к результату
                            if page_result.get("success", False) and "page_data" in page_result:
                                search_results[i]["detailed_analysis"] = page_result["page_data"]
                                
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
        
        # Подсчитываем слова и символы
        word_count = content_data.get("content", {}).get("word_count", 0)
        char_count = content_data.get("content", {}).get("char_count", 0)
        
        # Извлекаем ключевые слова
        keywords = {}
        if self.extract_semantic and full_text:
            try:
                # В реальном использовании мы бы применили текстовый процессор
                # Но для отладки используем упрощенный подход
                words = full_text.lower().split()
                word_freq = {}
                for word in words:
                    word = re.sub(r'[^\w\s]', '', word)
                    if word and len(word) > 3:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Берем топ-10 слов
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                for word, freq in sorted_words[:10]:
                    keywords[word] = freq / max(word_freq.values())
                
            except Exception as e:
                logger.warning(f"Error extracting keywords: {str(e)}")
        
        # Определяем язык
        language = "en"  # Упрощение для отладки
        
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
        
        # Получаем ссылки и изображения
        links = meta_data.get("links", {"internal": [], "external": []})
        images = meta_data.get("images", [])
        
        # Создаем объект PageStructure
        page_structure = PageStructure(
            title=title,
            headings=headings,
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
            canonical=meta_data.get("meta_tags", {}).get("canonical"),
            robots=meta_data.get("meta_tags", {}).get("robots"),
            og_tags=meta_data.get("meta_tags", {}).get("og", {}),
            twitter_tags=meta_data.get("meta_tags", {}).get("twitter", {}),
            schema_org=meta_data.get("schema_org", []),
            additional_meta=meta_data.get("meta_tags", {}).get("other", {})
        )
        
        # Создаем статистику HTML
        html_stats = {
            "html_size": len(html_content) if html_content else 0,
            "text_ratio": char_count / len(html_content) if html_content and len(html_content) > 0 else 0,
            "headings_count": sum(len(texts) for texts in headings.values()),
            "links_count": len(links.get("internal", [])) + len(links.get("external", [])),
            "images_count": len(images)
        }
        
        # Добавляем метрики читаемости, если нужно
        performance = {}
        if self.extract_readability and full_text:
            try:
                # Упрощенная оценка читаемости
                words = full_text.split()
                sentences = content_data.get("content", {}).get("sentences", [])
                
                if sentences and words:
                    avg_sentence_length = len(words) / len(sentences)
                    avg_word_length = sum(len(word) for word in words) / len(words)
                    
                    # Очень упрощенный расчет легкости чтения
                    readability = max(0, min(1, 1 - (avg_sentence_length / 30) - (avg_word_length / 10)))
                    
                    performance["readability"] = {
                        "score": readability,
                        "avg_sentence_length": avg_sentence_length,
                        "avg_word_length": avg_word_length
                    }
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
                if level in headings_stats:
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
