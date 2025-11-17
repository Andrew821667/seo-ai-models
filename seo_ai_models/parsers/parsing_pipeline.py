"""
Модуль парсинг-конвейера для проекта SEO AI Models.
Интегрирует различные компоненты парсинга в единый конвейер.
"""

import logging
import re
import time
import json
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

from seo_ai_models.parsers.crawlers.web_crawler import WebCrawler
from seo_ai_models.parsers.extractors.content_extractor import ContentExtractor
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.analyzers.serp_analyzer import SERPAnalyzer
from seo_ai_models.parsers.utils.request_utils import create_session, fetch_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ParsingPipeline:
    """
    Интегрирует различные компоненты парсинга в единый конвейер.
    Предоставляет методы высокого уровня для общих задач парсинга.
    """

    def __init__(
        self,
        user_agent: str = "SEOAIModels ParsingPipeline/1.0",
        respect_robots: bool = True,
        delay: float = 1.0,
        max_pages: int = 100,
        search_engine: str = "google",
    ):
        """
        Инициализация ParsingPipeline.

        Args:
            user_agent: User-Agent для запросов
            respect_robots: Уважать ли robots.txt
            delay: Задержка между запросами в секундах
            max_pages: Максимальное количество страниц для сканирования
            search_engine: Поисковая система для использования в анализе SERP
        """
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.delay = delay
        self.max_pages = max_pages
        self.search_engine = search_engine

        # Инициализация компонентов
        self.session = create_session(user_agent=user_agent)
        self.content_extractor = ContentExtractor()
        self.meta_extractor = MetaExtractor()
        self.serp_analyzer = SERPAnalyzer(
            user_agent=user_agent,
            search_engine=search_engine,
            delay=delay,
            content_extractor=self.content_extractor,
            meta_extractor=self.meta_extractor,
        )

    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Анализ одного URL, извлечение контента и мета-информации.

        Args:
            url: URL для анализа

        Returns:
            Dict: Результаты анализа
        """
        logger.info(f"Анализ URL: {url}")

        # Получение контента
        html_content, status_code, headers = fetch_url(url, session=self.session)

        if not html_content or status_code != 200:
            logger.warning(f"Не удалось получить {url}: HTTP {status_code}")
            return {
                "url": url,
                "success": False,
                "status_code": status_code,
                "error": f"Не удалось получить URL: HTTP {status_code}",
            }

        # Извлечение контента и мета-информации
        try:
            content_data = self.content_extractor.extract_content(html_content, url)
            meta_data = self.meta_extractor.extract_meta_information(html_content, url)

            # Объединение результатов
            return {
                "url": url,
                "success": True,
                "status_code": status_code,
                "content_analysis": content_data,
                "meta_analysis": meta_data,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе {url}: {e}")
            return {
                "url": url,
                "success": False,
                "status_code": status_code,
                "error": f"Ошибка при анализе контента: {str(e)}",
            }

    def crawl_and_analyze_site(
        self,
        base_url: str,
        max_pages: Optional[int] = None,
        custom_crawler: Optional[WebCrawler] = None,
    ) -> Dict[str, Any]:
        """
        Сканирование веб-сайта и анализ его страниц.

        Args:
            base_url: Начальный URL для сканирования
            max_pages: Максимальное количество страниц для сканирования (переопределяет настройку экземпляра)
            custom_crawler: Пользовательский экземпляр WebCrawler для использования

        Returns:
            Dict: Результаты сканирования и анализа
        """
        logger.info(f"Сканирование и анализ сайта: {base_url}")

        # Использование предоставленного max_pages или значения по умолчанию
        pages_limit = max_pages or self.max_pages

        # Создание краулера или использование пользовательского
        crawler = custom_crawler or WebCrawler(
            base_url=base_url,
            max_pages=pages_limit,
            delay=self.delay,
            respect_robots=self.respect_robots,
            user_agent=self.user_agent,
        )

        # Сканирование сайта
        crawl_results = crawler.crawl()
        crawled_urls = crawl_results["crawled_urls"]

        logger.info(f"Просканировано {len(crawled_urls)} страниц. Анализ контента...")

        # Анализ каждой просканированной страницы
        page_analyses = []

        for url in crawled_urls:
            page_analysis = self.analyze_url(url)
            page_analyses.append(page_analysis)

        # Создание структуры сайта и статистики
        site_structure = self._build_site_structure(page_analyses)
        site_statistics = self._calculate_site_statistics(page_analyses)

        return {
            "base_url": base_url,
            "crawl_results": crawl_results,
            "page_analyses": page_analyses,
            "site_structure": site_structure,
            "site_statistics": site_statistics,
            "timestamp": time.time(),
        }

    def analyze_keyword(self, keyword: str, analyze_competitors: bool = True) -> Dict[str, Any]:
        """
        Анализ ключевого слова путем проверки результатов поиска и, при необходимости, анализа конкурентов.

        Args:
            keyword: Ключевое слово для анализа
            analyze_competitors: Анализировать ли страницы конкурентов

        Returns:
            Dict: Результаты анализа ключевого слова
        """
        logger.info(f"Анализ ключевого слова: {keyword}")

        # Анализ результатов поиска
        serp_analysis = self.serp_analyzer.analyze_top_results(keyword)

        # Получение связанных ключевых слов
        related_keywords = self.serp_analyzer.get_related_queries(keyword)

        result = {
            "keyword": keyword,
            "serp_analysis": serp_analysis,
            "related_keywords": related_keywords,
            "timestamp": time.time(),
        }

        # Если запрошено, анализ топ-страниц конкурентов более подробно
        if analyze_competitors and serp_analysis.get("results"):
            competitors_analysis = []

            # Анализ топ-3 конкурентов
            for competitor in serp_analysis["results"][:3]:
                competitor_url = competitor.get("url")
                if competitor_url:
                    competitor_analysis = self.analyze_url(competitor_url)
                    competitors_analysis.append(competitor_analysis)

            result["competitors_analysis"] = competitors_analysis

        return result

    def analyze_content_against_keywords(
        self, content: str, keywords: List[str], content_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Анализ контента по списку целевых ключевых слов.

        Args:
            content: Контент для анализа (HTML или обычный текст)
            keywords: Список целевых ключевых слов
            content_url: URL контента (для ссылки)

        Returns:
            Dict: Результаты анализа контента
        """
        logger.info(f"Анализ контента по {len(keywords)} ключевым словам")

        # Определение, является ли контент HTML
        is_html = bool(re.search(r"<\s*html", content, re.IGNORECASE))

        # Если HTML, используйте экстрактор контента
        if is_html:
            content_data = self.content_extractor.extract_content(content, content_url)
            all_text = content_data.get("content", {}).get("all_text", "")
        else:
            all_text = content

        # Анализ наличия и плотности ключевых слов
        keyword_analysis = {}

        for keyword in keywords:
            # Подсчет точных совпадений (без учета регистра)
            exact_count = len(re.findall(rf"\b{re.escape(keyword)}\b", all_text, re.IGNORECASE))

            # Расчет плотности ключевых слов
            word_count = len(all_text.split())
            density = round((exact_count / word_count) * 100, 2) if word_count > 0 else 0

            # Проверка наличия ключевого слова в заголовке, заголовках
            in_title = False
            in_headings = []

            if is_html:
                title = content_data.get("title", "")
                in_title = bool(re.search(rf"\b{re.escape(keyword)}\b", title, re.IGNORECASE))

                headings = content_data.get("headings", {})
                for h_level, h_texts in headings.items():
                    for h_text in h_texts:
                        if re.search(rf"\b{re.escape(keyword)}\b", h_text, re.IGNORECASE):
                            in_headings.append(h_level)
                            break

            keyword_analysis[keyword] = {
                "count": exact_count,
                "density": density,
                "in_title": in_title,
                "in_headings": in_headings,
            }

        # Теперь сравните с конкурентами SERP для этих ключевых слов
        competitors_data = {}

        for keyword in keywords:
            try:
                serp_analysis = self.serp_analyzer.analyze_top_results(keyword)
                # Извлечение только соответствующей статистики
                competitors_data[keyword] = {
                    "avg_word_count": serp_analysis.get("analysis", {})
                    .get("word_count", {})
                    .get("average", 0),
                    "avg_keyword_presence": self._calculate_avg_keyword_presence(
                        keyword, serp_analysis.get("detailed_results", [])
                    ),
                }
            except Exception as e:
                logger.error(f"Ошибка при анализе SERP для ключевого слова '{keyword}': {e}")
                competitors_data[keyword] = {"error": str(e)}

        return {
            "content_url": content_url,
            "content_length": len(all_text),
            "word_count": len(all_text.split()),
            "keyword_analysis": keyword_analysis,
            "competitors_data": competitors_data,
            "timestamp": time.time(),
        }

    def _build_site_structure(self, page_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Построение структуры сайта на основе анализа страниц.

        Args:
            page_analyses: Список результатов анализа страниц

        Returns:
            Dict: Структура сайта
        """
        # Создание отображения путей на страницы
        paths_map = {}

        for page in page_analyses:
            if not page.get("success", False):
                continue

            url = page.get("url", "")
            if not url:
                continue

            # Разбор URL для получения пути
            parsed_url = urlparse(url)
            path = parsed_url.path or "/"

            # Сохранение информации о странице
            paths_map[path] = {
                "url": url,
                "title": page.get("content_analysis", {}).get("title", ""),
                "headings": page.get("content_analysis", {}).get("headings", {}),
            }

        # Построение структуры дерева
        root = {"name": "/", "children": [], "url": "", "title": "Root"}

        for path, page_info in sorted(paths_map.items()):
            # Разделение пути на сегменты
            segments = [s for s in path.split("/") if s]

            # Переход к правильной позиции в дереве
            current = root
            current_path = ""

            for i, segment in enumerate(segments):
                current_path += f"/{segment}"

                # Поиск дочернего элемента с этим именем
                child = next((c for c in current["children"] if c["name"] == segment), None)

                if child is None:
                    # Создание нового узла
                    child = {
                        "name": segment,
                        "path": current_path,
                        "children": [],
                        "url": page_info["url"] if current_path == path else "",
                        "title": page_info["title"] if current_path == path else segment,
                    }
                    current["children"].append(child)

                current = child

        return root

    def _calculate_site_statistics(self, page_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Расчет статистики по всему сайту на основе анализа страниц.

        Args:
            page_analyses: Список результатов анализа страниц

        Returns:
            Dict: Статистика по сайту
        """
        # Фильтрация успешных анализов
        successful_analyses = [p for p in page_analyses if p.get("success", False)]

        if not successful_analyses:
            return {"pages_count": 0, "error": "Нет успешно проанализированных страниц"}

        # Базовые подсчеты
        pages_count = len(successful_analyses)

        # Статистика контента
        word_counts = []
        content_lengths = []

        # Статистика мета-тегов
        title_lengths = []
        meta_desc_lengths = []
        missing_meta_desc = 0
        missing_titles = 0

        # Статистика структуры
        h1_counts = []
        heading_counts = {f"h{i}": [] for i in range(1, 7)}

        # Статистика ссылок
        internal_link_counts = []
        external_link_counts = []

        for page in successful_analyses:
            # Извлечение статистики контента
            content_analysis = page.get("content_analysis", {})
            meta_analysis = page.get("meta_analysis", {})

            # Статистика текста
            all_text = content_analysis.get("content", {}).get("all_text", "")
            content_lengths.append(len(all_text))
            word_counts.append(len(all_text.split()))

            # Статистика мета-тегов
            meta_tags = meta_analysis.get("meta_tags", {})

            title = meta_tags.get("title", "")
            title_lengths.append(len(title) if title else 0)
            if not title:
                missing_titles += 1

            desc = meta_tags.get("description", "")
            meta_desc_lengths.append(len(desc) if desc else 0)
            if not desc:
                missing_meta_desc += 1

            # Статистика заголовков
            headings = content_analysis.get("headings", {})
            for level, texts in headings.items():
                if level in heading_counts:
                    heading_counts[level].append(len(texts))

            h1_counts.append(len(headings.get("h1", [])))

            # Статистика ссылок
            links = meta_analysis.get("links", {})
            internal_link_counts.append(len(links.get("internal", [])))
            external_link_counts.append(len(links.get("external", [])))

        def safe_avg(values):
            return sum(values) / len(values) if values else 0

        # Компиляция статистики
        statistics = {
            "pages_count": pages_count,
            "content": {
                "avg_word_count": safe_avg(word_counts),
                "min_word_count": min(word_counts) if word_counts else 0,
                "max_word_count": max(word_counts) if word_counts else 0,
                "avg_content_length": safe_avg(content_lengths),
            },
            "meta": {
                "avg_title_length": safe_avg(title_lengths),
                "missing_titles_count": missing_titles,
                "missing_titles_percentage": (
                    (missing_titles / pages_count * 100) if pages_count else 0
                ),
                "avg_meta_desc_length": safe_avg(meta_desc_lengths),
                "missing_meta_desc_count": missing_meta_desc,
                "missing_meta_desc_percentage": (
                    (missing_meta_desc / pages_count * 100) if pages_count else 0
                ),
            },
            "structure": {
                "avg_h1_count": safe_avg(h1_counts),
                "pages_without_h1": sum(1 for count in h1_counts if count == 0),
                "pages_with_multiple_h1": sum(1 for count in h1_counts if count > 1),
                "heading_distribution": {
                    level: {
                        "avg_count": safe_avg(counts),
                        "max_count": max(counts) if counts else 0,
                    }
                    for level, counts in heading_counts.items()
                },
            },
            "links": {
                "avg_internal_links": safe_avg(internal_link_counts),
                "avg_external_links": safe_avg(external_link_counts),
            },
        }

        return statistics

    def _calculate_avg_keyword_presence(
        self, keyword: str, detailed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Расчет среднего присутствия ключевых слов на страницах конкурентов.

        Args:
            keyword: Целевое ключевое слово
            detailed_results: Список подробных результатов SERP

        Returns:
            Dict: Статистика присутствия ключевых слов
        """
        if not detailed_results:
            return {}

        in_title_count = 0
        in_h1_count = 0
        in_other_headings_count = 0
        densities = []

        for result in detailed_results:
            content_analysis = result.get("content_analysis", {})

            # Проверка заголовка
            title = content_analysis.get("title", "")
            if re.search(rf"\b{re.escape(keyword)}\b", title, re.IGNORECASE):
                in_title_count += 1

            # Проверка заголовков
            headings = content_analysis.get("headings", {})
            h1_texts = headings.get("h1", [])
            if any(re.search(rf"\b{re.escape(keyword)}\b", h, re.IGNORECASE) for h in h1_texts):
                in_h1_count += 1

            other_headings = []
            for h_level, h_texts in headings.items():
                if h_level != "h1":
                    other_headings.extend(h_texts)

            if any(
                re.search(rf"\b{re.escape(keyword)}\b", h, re.IGNORECASE) for h in other_headings
            ):
                in_other_headings_count += 1

            # Расчет плотности ключевых слов
            all_text = content_analysis.get("content", {}).get("all_text", "")
            if all_text:
                word_count = len(all_text.split())
                keyword_count = len(
                    re.findall(rf"\b{re.escape(keyword)}\b", all_text, re.IGNORECASE)
                )

                if word_count > 0:
                    density = (keyword_count / word_count) * 100
                    densities.append(density)

        total = len(detailed_results)

        return {
            "in_title_percentage": (in_title_count / total * 100) if total else 0,
            "in_h1_percentage": (in_h1_count / total * 100) if total else 0,
            "in_other_headings_percentage": (in_other_headings_count / total * 100) if total else 0,
            "avg_density": sum(densities) / len(densities) if densities else 0,
        }

    def save_analysis_to_file(self, analysis: Dict[str, Any], filename: str) -> bool:
        """
        Сохранение результатов анализа в JSON-файл.

        Args:
            analysis: Результаты анализа
            filename: Имя выходного файла

        Returns:
            bool: Статус успеха
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"Анализ сохранен в {filename}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении анализа в {filename}: {e}")
            return False
