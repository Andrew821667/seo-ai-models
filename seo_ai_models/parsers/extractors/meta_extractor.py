"""
Meta Extractor модуль для проекта SEO AI Models.
Предоставляет функциональность для извлечения мета-тегов, заголовков и ссылок из HTML.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetaExtractor:
    """
    Извлекает мета-информацию из HTML-страниц, включая мета-теги,
    заголовки, структурированные данные и связи между ссылками.
    """

    def __init__(self):
        """Инициализация MetaExtractor."""
        # Распространенные мета-теги для извлечения
        self.meta_tags = [
            "description",
            "keywords",
            "author",
            "viewport",
            "robots",
            "canonical",
            "og:title",
            "og:description",
            "og:image",
            "og:url",
            "og:type",
            "twitter:card",
            "twitter:title",
            "twitter:description",
            "twitter:image",
        ]

    def extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Извлечение мета-тегов из HTML.

        Args:
            soup: BeautifulSoup объект

        Returns:
            Dict: Мета-теги и их содержимое
        """
        meta_data = {}
                logger.info(f"HTML content length: {len(str(soup))}")

        # Извлечение стандартных мета-тегов
        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")

            if name and content:
                meta_data[name.lower()] = content

        # Извлечение заголовка
        title_tag = soup.find("title")
        if title_tag:
            meta_data["title"] = title_tag.text.strip()

        # Извлечение канонического URL
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            meta_data["canonical"] = canonical["href"]

                # Extract meta from Next.js __NEXT_DATA__ if present (common in Next.js SPAs)
        try:
            next_data_script = soup.find("script", id="__NEXT_DATA__", type="application/json")
                                                    logger.info(f"Searching for __NEXT_DATA__ script tag...")
                                                    logger.info(f"__NEXT_DATA__ script found: {next_data_script is not None}"))
            if next_data_script and next_data_script.string:
                import json
                next_data = json.loads(next_data_script.string)
                            logger.info(f"next_data keys: {list(next_data.keys()) if isinstance(next_data, dict) else 'Not a dict'}")
                
                # Try to extract title and description from Next.js page props
                page_props = next_data.get("props", {}).get("pageProps", {})
                            logger.info(f"page_props keys: {list(page_props.keys()) if isinstance(page_props, dict) else 'Not a dict'}")
                
                # Check for SEO/meta data in various common locations
                seo_data = page_props.get("seo", {}) or page_props.get("meta", {}) or page_props
                
                # Extract title if not already found
                if "title" not in meta_data or not meta_data["title"]:
                    title = (
                        seo_data.get("title") or 
                        seo_data.get("metaTitle") or 
                        page_props.get("title") or
                        next_data.get("props", {}).get("title", "")
                    )
                    if title:
                        meta_data["title"] = title
                        logger.info(f"Extracted title from __NEXT_DATA__: {title}")
                
                # Extract description if not already found
                if "description" not in meta_data or not meta_data["description"]:
                    description = (
                        seo_data.get("description") or 
                        seo_data.get("metaDescription") or
                        page_props.get("description", "")
                    )
                    if description:
                        meta_data["description"] = description
                        logger.info(f"Extracted description from __NEXT_DATA__: {description[:100]}")
                        
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.warning(f"Could not extract meta from __NEXT_DATA__: {e}")


        return meta_data

    def extract_structured_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлечение структурированных данных (JSON-LD, microdata) из HTML.

        Args:
            soup: BeautifulSoup объект

        Returns:
            List[Dict]: Список объектов структурированных данных
        """
        structured_data = []

        # Извлечение JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json

                data = json.loads(script.string)
                structured_data.append({"type": "json-ld", "data": data})
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Ошибка парсинга JSON-LD: {e}")

        # Базовое извлечение микроданных (упрощенно)
        items_with_itemtype = soup.find_all(itemtype=True)
        for item in items_with_itemtype:
            item_type = item.get("itemtype", "")

            if not item_type:
                continue

            properties = {}
            for prop in item.find_all(itemprop=True):
                prop_name = prop.get("itemprop", "")

                # Попытка извлечения значения свойства
                if prop.name == "meta":
                    prop_value = prop.get("content", "")
                elif prop.name == "link":
                    prop_value = prop.get("href", "")
                elif prop.name == "img":
                    prop_value = prop.get("src", "")
                elif prop.name == "time":
                    prop_value = prop.get("datetime", prop.text.strip())
                else:
                    prop_value = prop.text.strip()

                if prop_name and prop_value:
                    properties[prop_name] = prop_value

            if properties:
                structured_data.append(
                    {"type": "microdata", "itemtype": item_type, "properties": properties}
                )

        return structured_data

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Извлечение и категоризация ссылок из HTML.

        Args:
            soup: BeautifulSoup объект
            base_url: Базовый URL для разрешения относительных ссылок

        Returns:
            Dict: Категоризированные ссылки
        """
        links = {"internal": [], "external": [], "resources": [], "social": []}

        base_domain = urlparse(base_url).netloc
        social_domains = [
            "facebook.com",
            "twitter.com",
            "linkedin.com",
            "instagram.com",
            "youtube.com",
            "pinterest.com",
            "tiktok.com",
        ]

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            text = anchor.text.strip()

            # Разрешение относительных URL
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)

            # Пропуск URL только с фрагментами и javascript: ссылок
            if not parsed_url.netloc and not parsed_url.path:
                continue
            if href.startswith("javascript:") or href.startswith("#"):
                continue

            link_info = {
                "url": absolute_url,
                "text": text,
                "rel": anchor.get("rel", []),
                "title": anchor.get("title", ""),
            }

            # Категоризация ссылки
            if any(social in parsed_url.netloc for social in social_domains):
                links["social"].append(link_info)
            elif parsed_url.netloc == base_domain:
                links["internal"].append(link_info)
            else:
                links["external"].append(link_info)

        # Извлечение ссылок на ресурсы
        resource_elements = soup.find_all(["img", "script", "link", "video", "audio", "source"])
        for element in resource_elements:
            src = element.get("src") or element.get("href")
            if not src:
                continue

            absolute_url = urljoin(base_url, src)
            resource_info = {
                "url": absolute_url,
                "type": element.name,
                "alt": element.get("alt", "") if element.name == "img" else "",
                "rel": element.get("rel", []) if element.name == "link" else [],
            }

            links["resources"].append(resource_info)

        return links

    def extract_headers(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Извлечение заголовков (h1-h6) из HTML.

        Args:
            soup: BeautifulSoup объект

        Returns:
            Dict: Заголовки по уровням
        """
        headers = {}

        for level in range(1, 7):
            tag = f"h{level}"
            headers[tag] = [h.text.strip() for h in soup.find_all(tag)]

        return headers

    def extract_meta_information(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Извлечение всей мета-информации из HTML.

        Args:
            html_content: HTML-контент
            url: URL страницы

        Returns:
            Dict: Вся извлеченная мета-информация
        """
        soup = BeautifulSoup(html_content, "html.parser")

        meta_tags = self.extract_meta_tags(soup)
        structured_data = self.extract_structured_data(soup)
        links = self.extract_links(soup, url)
        headers = self.extract_headers(soup)

        # Расчет некоторых статистик
        word_count = len(re.findall(r"\w+", soup.get_text()))

        # Извлечение данных OpenGraph и Twitter card
        og_data = {k: v for k, v in meta_tags.items() if k.startswith("og:")}
        twitter_data = {k: v for k, v in meta_tags.items() if k.startswith("twitter:")}

        # Разбор компонентов URL
        parsed_url = urlparse(url)

        return {
            "url": {
                "full": url,
                "scheme": parsed_url.scheme,
                "domain": parsed_url.netloc,
                "path": parsed_url.path,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            },
            "meta_tags": meta_tags,
            "opengraph": og_data,
            "twitter_card": twitter_data,
            "structured_data": structured_data,
            "headers": headers,
            "links": links,
            "statistics": {
                "word_count": word_count,
                "internal_links": len(links["internal"]),
                "external_links": len(links["external"]),
                "social_links": len(links["social"]),
                "resource_links": len(links["resources"]),
            },
        }
