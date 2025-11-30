"""
Экстрактор метаданных для унифицированного парсера.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from seo_ai_models.parsers.unified.utils.request_utils import fetch_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetaExtractor:
    """
    Экстрактор метаданных из HTML-страниц.
    Извлекает мета-теги, Open Graph, Twitter Cards, Schema.org и другие.

    Экземпляры класса не требуют специальной инициализации,
    все методы работают со входными параметрами.
    """

    def extract_meta_information(self, html: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Извлекает метаданные из HTML-страницы.

        Args:
            html: HTML-контент
            url: URL страницы (для контекста)

        Returns:
            Dict[str, Any]: Извлеченные метаданные
        """
        if not html:
            return {"error": "Empty HTML content"}

        soup = BeautifulSoup(html, "html.parser")

        # Извлекаем мета-теги
        meta_tags = self._extract_meta_tags(soup)

        # Извлекаем Schema.org разметку
        schema_org = self._extract_schema_org(soup)

        # Извлекаем ссылки
        links = self._extract_links(soup, url)

        # Извлекаем изображения
        images = self._extract_images(soup, url)

        # Формируем результат
        result = {
            "meta_tags": meta_tags,
            "schema_org": schema_org,
            "links": links,
            "images": images,
        }

        return result

    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Извлекает метаданные из URL.

        Args:
            url: URL для извлечения метаданных

        Returns:
            Dict[str, Any]: Извлеченные метаданные
        """
        try:
            # Загружаем страницу
            html, status_code, error = fetch_url(url)

            if not html or status_code != 200:
                logger.error(f"Failed to fetch {url}: {error}")
                return {"error": f"Failed to fetch URL: {error}"}

            # Извлекаем метаданные
            return self.extract_meta_information(html, url)

        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {str(e)}")
            return {"error": f"Error extracting metadata: {str(e)}"}

    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Извлекает мета-теги.

        Args:
            soup: BeautifulSoup-объект

        Returns:
            Dict[str, Any]: Извлеченные мета-теги
        """
        meta_tags = {
            "title": "",
            "description": "",
            "keywords": "",
            "canonical": None,
            "robots": None,
            "og": {},
            "twitter": {},
            "other": {},
        }

                # Add diagnostic logging
        logger.info(f"Extracting meta tags from HTML (length: {len(soup.prettify())} chars)")

        # Заголовок
        title_tag = soup.find("title")
        if title_tag:
            meta_tags["title"] = title_tag.text.strip()
                    logger.info(f"Found title tag: '{title_tag.text.strip()[:100]}...'")
                else:
        logger.warning("No title tag found in HTML")

        # Извлекаем мета-теги
        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            property = meta.get("property", "").lower()
            content = meta.get("content", "")

            if name == "description":
                meta_tags["description"] = content
                            logger.info(f"Found description meta: '{content[:100]}...'" if len(content) > 100 else f"Found description meta: '{content}'")
            elif name == "keywords":
                meta_tags["keywords"] = content
            elif name == "robots":
                meta_tags["robots"] = content
            elif property.startswith("og:"):
                meta_tags["og"][property[3:]] = content
            elif property.startswith("twitter:"):
                meta_tags["twitter"][property[8:]] = content
            elif name:
                meta_tags["other"][name] = content

        # Канонический URL
        canonical = soup.find("link", rel="canonical")
        if canonical:
            meta_tags["canonical"] = canonical.get("href", "")

        return meta_tags

    def _extract_schema_org(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает Schema.org разметку.

        Args:
            soup: BeautifulSoup-объект

        Returns:
            List[Dict[str, Any]]: Извлеченная Schema.org разметка
        """
        schema_data = []

        # Извлекаем JSON-LD
        ld_scripts = soup.find_all("script", type="application/ld+json")
        for script in ld_scripts:
            try:
                json_text = script.string
                if json_text:
                    data = json.loads(json_text)
                    if isinstance(data, list):
                        schema_data.extend(data)
                    else:
                        schema_data.append(data)
            except json.JSONDecodeError:
                logger.warning("Error parsing JSON-LD")
                continue

        # Извлекаем микроданные (упрощенно)
        items = soup.find_all(itemscope=True)
        for item in items:
            item_type = item.get("itemtype", "")
            if item_type:
                micro_data = {"@type": item_type.split("/")[-1], "@context": "http://schema.org"}

                # Извлекаем свойства
                for prop in item.find_all(itemprop=True):
                    prop_name = prop.get("itemprop", "")
                    prop_value = prop.get("content", "") or prop.text.strip()

                    if prop_name and prop_value:
                        micro_data[prop_name] = prop_value

                schema_data.append(micro_data)

        return schema_data

    def _extract_links(
        self, soup: BeautifulSoup, base_url: Optional[str] = None
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Извлекает ссылки.

        Args:
            soup: BeautifulSoup-объект
            base_url: Базовый URL для преобразования относительных ссылок

        Returns:
            Dict[str, List[Dict[str, str]]]: Извлеченные ссылки
        """
        links = {"internal": [], "external": []}

        if not base_url:
            return links

        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            text = a_tag.text.strip()

            # Пропускаем пустые ссылки и якоря
            if not href or href.startswith("#"):
                continue

            # Преобразуем относительные URL в абсолютные
            absolute_url = urljoin(base_url, href)

            # Удаляем якоря
            absolute_url = absolute_url.split("#")[0]

            # Проверяем, внутренняя или внешняя ссылка
            link_domain = urlparse(absolute_url).netloc

            link_info = {
                "url": absolute_url,
                "text": text,
                "nofollow": "rel" in a_tag.attrs and "nofollow" in a_tag["rel"],
            }

            if (
                link_domain == base_domain
                or link_domain == f"www.{base_domain}"
                or f"www.{link_domain}" == base_domain
            ):
                links["internal"].append(link_info)
            else:
                links["external"].append(link_info)

        return links

    def _extract_images(
        self, soup: BeautifulSoup, base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Извлекает изображения.

        Args:
            soup: BeautifulSoup-объект
            base_url: Базовый URL для преобразования относительных путей

        Returns:
            List[Dict[str, str]]: Извлеченные изображения
        """
        images = []

        for img in soup.find_all("img"):
            src = img.get("src", "")
            alt = img.get("alt", "")
            title = img.get("title", "")

            if src:
                # Преобразуем относительные URL в абсолютные
                if base_url:
                    src = urljoin(base_url, src)

                images.append({"src": src, "alt": alt, "title": title})

        return images
