"""
Экстрактор контента для унифицированного парсера.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    Экстрактор контента из HTML-страниц.
    Извлекает текст, заголовки, параграфы и другие элементы.

    Экземпляры класса не требуют специальной инициализации,
    все методы работают со входными параметрами.
    """

    def extract_content(self, html: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Извлекает контент из HTML-страницы.

        Args:
            html: HTML-контент
            url: URL страницы (для контекста)

        Returns:
            Dict[str, Any]: Извлеченный контент
        """
        if not html:
            return {"error": "Empty HTML content"}

        soup = BeautifulSoup(html, "html.parser")

        # Извлекаем заголовок страницы
        title = self._extract_title(soup)

        # Извлекаем заголовки
        headings = self._extract_headings(soup)

        # Извлекаем текстовый контент
        content = self._extract_text_content(soup)

        # Извлекаем параграфы
        paragraphs = self._extract_paragraphs(soup)

        # Извлекаем списки
        lists = self._extract_lists(soup)

        # Извлекаем изображения
        images = self._extract_images(soup, url)

        # Формируем результат
        result = {
            "title": title,
            "headings": headings,
            "content": content,
            "paragraphs": paragraphs,
            "lists": lists,
            "images": images,
        }

        return result

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Извлекает заголовок страницы.

        Args:
            soup: BeautifulSoup-объект

        Returns:
            str: Заголовок страницы
        """
        title_tag = soup.find("title")
        h1_tag = soup.find("h1")

        # Предпочитаем title, но если его нет, используем h1
        if title_tag and title_tag.text.strip():
            return title_tag.text.strip()
        elif h1_tag and h1_tag.text.strip():
            return h1_tag.text.strip()
        else:
            return ""

    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Извлекает заголовки всех уровней.

        Args:
            soup: BeautifulSoup-объект

        Returns:
            Dict[str, List[str]]: Словарь заголовков по уровням
        """
        headings = {f"h{i}": [] for i in range(1, 7)}

        for i in range(1, 7):
            heading_tags = soup.find_all(f"h{i}")
            headings[f"h{i}"] = [tag.text.strip() for tag in heading_tags if tag.text.strip()]

        return headings

    def _extract_text_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Извлекает текстовый контент страницы.

        Args:
            soup: BeautifulSoup-объект

        Returns:
            Dict[str, Any]: Текстовый контент
        """
        # Удаляем скрипты, стили и другие нежелательные элементы
        for unwanted in soup.select("script, style, meta, noscript, iframe"):
            unwanted.decompose()

        # Получаем весь текст
        all_text = soup.get_text(separator=" ", strip=True)

        # Очищаем лишние пробелы и переносы строк
        all_text = re.sub(r"\s+", " ", all_text).strip()

        # Разбиваем на предложения
        sentences = re.split(r"[.!?]+", all_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Подсчитываем слова и символы
        word_count = len(all_text.split())
        char_count = len(all_text)

        return {
            "all_text": all_text,
            "sentences": sentences,
            "word_count": word_count,
            "char_count": char_count,
        }

    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """
        Извлекает параграфы страницы.

        Args:
            soup: BeautifulSoup-объект

        Returns:
            List[str]: Список параграфов
        """
        paragraphs = []
        p_tags = soup.find_all("p")

        for p in p_tags:
            text = p.text.strip()
            if text:
                paragraphs.append(text)

        return paragraphs

    def _extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает списки страницы.

        Args:
            soup: BeautifulSoup-объект

        Returns:
            List[Dict[str, Any]]: Список списков
        """
        lists = []

        # Извлекаем нумерованные списки
        for ol in soup.find_all("ol"):
            items = [li.text.strip() for li in ol.find_all("li") if li.text.strip()]
            if items:
                lists.append({"type": "ordered", "items": items})

        # Извлекаем маркированные списки
        for ul in soup.find_all("ul"):
            items = [li.text.strip() for li in ul.find_all("li") if li.text.strip()]
            if items:
                lists.append({"type": "unordered", "items": items})

        return lists

    def _extract_images(
        self, soup: BeautifulSoup, base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Извлекает изображения страницы.

        Args:
            soup: BeautifulSoup-объект
            base_url: Базовый URL для преобразования относительных путей

        Returns:
            List[Dict[str, str]]: Список изображений
        """
        images = []
        img_tags = soup.find_all("img")

        for img in img_tags:
            src = img.get("src", "")
            alt = img.get("alt", "")
            width = img.get("width", "")
            height = img.get("height", "")

            if src:
                # Преобразуем относительные URL в абсолютные
                if base_url and not (src.startswith("http://") or src.startswith("https://")):
                    domain = urlparse(base_url).netloc
                    scheme = urlparse(base_url).scheme

                    if src.startswith("//"):
                        src = f"{scheme}:{src}"
                    elif src.startswith("/"):
                        src = f"{scheme}://{domain}{src}"
                    else:
                        path = urlparse(base_url).path
                        if not path.endswith("/"):
                            path = path.rsplit("/", 1)[0] + "/"
                        src = f"{scheme}://{domain}{path}{src}"

                images.append({"src": src, "alt": alt, "width": width, "height": height})

        return images
