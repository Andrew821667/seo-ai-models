"""
Обновление для MetaExtractor с добавлением метода extract_from_url.
"""

from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.utils.request_utils import fetch_url


def update_meta_extractor(meta_extractor_class):
    """
    Добавляет метод extract_from_url в класс MetaExtractor.

    Args:
        meta_extractor_class: Класс MetaExtractor для обновления
    """

    def extract_from_url(self, url):
        """
        Извлекает метаданные из URL.

        Args:
            url: URL для извлечения метаданных

        Returns:
            Dict: Метаданные страницы
        """
        html_content, headers, error = fetch_url(url)

        if error or not html_content:
            return {"url": url, "success": False, "error": error or "No HTML content received"}

        return self.extract_meta_information(html_content, url)

    # Добавляем метод в класс
    setattr(meta_extractor_class, "extract_from_url", extract_from_url)

    return meta_extractor_class
