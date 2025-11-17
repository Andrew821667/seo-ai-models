"""
Структуры данных для результатов парсинга в проекте SEO AI Models.
Обеспечивает стандартизированный формат данных, совместимый с ядром системы.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


@dataclass
class TextContent:
    """
    Структура для хранения текстового контента страницы.

    Attributes:
        full_text: Полный текст страницы
        paragraphs: Текст разбитый на параграфы
        sentences: Текст разбитый на предложения
        word_count: Количество слов
        char_count: Количество символов
        keywords: Извлеченные ключевые слова
        language: Определенный язык текста
    """

    full_text: str
    paragraphs: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    keywords: Dict[str, float] = field(default_factory=dict)
    language: Optional[str] = None


@dataclass
class PageStructure:
    """
    Структура страницы.

    Attributes:
        title: Заголовок страницы
        headings: Заголовки на странице по уровням (h1-h6)
        links: Ссылки на странице
        images: Изображения на странице
        tables: Таблицы на странице
        lists: Списки на странице
    """

    title: str
    headings: Dict[str, List[str]] = field(
        default_factory=lambda: {f"h{i}": [] for i in range(1, 7)}
    )
    links: Dict[str, List[Dict[str, str]]] = field(
        default_factory=lambda: {"internal": [], "external": []}
    )
    images: List[Dict[str, str]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    lists: List[Dict[str, List[str]]] = field(default_factory=list)


@dataclass
class MetaData:
    """
    Метаданные страницы.

    Attributes:
        title: Title страницы
        description: Meta description
        keywords: Meta keywords
        canonical: Canonical URL
        robots: Директивы robots
        og_tags: Open Graph теги
        twitter_tags: Twitter Card теги
        schema_org: Разметка schema.org
        additional_meta: Дополнительные meta-теги
    """

    title: str = ""
    description: str = ""
    keywords: str = ""
    canonical: Optional[str] = None
    robots: Optional[str] = None
    og_tags: Dict[str, str] = field(default_factory=dict)
    twitter_tags: Dict[str, str] = field(default_factory=dict)
    schema_org: List[Dict[str, Any]] = field(default_factory=list)
    additional_meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class PageData:
    """
    Данные страницы, объединяющие контент, структуру и метаданные.

    Attributes:
        url: URL страницы
        content: Текстовый контент
        structure: Структура страницы
        metadata: Метаданные
        html_stats: Статистика HTML
        performance: Метрики производительности
        parsed_at: Время парсинга
    """

    url: str
    content: TextContent
    structure: PageStructure
    metadata: MetaData
    html_stats: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    parsed_at: datetime = field(default_factory=datetime.now)


@dataclass
class SiteData:
    """
    Данные сайта, собранные в результате сканирования.

    Attributes:
        base_url: Базовый URL сайта
        pages: Данные страниц
        site_structure: Структура сайта
        statistics: Общая статистика по сайту
        crawl_info: Информация о процессе сканирования
    """

    base_url: str
    pages: Dict[str, PageData] = field(default_factory=dict)
    site_structure: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    crawl_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResultData:
    """
    Данные результатов поиска.

    Attributes:
        query: Поисковый запрос
        results: Результаты поиска
        related_queries: Связанные запросы
        search_features: Особенности поисковой выдачи
        search_engine: Поисковая система
        timestamp: Время запроса
    """

    query: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    search_features: Dict[str, Any] = field(default_factory=dict)
    search_engine: str = "google"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ParserResult:
    """
    Общий результат парсинга, который может содержать различные типы данных.

    Attributes:
        success: Успешность парсинга
        error: Сообщение об ошибке
        page_data: Данные страницы (если парсился URL)
        site_data: Данные сайта (если сканировался сайт)
        search_data: Данные поиска (если парсились результаты поиска)
        processing_time: Время обработки
    """

    success: bool
    error: Optional[str] = None
    page_data: Optional[PageData] = None
    site_data: Optional[SiteData] = None
    search_data: Optional[SearchResultData] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует результат парсинга в словарь.

        Returns:
            Dict[str, Any]: Словарное представление результата
        """
        result = {"success": self.success, "processing_time": self.processing_time}

        if self.error:
            result["error"] = self.error

        if self.page_data:
            # Преобразование PageData в словарь
            result["page_data"] = asdict(self.page_data)

        if self.site_data:
            # Преобразование SiteData в словарь
            result["site_data"] = asdict(self.site_data)

        if self.search_data:
            # Преобразование SearchResultData в словарь
            result["search_data"] = asdict(self.search_data)

        return result

    @classmethod
    def create_error(cls, error_message: str, processing_time: float = 0.0) -> "ParserResult":
        """
        Создает результат парсинга с ошибкой.

        Args:
            error_message: Сообщение об ошибке
            processing_time: Время обработки

        Returns:
            ParserResult: Результат парсинга с ошибкой
        """
        return cls(success=False, error=error_message, processing_time=processing_time)

    @classmethod
    def create_page_result(
        cls, page_data: PageData, processing_time: float = 0.0
    ) -> "ParserResult":
        """
        Создает результат парсинга страницы.

        Args:
            page_data: Данные страницы
            processing_time: Время обработки

        Returns:
            ParserResult: Результат парсинга страницы
        """
        return cls(success=True, page_data=page_data, processing_time=processing_time)

    @classmethod
    def create_site_result(
        cls, site_data: SiteData, processing_time: float = 0.0
    ) -> "ParserResult":
        """
        Создает результат парсинга сайта.

        Args:
            site_data: Данные сайта
            processing_time: Время обработки

        Returns:
            ParserResult: Результат парсинга сайта
        """
        return cls(success=True, site_data=site_data, processing_time=processing_time)

    @classmethod
    def create_search_result(
        cls, search_data: SearchResultData, processing_time: float = 0.0
    ) -> "ParserResult":
        """
        Создает результат парсинга поисковой выдачи.

        Args:
            search_data: Данные поисковой выдачи
            processing_time: Время обработки

        Returns:
            ParserResult: Результат парсинга поисковой выдачи
        """
        return cls(success=True, search_data=search_data, processing_time=processing_time)
