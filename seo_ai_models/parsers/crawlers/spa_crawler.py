"""
SPA Crawler модуль для проекта SEO AI Models.
Предоставляет функциональность для сканирования SPA-сайтов с поддержкой JavaScript.
"""

import logging
import time
import random
import asyncio
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Optional, Any, Callable, Union

from playwright.async_api import (
    async_playwright,
    Browser,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SPACrawler:
    """
    Реализация веб-краулера для SPA с поддержкой JavaScript рендеринга.
    Использует Playwright для полного рендеринга страниц, включая JavaScript-контент.
    """

    def __init__(
        self,
        base_url: str,
        max_pages: int = 100,
        delay: float = 1.0,
        wait_for_idle: int = 2000,  # мс ожидания загрузки страницы после событий idle
        wait_for_timeout: int = 5000,  # мс максимального ожидания загрузки
        respect_robots: bool = True,
        headless: bool = True,
        user_agent: str = "SEOAIModels SPACrawler/1.0",
        custom_headers: Optional[Dict[str, str]] = None,
        url_filter: Optional[Callable[[str], bool]] = None,
        browser_type: str = "chromium",  # 'chromium', 'firefox', or 'webkit'
    ):
        """
        Инициализация SPACrawler.

        Args:
            base_url: Начальный URL для сканирования
            max_pages: Максимальное количество страниц для сканирования
            delay: Задержка между запросами в секундах
            wait_for_idle: Время ожидания в мс после событий 'networkidle'
            wait_for_timeout: Максимальное время ожидания в мс для загрузки страницы
            respect_robots: Уважать ли robots.txt
            headless: Запускать ли браузер в режиме headless
            user_agent: User-Agent для запросов
            custom_headers: Дополнительные заголовки для запросов
            url_filter: Функция для фильтрации URL (возвращает True для включения URL)
            browser_type: Тип браузера для использования ('chromium', 'firefox', 'webkit')
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.wait_for_idle = wait_for_idle
        self.wait_for_timeout = wait_for_timeout
        self.respect_robots = respect_robots
        self.headless = headless
        self.user_agent = user_agent
        self.browser_type = browser_type

        self.headers = {}
        if custom_headers:
            self.headers.update(custom_headers)

        self.url_filter = url_filter
        self.crawled_urls: Set[str] = set()
        self.found_urls: Set[str] = set()
        self.failed_urls: Dict[str, str] = {}
        self.html_content: Dict[str, str] = {}

        # Извлечение домена для проверки того же домена
        parsed_url = urlparse(self.base_url)
        self.domain = parsed_url.netloc

    def is_valid_url(self, url: str) -> bool:
        """
        Проверка, является ли URL допустимым и должен ли он быть просканирован.

        Args:
            url: URL для проверки

        Returns:
            bool: True, если URL должен быть просканирован
        """
        parsed = urlparse(url)

        # Проверка, относится ли он к тому же домену
        if parsed.netloc != self.domain and parsed.netloc != "":
            return False

        # Игнорировать распространенные нетекстовые расширения
        ignored_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".pdf",
            ".zip",
            ".css",
            ".js",
            ".ico",
            ".svg",
            ".mp4",
            ".mp3",
        ]
        if any(url.lower().endswith(ext) for ext in ignored_extensions):
            return False

        # Применить пользовательский фильтр, если он предоставлен
        if self.url_filter and not self.url_filter(url):
            return False

        return True

    async def extract_urls_from_page(self, page: Page, current_url: str) -> List[str]:
        """
        Извлечение всех URL из страницы.

        Args:
            page: Объект Page Playwright
            current_url: Текущий URL для разрешения относительных ссылок

        Returns:
            List[str]: Список извлеченных URL
        """
        urls = []

        # Получение всех ссылок на странице после полной загрузки JavaScript
        hrefs = await page.evaluate(
            """() => {
            const links = Array.from(document.querySelectorAll('a[href]'));
            return links.map(link => link.href);
        }"""
        )

        for href in hrefs:
            # Очистка и нормализация URL
            parsed = urlparse(href)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                normalized_url += f"?{parsed.query}"

            if self.is_valid_url(normalized_url):
                urls.append(normalized_url)

        return urls

    async def get_rendered_html(self, page: Page) -> str:
        """
        Получение полностью отрендеренного HTML страницы.

        Args:
            page: Объект Page Playwright

        Returns:
            str: HTML-контент страницы после рендеринга
        """
        return await page.content()

    async def crawl(self) -> Dict[str, Any]:
        """
        Выполнение операции сканирования с использованием Playwright.

        Returns:
            Dict, содержащий crawled_urls, found_urls и failed_urls
        """
        logger.info(f"Начало SPA-сканирования с {self.base_url}")
        to_crawl = [self.base_url]

        async with async_playwright() as p:
            # Выбор браузера в зависимости от настройки
            if self.browser_type == "firefox":
                browser_instance = p.firefox
            elif self.browser_type == "webkit":
                browser_instance = p.webkit
            else:
                browser_instance = p.chromium  # По умолчанию

            # Запуск браузера
            browser = await browser_instance.launch(headless=self.headless)

            try:
                context = await browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1366, "height": 768},
                    extra_http_headers=self.headers,
                )

                while to_crawl and len(self.crawled_urls) < self.max_pages:
                    current_url = to_crawl.pop(0)

                    if current_url in self.crawled_urls:
                        continue

                    logger.info(f"SPA-сканирование: {current_url}")

                    try:
                        page = await context.new_page()

                        # Настройка перехватчика запросов (для оптимизации, при необходимости)
                        await page.route(
                            "**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}",
                            lambda route: route.abort(),
                        )

                        # Переход на страницу и ожидание загрузки
                        response = await page.goto(
                            current_url, wait_until="networkidle", timeout=self.wait_for_timeout
                        )

                        # Дополнительная задержка для полной загрузки динамического контента
                        await page.wait_for_timeout(self.wait_for_idle)

                        if response and response.status == 200:
                            self.crawled_urls.add(current_url)

                            # Сохранение отрендеренного HTML
                            self.html_content[current_url] = await self.get_rendered_html(page)

                            # Извлечение URL со страницы
                            new_urls = await self.extract_urls_from_page(page, current_url)

                            # Добавление URL в очередь, если они еще не просканированы или не в очереди
                            for url in new_urls:
                                self.found_urls.add(url)
                                if (
                                    url not in self.crawled_urls
                                    and url not in to_crawl
                                    and len(self.crawled_urls) < self.max_pages
                                ):
                                    to_crawl.append(url)
                        else:
                            status_code = response.status if response else "No response"
                            self.failed_urls[current_url] = f"HTTP {status_code}"

                    except PlaywrightTimeoutError:
                        logger.error(f"Timeout при сканировании {current_url}")
                        self.failed_urls[current_url] = "Timeout Error"

                    except Exception as e:
                        logger.error(f"Ошибка при SPA-сканировании {current_url}: {str(e)}")
                        self.failed_urls[current_url] = str(e)

                    finally:
                        await page.close()

                    # Уважительное сканирование - добавление задержки
                    await asyncio.sleep(self.delay + random.uniform(0, 0.5))

            finally:
                await browser.close()

        logger.info(f"SPA-сканирование завершено. Просканировано {len(self.crawled_urls)} страниц.")

        return {
            "crawled_urls": list(self.crawled_urls),
            "found_urls": list(self.found_urls),
            "failed_urls": self.failed_urls,
            "total_crawled": len(self.crawled_urls),
            "total_found": len(self.found_urls),
            "total_failed": len(self.failed_urls),
        }

    async def get_url_content(self, url: str) -> Optional[str]:
        """
        Получение HTML-контента для URL.

        Args:
            url: URL для получения контента

        Returns:
            Optional[str]: HTML-контент или None, если URL не был просканирован
        """
        return self.html_content.get(url)

    def get_crawled_urls(self) -> List[str]:
        """
        Получение списка успешно просканированных URL.

        Returns:
            List[str]: Список просканированных URL
        """
        return list(self.crawled_urls)

    def get_failed_urls(self) -> Dict[str, str]:
        """
        Получение словаря неудачных URL с сообщениями об ошибках.

        Returns:
            Dict[str, str]: Неудачные URL с ошибками
        """
        return self.failed_urls

    def run_crawler(self) -> Dict[str, Any]:
        """
        Синхронная обертка для запуска асинхронного сканера.

        Returns:
            Dict: Результаты сканирования
        """
        return asyncio.run(self.crawl())
