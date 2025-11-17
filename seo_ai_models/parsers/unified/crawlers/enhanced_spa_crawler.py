"""
Усовершенствованный SPA-краулер с использованием Playwright.
Обеспечивает поддержку современных JavaScript-приложений и фреймворков.
"""

import logging
import asyncio
import time
import json
from typing import Dict, List, Set, Optional, Any, Union
from urllib.parse import urlparse, urljoin

import playwright
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedSPACrawler:
    """
    Усовершенствованный краулер для SPA (Single Page Applications).
    Использует Playwright для полноценного рендеринга JavaScript.
    """

    def __init__(
        self,
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        delay: float = 1.0,
        respect_robots: bool = True,
        user_agent: str = "SEOAIModels EnhancedSPACrawler/2.0",
        browser_type: str = "chromium",
        headless: bool = True,
        wait_for_idle: int = 2000,
        wait_for_timeout: int = 10000,
        intercept_ajax: bool = True,
        wait_for_selectors: Optional[List[str]] = None,
        browser_args: Optional[List[str]] = None,
    ):
        """
        Инициализация усовершенствованного SPA-краулера.

        Args:
            base_url: Базовый URL для сканирования
            max_pages: Максимальное количество страниц для сканирования
            max_depth: Максимальная глубина сканирования
            delay: Задержка между запросами в секундах
            respect_robots: Уважать ли robots.txt
            user_agent: User-Agent для запросов
            browser_type: Тип браузера ('chromium', 'firefox', 'webkit')
            headless: Запускать ли браузер в режиме headless
            wait_for_idle: Время ожидания в мс после событий 'networkidle'
            wait_for_timeout: Максимальное время ожидания в мс
            intercept_ajax: Перехватывать ли AJAX-запросы
            wait_for_selectors: Селекторы, ожидание которых указывает на загрузку контента
            browser_args: Дополнительные аргументы для запуска браузера
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self.browser_type = browser_type
        self.headless = headless
        self.wait_for_idle = wait_for_idle
        self.wait_for_timeout = wait_for_timeout
        self.intercept_ajax = intercept_ajax
        self.wait_for_selectors = wait_for_selectors or [
            ".main",
            "main",
            "#content",
            "#main",
            "article",
        ]
        self.browser_args = browser_args or []

        # Парсинг базового URL
        self.base_netloc = urlparse(base_url).netloc

        # Инициализация наборов для хранения URL
        self.crawled_urls = set()
        self.found_urls = set()
        self.failed_urls = set()

        # Для хранения AJAX-запросов
        self.ajax_requests = {}

        # Инициализация robots.txt
        self.robots_rules = []
        if self.respect_robots:
            # TODO: В будущем реализовать загрузку и парсинг robots.txt
            # Например, используя библиотеку robotexclusionrulesparser или urllib.robotparser
            logger.debug(f"robots.txt support is enabled but not yet implemented for {base_url}")

        logger.info(f"EnhancedSPACrawler initialized for {base_url} with {browser_type} browser")

    async def run_crawler(self) -> Dict[str, Any]:
        """
        Запускает сканирование SPA-сайта.

        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        logger.info(f"Starting crawl of SPA site: {self.base_url}")
        start_time = time.time()

        async with async_playwright() as p:
            # Выбор типа браузера
            if self.browser_type == "firefox":
                browser_instance = p.firefox
            elif self.browser_type == "webkit":
                browser_instance = p.webkit
            else:
                browser_instance = p.chromium

            # Запуск браузера
            browser = await browser_instance.launch(headless=self.headless, args=self.browser_args)

            try:
                # Создание нового контекста
                context = await browser.new_context(
                    user_agent=self.user_agent, viewport={"width": 1366, "height": 768}
                )

                # Добавляем обработчик событий для AJAX-запросов
                if self.intercept_ajax:
                    await context.route("**/*.{json,xhr}", self._handle_ajax_request)

                # Создание URL очереди
                # Структура (url, depth)
                urls_to_visit = [(self.base_url, 0)]

                # Сканирование с заданными ограничениями
                while urls_to_visit and len(self.crawled_urls) < self.max_pages:
                    current_url, depth = urls_to_visit.pop(0)

                    # Проверяем, был ли URL уже просканирован
                    if current_url in self.crawled_urls or current_url in self.failed_urls:
                        continue

                    # Проверяем принадлежность к одному домену
                    if not self._is_same_domain(current_url):
                        continue

                    # Проверяем разрешение robots.txt
                    if self.respect_robots and not self._is_allowed_by_robots(current_url):
                        logger.info(f"Skipping {current_url} (disallowed by robots.txt)")
                        continue

                    logger.info(f"Crawling SPA URL: {current_url} (depth {depth})")

                    try:
                        # Создаем новую страницу
                        page = await context.new_page()

                        # Задаем обработчики события для отладки
                        page.on(
                            "console", lambda msg: logger.debug(f"Console {msg.type}: {msg.text}")
                        )
                        page.on("pageerror", lambda err: logger.warning(f"Page error: {err}"))

                        # Загружаем страницу
                        response = await page.goto(
                            current_url, wait_until="networkidle", timeout=self.wait_for_timeout
                        )

                        # Проверяем успешность загрузки
                        if not response or response.status >= 400:
                            logger.warning(
                                f"Failed to load {current_url}: HTTP {response.status if response else 'unknown'}"
                            )
                            self.failed_urls.add(current_url)
                            await page.close()
                            continue

                        # Дополнительное ожидание для SPA
                        try:
                            # Ожидаем селекторы, указывающие на загрузку контента
                            for selector in self.wait_for_selectors:
                                try:
                                    await page.wait_for_selector(
                                        selector, timeout=self.wait_for_idle
                                    )
                                    break  # Если нашли хотя бы один селектор, выходим
                                except:
                                    continue
                        except:
                            # Если не смогли дождаться селекторов, просто продолжаем
                            logger.debug(f"No content selectors found for {current_url}")

                        # Ожидаем дополнительное время для полной загрузки
                        await asyncio.sleep(self.delay)

                        # Получаем HTML-контент после выполнения JavaScript
                        html_content = await page.content()

                        # Извлекаем ссылки
                        links = await self._extract_links_from_page(page, current_url)

                        # Добавляем URL в список просканированных
                        self.crawled_urls.add(current_url)

                        # Добавляем найденные ссылки в общий список
                        self.found_urls.update(links)

                        # Если не достигли максимальной глубины, добавляем ссылки в очередь
                        if depth < self.max_depth:
                            for link in links:
                                if link not in self.crawled_urls and link not in self.failed_urls:
                                    if link not in [u for u, _ in urls_to_visit]:
                                        urls_to_visit.append((link, depth + 1))

                        # Закрываем страницу для освобождения ресурсов
                        await page.close()

                    except Exception as e:
                        logger.error(f"Error crawling {current_url}: {str(e)}")
                        self.failed_urls.add(current_url)
                        try:
                            await page.close()
                        except Exception as close_error:
                            logger.debug(
                                f"Failed to close page for {current_url}: {str(close_error)}"
                            )

            finally:
                # Закрываем браузер
                await browser.close()

        # Формируем результат
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"SPA crawl completed in {duration:.2f}s")
        logger.info(f"Crawled {len(self.crawled_urls)} URLs")
        logger.info(f"Found {len(self.found_urls)} URLs")
        logger.info(f"Failed {len(self.failed_urls)} URLs")

        result = {
            "base_url": self.base_url,
            "crawled_urls": list(self.crawled_urls),
            "found_urls": list(self.found_urls),
            "failed_urls": list(self.failed_urls),
            "ajax_requests": self.ajax_requests,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "respect_robots": self.respect_robots,
            "duration": duration,
            "timestamp": time.time(),
        }

        return result

    def crawl(self) -> Dict[str, Any]:
        """
        Синхронная обертка для запуска асинхронного сканирования.

        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.run_crawler())

    async def _extract_links_from_page(self, page: Page, current_url: str) -> Set[str]:
        """
        Извлекает ссылки из страницы с помощью Playwright.

        Args:
            page: Объект страницы Playwright
            current_url: Текущий URL для преобразования относительных ссылок

        Returns:
            Set[str]: Набор найденных URL
        """
        # Выполняем JavaScript для получения всех ссылок на странице
        links_js = """
        Array.from(document.querySelectorAll('a[href]'))
            .map(a => ({
                href: a.href,
                text: a.innerText.trim(),
                rel: a.rel
            }))
        """
        links_data = await page.evaluate(links_js)

        links = set()

        for link in links_data:
            href = link.get("href", "")

            # Пропускаем пустые ссылки, якоря и JavaScript
            if (
                not href
                or href.startswith("#")
                or href.startswith("javascript:")
                or href.startswith("mailto:")
                or href.startswith("tel:")
            ):
                continue

            # Преобразуем относительные URL в абсолютные
            absolute_url = urljoin(current_url, href)

            # Удаляем якоря и приводим к нижнему регистру
            absolute_url = absolute_url.split("#")[0].lower()

            # Проверяем, что URL принадлежит тому же домену
            if self._is_same_domain(absolute_url):
                links.add(absolute_url)

        return links

    async def _handle_ajax_request(self, route, request):
        """
        Обрабатывает AJAX-запросы, перехватывая их для анализа.

        Args:
            route: Объект маршрута для перехвата запросов
            request: Перехваченный запрос
        """
        url = request.url
        method = request.method

        # Позволяем запросу продолжить выполнение
        await route.continue_()

        # Сохраняем информацию о запросе
        self.ajax_requests[url] = {
            "method": method,
            "url": url,
            "headers": request.headers,
            "resource_type": request.resource_type,
        }

    def _is_same_domain(self, url: str) -> bool:
        """
        Проверяет, принадлежит ли URL тому же домену, что и базовый URL.

        Args:
            url: URL для проверки

        Returns:
            bool: True, если URL принадлежит тому же домену, иначе False
        """
        netloc = urlparse(url).netloc
        return (
            netloc == self.base_netloc
            or netloc == f"www.{self.base_netloc}"
            or f"www.{netloc}" == self.base_netloc
        )

    def _is_allowed_by_robots(self, url: str) -> bool:
        """
        Проверяет, разрешен ли URL правилами robots.txt.

        Args:
            url: URL для проверки

        Returns:
            bool: True, если URL разрешен, иначе False
        """
        # Упрощенная реализация для демонстрации
        # В реальной системе здесь был бы код проверки robots.txt
        return True
