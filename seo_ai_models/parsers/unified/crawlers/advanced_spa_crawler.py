"""
Продвинутый SPA-краулер с поддержкой WebSocket, GraphQL и клиентской маршрутизации.
Расширяет возможности парсинга современных веб-приложений.
"""

import logging
import asyncio
import time
import json
import re
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from urllib.parse import urlparse, urljoin

import playwright
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError, Response, Request

from seo_ai_models.parsers.unified.js_processing.websocket_analyzer import WebSocketAnalyzer
from seo_ai_models.parsers.unified.js_processing.graphql_interceptor import GraphQLInterceptor
from seo_ai_models.parsers.unified.js_processing.client_routing_handler import ClientRoutingHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedSPACrawler:
    """
    Продвинутый краулер для SPA (Single Page Applications) с расширенными возможностями.
    Поддерживает анализ WebSocket, GraphQL и клиентскую маршрутизацию.
    """
    
    def __init__(
        self,
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        delay: float = 1.0,
        respect_robots: bool = True,
        user_agent: str = "SEOAIModels AdvancedSPACrawler/1.0",
        browser_type: str = "chromium",
        headless: bool = True,
        wait_for_idle: int = 2000,
        wait_for_timeout: int = 10000,
        intercept_requests: bool = True,
        wait_for_selectors: Optional[List[str]] = None,
        browser_args: Optional[List[str]] = None,
        enable_websocket: bool = True,
        enable_graphql: bool = True,
        enable_client_routing: bool = True,
        emulate_user_behavior: bool = False,
        bypass_protection: bool = False,
        viewport_size: Optional[Dict[str, int]] = None
    ):
        """
        Инициализация продвинутого SPA-краулера.
        
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
            intercept_requests: Перехватывать ли запросы (AJAX, GraphQL)
            wait_for_selectors: Селекторы, ожидание которых указывает на загрузку контента
            browser_args: Дополнительные аргументы для запуска браузера
            enable_websocket: Включить анализ WebSocket
            enable_graphql: Включить анализ GraphQL
            enable_client_routing: Включить обработку клиентской маршрутизации
            emulate_user_behavior: Эмулировать поведение пользователя для обхода защиты
            bypass_protection: Использовать методы обхода защиты от ботов
            viewport_size: Размер окна просмотра (по умолчанию 1366x768)
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
        self.intercept_requests = intercept_requests
        self.wait_for_selectors = wait_for_selectors or ['.main', 'main', '#content', '#main', 'article']
        self.browser_args = browser_args or []
        self.enable_websocket = enable_websocket
        self.enable_graphql = enable_graphql
        self.enable_client_routing = enable_client_routing
        self.emulate_user_behavior = emulate_user_behavior
        self.bypass_protection = bypass_protection
        self.viewport_size = viewport_size or {"width": 1366, "height": 768}
        
        # Парсинг базового URL
        self.base_netloc = urlparse(base_url).netloc
        
        # Инициализация наборов для хранения URL
        self.crawled_urls = set()
        self.found_urls = set()
        self.failed_urls = set()
        
        # Инициализация анализаторов, если включены
        if self.enable_websocket:
            self.websocket_analyzer = WebSocketAnalyzer()
            
        if self.enable_graphql:
            self.graphql_interceptor = GraphQLInterceptor()
            
        if self.enable_client_routing:
            self.routing_handler = ClientRoutingHandler()
            
        # Для хранения перехваченных запросов
        self.intercepted_requests = {}
        
        # Для хранения найденного контента
        self.page_contents = {}
        
        logger.info(f"AdvancedSPACrawler инициализирован для {base_url} с браузером {browser_type}")
        
        if enable_websocket:
            logger.info("Включен анализ WebSocket")
        if enable_graphql:
            logger.info("Включен анализ GraphQL")
        if enable_client_routing:
            logger.info("Включена обработка клиентской маршрутизации")
        if emulate_user_behavior:
            logger.info("Включена эмуляция поведения пользователя")
        if bypass_protection:
            logger.info("Включен обход защиты от ботов")
    
    async def run_crawler(self) -> Dict[str, Any]:
        """
        Запускает сканирование SPA-сайта.
        
        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        logger.info(f"Начало сканирования SPA-сайта: {self.base_url}")
        start_time = time.time()
        
        async with async_playwright() as p:
            # Выбор типа браузера
            if self.browser_type == "firefox":
                browser_instance = p.firefox
            elif self.browser_type == "webkit":
                browser_instance = p.webkit
            else:
                browser_instance = p.chromium
                
            # Настройка браузерных аргументов для обхода обнаружения
            browser_args = self.browser_args.copy()
            if self.bypass_protection:
                browser_args.extend([
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--no-sandbox'
                ])
                
            # Запуск браузера
            browser = await browser_instance.launch(
                headless=self.headless,
                args=browser_args
            )
            
            try:
                # Настройка контекста браузера
                context_options = {
                    "user_agent": self.user_agent,
                    "viewport": self.viewport_size,
                    "ignore_https_errors": True
                }
                
                # Дополнительные опции для обхода обнаружения
                if self.bypass_protection:
                    context_options["permissions"] = ["geolocation"]
                    
                context = await browser.new_context(**context_options)
                
                # Добавляем обработчики событий для перехвата запросов
                if self.intercept_requests:
                    await context.route('**/*', self._handle_request)
                
                # Обработчик для перехвата WebSocket
                if self.enable_websocket:
                    context.on("websocket", self._handle_websocket)
                
                # Создание очереди URL
                # Структура (url, depth, parent_url)
                urls_to_visit = [(self.base_url, 0, None)]
                
                # Сканирование с заданными ограничениями
                while urls_to_visit and len(self.crawled_urls) < self.max_pages:
                    current_url, depth, parent_url = urls_to_visit.pop(0)
                    
                    # Проверяем, был ли URL уже просканирован
                    if current_url in self.crawled_urls or current_url in self.failed_urls:
                        continue
                        
                    # Проверяем принадлежность к одному домену
                    if not self._is_same_domain(current_url):
                        logger.info(f"Пропуск {current_url} (другой домен)")
                        continue
                        
                    logger.info(f"Сканирование SPA URL: {current_url} (глубина {depth})")
                    
                    try:
                        # Создаем новую страницу
                        page = await context.new_page()
                        
                        # Задаем обработчики события для отладки
                        page.on("console", lambda msg: logger.debug(f"Консоль {msg.type}: {msg.text}"))
                        page.on("pageerror", lambda err: logger.warning(f"Ошибка страницы: {err}"))
                        
                        # Обработчик для клиентской маршрутизации
                        if self.enable_client_routing:
                            await self._setup_routing_listener(page)
                        
                        # Переход по URL
                        from_url = parent_url or "about:blank"
                        await self._navigate_to_url(page, current_url, from_url)
                        
                        # Проверяем успешность загрузки
                        status = await page.evaluate("() => window.status")
                        if status and status >= 400:
                            logger.warning(f"Не удалось загрузить {current_url}: HTTP {status}")
                            self.failed_urls.add(current_url)
                            await page.close()
                            continue
                            
                        # Эмуляция поведения пользователя, если включена
                        if self.emulate_user_behavior:
                            await self._emulate_user_behavior(page)
                            
                        # Получаем HTML-контент после выполнения JavaScript
                        html_content = await page.content()
                        
                        # Сохраняем контент
                        self.page_contents[current_url] = {
                            "html": html_content,
                            "title": await page.title(),
                            "url": current_url,
                            "status": status,
                            "timestamp": time.time()
                        }
                        
                        # Определяем фреймворк маршрутизации, если включено
                        if self.enable_client_routing:
                            # Получаем доступные объекты window для определения фреймворка
                            window_objects = await page.evaluate("""
                            () => {
                                return Object.keys(window).filter(key => 
                                    key.includes('Router') || 
                                    key.includes('router') || 
                                    key.includes('Route') || 
                                    key.includes('Navigation')
                                );
                            }
                            """)
                            
                            self.routing_handler.detect_router_framework(html_content, window_objects)
                        
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
                                    # Проверяем, нет ли уже в очереди
                                    if not any(l[0] == link for l in urls_to_visit):
                                        urls_to_visit.append((link, depth + 1, current_url))
                        
                        # Закрываем страницу для освобождения ресурсов
                        await page.close()
                        
                    except Exception as e:
                        logger.error(f"Ошибка сканирования {current_url}: {str(e)}")
                        self.failed_urls.add(current_url)
                        try:
                            await page.close()
                        except Exception as e:
                            logger.debug(f"Exception: {str(e)}")
                
            finally:
                # Закрываем браузер
                await browser.close()
                
        # Формируем результат
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Сканирование SPA завершено за {duration:.2f}с")
        logger.info(f"Просканировано {len(self.crawled_urls)} URL")
        logger.info(f"Найдено {len(self.found_urls)} URL")
        logger.info(f"Не удалось просканировать {len(self.failed_urls)} URL")
        
        # Составляем полный результат с данными от анализаторов
        result = {
            "base_url": self.base_url,
            "crawled_urls": list(self.crawled_urls),
            "found_urls": list(self.found_urls),
            "failed_urls": list(self.failed_urls),
            "intercepted_requests": self.intercepted_requests,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "duration": duration,
            "timestamp": time.time()
        }
        
        # Добавляем данные о WebSocket, если включено
        if self.enable_websocket:
            result["websocket"] = {
                "statistics": self.websocket_analyzer.get_message_statistics(),
                "connections": self.websocket_analyzer.get_active_connections()
            }
            
        # Добавляем данные о GraphQL, если включено
        if self.enable_graphql:
            result["graphql"] = {
                "statistics": self.graphql_interceptor.get_statistics(),
                "data": self.graphql_interceptor.get_extracted_data()
            }
            
        # Добавляем данные о маршрутизации, если включено
        if self.enable_client_routing:
            result["routing"] = {
                "statistics": self.routing_handler.get_route_statistics(),
                "history": self.routing_handler.get_route_history(),
                "unique_routes": self.routing_handler.get_unique_routes()
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
    
    async def _navigate_to_url(self, page: Page, url: str, from_url: str) -> None:
        """
        Выполняет навигацию по URL с обработкой ожидания загрузки.
        
        Args:
            page: Объект страницы Playwright
            url: URL для перехода
            from_url: URL, с которого выполняется переход
        """
        try:
            # Регистрируем изменение маршрута, если включено
            if self.enable_client_routing:
                self.routing_handler.record_route_change(from_url, url)
            
            # Переходим по URL и ждем загрузки
            response = await page.goto(
                url, 
                wait_until="networkidle",
                timeout=self.wait_for_timeout
            )
            
            # Дополнительное ожидание для элементов контента
            try:
                # Ожидаем появления селекторов, указывающих на загрузку контента
                for selector in self.wait_for_selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=self.wait_for_idle // 2)
                        break  # Если нашли хотя бы один селектор, выходим
                    except:
                        continue
            except:
                # Если не смогли дождаться селекторов, просто продолжаем
                logger.debug(f"Селекторы контента не найдены для {url}")
                
            # Ожидаем дополнительное время для полной загрузки
            await asyncio.sleep(self.delay)
            
        except Exception as e:
            logger.error(f"Ошибка при навигации к {url}: {str(e)}")
            raise
    
    async def _handle_request(self, route, request):
        """
        Обрабатывает перехват запросов.
        
        Args:
            route: Объект маршрута для перехвата запросов
            request: Перехваченный запрос
        """
        url = request.url
        method = request.method
        resource_type = request.resource_type
        
        # Позволяем запросу продолжить выполнение
        await route.continue_()
        
        # Анализируем запрос
        try:
            # Сохраняем базовую информацию о запросе
            if url not in self.intercepted_requests:
                self.intercepted_requests[url] = {
                    "method": method,
                    "url": url,
                    "resource_type": resource_type,
                    "timestamp": time.time()
                }
            
            # Проверяем, является ли это GraphQL-запросом
            if self.enable_graphql and (
                resource_type == "fetch" or resource_type == "xhr" or
                url.endswith("/graphql") or "graphql" in url
            ):
                try:
                    # Получаем заголовки и тело запроса
                    headers = request.headers
                    post_data = request.post_data
                    
                    if post_data:
                        # Пытаемся распарсить как GraphQL
                        try:
                            body = json.loads(post_data)
                            # Проверяем признаки GraphQL-запроса
                            if (isinstance(body, dict) and 
                                ("query" in body or "operationName" in body or "variables" in body)):
                                
                                # Перехватываем GraphQL-запрос
                                operations = self.graphql_interceptor.intercept_request(url, headers, body)
                                
                                # Сохраняем связь между запросом и операциями
                                self.intercepted_requests[url]["graphql"] = {
                                    "operations": [op.to_dict() for op in operations],
                                    "timestamp": time.time()
                                }
                                
                        except Exception as e:
                            logger.debug(f"Exception: {str(e)}")
                except Exception as e:
                    logger.debug(f"Exception: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса {url}: {str(e)}")
    
    async def _handle_response(self, response: Response):
        """
        Обрабатывает перехват ответов.
        
        Args:
            response: Перехваченный ответ
        """
        url = response.url
        status = response.status
        
        # Анализируем ответ
        try:
            # Проверяем, является ли это ответом на GraphQL-запрос
            if self.enable_graphql and url in self.intercepted_requests and "graphql" in self.intercepted_requests[url]:
                try:
                    # Получаем тело ответа
                    body_text = await response.text()
                    
                    # Получаем связанные операции
                    operations = []
                    if "graphql" in self.intercepted_requests[url]:
                        graphql_data = self.intercepted_requests[url]["graphql"]
                        if "operations" in graphql_data:
                            operations = graphql_data["operations"]
                    
                    # Пытаемся распарсить как GraphQL-ответ
                    try:
                        body = json.loads(body_text)
                        
                        # Проверяем признаки GraphQL-ответа
                        if isinstance(body, dict) and ("data" in body or "errors" in body):
                            # Перехватываем GraphQL-ответ
                            self.graphql_interceptor.intercept_response(body, None, status)
                            
                    except Exception as e:
                        logger.debug(f"Exception: {str(e)}")
                except Exception as e:
                    logger.debug(f"Exception: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Ошибка при обработке ответа {url}: {str(e)}")
    
    async def _handle_websocket(self, websocket):
        """
        Обрабатывает WebSocket-соединения.
        
        Args:
            websocket: Объект WebSocket
        """
        if not self.enable_websocket:
            return
            
        url = websocket.url
        logger.info(f"Установлено WebSocket-соединение: {url}")
        
        # Регистрируем обработчики сообщений
        websocket.on("framesent", lambda payload: self._handle_websocket_frame(payload, "outgoing", url))
        websocket.on("framereceived", lambda payload: self._handle_websocket_frame(payload, "incoming", url))
        
        # Ожидаем закрытия
        await websocket.wait_for_event("close")
        logger.info(f"WebSocket-соединение закрыто: {url}")
    
    def _handle_websocket_frame(self, payload, direction, url):
        """
        Обрабатывает фреймы WebSocket.
        
        Args:
            payload: Данные фрейма
            direction: Направление ('incoming' или 'outgoing')
            url: URL WebSocket-соединения
        """
        if not self.enable_websocket:
            return
            
        try:
            # Проверяем, является ли это текстовым сообщением
            if hasattr(payload, 'text'):
                self.websocket_analyzer.capture_message(payload.text, direction, url)
            elif hasattr(payload, 'bytes'):
                # Пытаемся декодировать байты как UTF-8
                try:
                    text = payload.bytes.decode('utf-8')
                    self.websocket_analyzer.capture_message(text, direction, url)
                except:
                    logger.debug(f"Невозможно декодировать бинарный WebSocket-фрейм: {url}")
        except Exception as e:
            logger.error(f"Ошибка при обработке WebSocket-фрейма {url}: {str(e)}")
    
    async def _setup_routing_listener(self, page: Page):
        """
        Настраивает слушатель изменений маршрутов для клиентской маршрутизации.
        
        Args:
            page: Объект страницы Playwright
        """
        if not self.enable_client_routing:
            return
            
        # Внедряем JavaScript для отслеживания изменений History API
        await page.evaluate("""
        () => {
            // Сохраняем ссылки на оригинальные методы
            const originalPushState = window.history.pushState;
            const originalReplaceState = window.history.replaceState;
            
            // Создаем кастомное событие для изменения маршрута
            const createRouteChangeEvent = (from, to, state) => {
                const event = new CustomEvent('spa-route-change', {
                    detail: { from, to, state }
                });
                document.dispatchEvent(event);
            };
            
            // Перехватываем pushState
            window.history.pushState = function(state, title, url) {
                const from = window.location.href;
                const to = new URL(url, window.location.origin).href;
                
                originalPushState.apply(this, [state, title, url]);
                createRouteChangeEvent(from, to, state);
            };
            
            // Перехватываем replaceState
            window.history.replaceState = function(state, title, url) {
                const from = window.location.href;
                const to = url ? new URL(url, window.location.origin).href : from;
                
                originalReplaceState.apply(this, [state, title, url]);
                createRouteChangeEvent(from, to, state);
            };
            
            // Перехватываем событие popstate
            window.addEventListener('popstate', function(e) {
                const from = document._lastLocation || window.location.href;
                const to = window.location.href;
                
                createRouteChangeEvent(from, to, e.state);
                document._lastLocation = to;
            });
            
            // Перехватываем click на ссылках
            document.addEventListener('click', function(e) {
                const link = e.target.closest('a');
                if (link && link.href) {
                    document._lastLocation = window.location.href;
                }
            }, true);
            
            // Сохраняем текущий URL
            document._lastLocation = window.location.href;
        }
        """)
        
        # Настраиваем обработчик события изменения маршрута
        await page.evaluate("""
        () => {
            document.addEventListener('spa-route-change', function(e) {
                // Сохраняем данные о маршруте в объекте window для доступа из Playwright
                window.__routeChangeInfo = e.detail;
                console.debug('SPA Route Change:', e.detail);
            });
        }
        """)
        
        # Подписка на изменения маршрута через console.debug
        page.on("console", self._handle_console_message)
    
    def _handle_console_message(self, message):
        """
        Обрабатывает сообщения консоли для отслеживания изменений маршрута.
        
        Args:
            message: Сообщение консоли
        """
        if not self.enable_client_routing:
            return
            
        try:
            # Проверяем, является ли это сообщением об изменении маршрута
            if message.type == "debug" and "SPA Route Change:" in message.text:
                # Извлекаем информацию о маршруте из аргументов
                if len(message.args) >= 2:
                    route_data = message.args[1].json_value()
                    if isinstance(route_data, dict) and "from" in route_data and "to" in route_data:
                        from_url = route_data["from"]
                        to_url = route_data["to"]
                        state = route_data.get("state")
                        
                        # Регистрируем изменение маршрута
                        self.routing_handler.record_route_change(from_url, to_url, state)
        except Exception as e:
            logger.debug(f"Exception: {str(e)}")
    
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
                rel: a.rel,
                target: a.target,
                onclick: a.hasAttribute('onclick')
            }))
        """
        links_data = await page.evaluate(links_js)
        
        links = set()
        
        for link in links_data:
            href = link.get('href', '')
            
            # Пропускаем пустые ссылки, якоря и JavaScript
            if (not href or href.startswith('#') or 
                href.startswith('javascript:') or 
                href.startswith('mailto:') or 
                href.startswith('tel:')):
                continue
                
            # Преобразуем относительные URL в абсолютные
            absolute_url = urljoin(current_url, href)
            
            # Удаляем якоря и приводим к нижнему регистру
            absolute_url = absolute_url.split('#')[0].lower()
            
            # Проверяем, что URL принадлежит тому же домену
            if self._is_same_domain(absolute_url):
                links.add(absolute_url)
        
        # Если включена клиентская маршрутизация, ищем также динамические ссылки
        if self.enable_client_routing:
            # Найдем элементы, которые могут быть клиентскими ссылками (имеющие onClick, role="link", и т.д.)
            potential_links_js = """
            Array.from(document.querySelectorAll('[onclick], [role="link"], [data-route], [data-link], [data-href]'))
                .filter(el => !el.matches('a[href]')) // Исключаем обычные ссылки, которые мы уже обработали
                .map(el => {
                    const potentialUrl = el.getAttribute('data-route') || 
                                        el.getAttribute('data-link') || 
                                        el.getAttribute('data-href') || 
                                        el.getAttribute('href') || 
                                        '';
                    
                    return {
                        element: el.tagName,
                        text: el.innerText.trim(),
                        href: potentialUrl,
                        isClientLink: true
                    };
                })
                .filter(item => item.href && !item.href.startsWith('#') && 
                              !item.href.startsWith('javascript:') && 
                              !item.href.startsWith('mailto:') && 
                              !item.href.startsWith('tel:'))
            """
            
            potential_links = await page.evaluate(potential_links_js)
            
            for link in potential_links:
                href = link.get('href', '')
                if href:
                    # Преобразуем относительные URL в абсолютные
                    absolute_url = urljoin(current_url, href)
                    
                    # Удаляем якоря и приводим к нижнему регистру
                    absolute_url = absolute_url.split('#')[0].lower()
                    
                    # Проверяем, что URL принадлежит тому же домену
                    if self._is_same_domain(absolute_url):
                        links.add(absolute_url)
                
        return links
    
    async def _emulate_user_behavior(self, page: Page) -> None:
        """
        Эмулирует поведение пользователя для обхода защиты от ботов.
        
        Args:
            page: Объект страницы Playwright
        """
        if not self.emulate_user_behavior:
            return
            
        try:
            # Получаем размер окна
            viewport_size = await page.evaluate("""
            () => {
                return {
                    width: window.innerWidth,
                    height: window.innerHeight
                };
            }
            """)
            
            # Случайное движение мыши
            for _ in range(3):
                x = viewport_size["width"] * 0.1 + (viewport_size["width"] * 0.8 * (0.2 + 0.6 * (time.time() % 1)))
                y = viewport_size["height"] * 0.1 + (viewport_size["height"] * 0.8 * (0.3 + 0.4 * (time.time() % 1)))
                
                await page.mouse.move(x, y)
                await asyncio.sleep(0.1 + 0.2 * (time.time() % 1))
                
            # Прокрутка страницы
            await page.mouse.wheel(0, 300)
            await asyncio.sleep(0.5)
            await page.mouse.wheel(0, 200)
            await asyncio.sleep(0.3)
            await page.mouse.wheel(0, -100)
            
            # Фокус на случайном элементе ввода (если есть)
            try:
                input_selectors = ['input', 'textarea', 'select', '[tabindex]']
                for selector in input_selectors:
                    elements = await page.query_selector_all(selector)
                    if elements and len(elements) > 0:
                        # Выбираем случайный элемент
                        element = elements[int(time.time() * 1000) % len(elements)]
                        await element.focus()
                        await asyncio.sleep(0.2)
                        break
            except Exception as e:
                logger.debug(f"Exception: {str(e)}")
                
        except Exception as e:
            logger.warning(f"Ошибка при эмуляции поведения пользователя: {str(e)}")
    
    def _is_same_domain(self, url: str) -> bool:
        """
        Проверяет, принадлежит ли URL тому же домену, что и базовый URL.
        
        Args:
            url: URL для проверки
            
        Returns:
            bool: True, если URL принадлежит тому же домену, иначе False
        """
        try:
            netloc = urlparse(url).netloc
            return netloc == self.base_netloc or netloc == f"www.{self.base_netloc}" or f"www.{netloc}" == self.base_netloc
        except:
            return False
