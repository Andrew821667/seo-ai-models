"""
EnhancedSPACrawlerLLM - улучшенный краулер для SPA с оптимизацией под LLM
с поддержкой русскоязычного контента.

Модуль предоставляет расширенные возможности для краулинга одностраничных приложений (SPA)
с оптимизацией для извлечения контента, важного для LLM-систем.
"""

import logging
import time
import re
import json
import os
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
from urllib.parse import urljoin, urlparse, unquote
import hashlib
import requests
from bs4 import BeautifulSoup, Comment
import langdetect
from collections import defaultdict

# Безопасный импорт playwright
try:
    from playwright.sync_api import (
        sync_playwright,
        Page,
        Browser,
        BrowserContext,
        ElementHandle,
        Response,
        TimeoutError as PlaywrightTimeoutError,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logging.warning(
        "Playwright не установлен. EnhancedSPACrawlerLLM будет работать в ограниченном режиме."
    )
    PLAYWRIGHT_AVAILABLE = False

# Настраиваем логгер
logger = logging.getLogger(__name__)


class EnhancedSPACrawlerLLM:
    """
    Расширенный краулер для SPA (Single Page Applications) с оптимизацией для LLM.

    Добавляет следующие улучшения:
    1. Продвинутая обработка динамического контента
    2. Извлечение скрытых элементов, важных для LLM
    3. Распознавание и сохранение семантической структуры
    4. Управление историей состояний SPA
    5. Интеллектуальное определение завершения загрузки контента
    6. Оптимизация и фильтрация контента для LLM-анализа
    7. Адаптация для русскоязычного контента
    8. Извлечение структурированных данных (Schema.org, Open Graph и т.д.)
    9. Обнаружение и обработка AJAX-запросов
    10. Интеллектуальная навигация по интерфейсу SPA
    """

    def __init__(
        self,
        headless: bool = True,
        wait_for_network_idle: bool = True,
        wait_time: float = 5.0,
        scroll_for_lazy_loading: bool = True,
        extract_hidden_content: bool = True,
        preserve_semantic_structure: bool = True,
        llm_content_optimization: bool = True,
        language: str = "ru",
        track_ajax_requests: bool = True,
        extract_structured_data: bool = True,
        browser_type: str = "chromium",
        device_scale_factor: float = 1.0,
        user_agent: Optional[str] = None,
        viewport: Optional[Dict[str, int]] = None,
        max_request_count: int = 100,
        request_timeout: int = 30000,
        max_navigation_time: int = 60000,
        ignore_https_errors: bool = False,
        javascript_enabled: bool = True,
        **kwargs,
    ):
        """
        Инициализирует EnhancedSPACrawlerLLM с расширенными опциями.

        Args:
            headless: Запуск браузера в безголовом режиме (без UI).
            wait_for_network_idle: Ожидать завершения сетевых запросов.
            wait_time: Время ожидания после загрузки страницы (секунды).
            scroll_for_lazy_loading: Прокручивать страницу для загрузки ленивого контента.
            extract_hidden_content: Извлекать скрытые элементы, важные для LLM.
            preserve_semantic_structure: Сохранять семантическую структуру.
            llm_content_optimization: Оптимизировать контент для LLM-анализа.
            language: Язык контента ('ru' для русского, 'en' для английского).
            track_ajax_requests: Отслеживать AJAX-запросы.
            extract_structured_data: Извлекать структурированные данные (Schema.org и т.д.).
            browser_type: Тип браузера ('chromium', 'firefox', 'webkit').
            device_scale_factor: Масштабирование устройства.
            user_agent: Пользовательский агент (если None, используется по умолчанию).
            viewport: Размеры окна браузера (словарь с ключами width и height).
            max_request_count: Максимальное количество запросов.
            request_timeout: Таймаут запросов (миллисекунды).
            max_navigation_time: Максимальное время навигации (миллисекунды).
            ignore_https_errors: Игнорировать ошибки HTTPS.
            javascript_enabled: Включить JavaScript.
            **kwargs: Дополнительные параметры.
        """
        # Основные настройки
        self.headless = headless
        self.wait_for_network_idle = wait_for_network_idle
        self.wait_time = wait_time
        self.scroll_for_lazy_loading = scroll_for_lazy_loading
        self.extract_hidden_content = extract_hidden_content
        self.preserve_semantic_structure = preserve_semantic_structure
        self.llm_content_optimization = llm_content_optimization
        self.language = language
        self.track_ajax_requests = track_ajax_requests
        self.extract_structured_data = extract_structured_data

        # Параметры браузера
        self.browser_type = browser_type
        self.device_scale_factor = device_scale_factor
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        )
        self.viewport = viewport or {"width": 1920, "height": 1080}

        # Лимиты и таймауты
        self.max_request_count = max_request_count
        self.request_timeout = request_timeout
        self.max_navigation_time = max_navigation_time
        self.ignore_https_errors = ignore_https_errors
        self.javascript_enabled = javascript_enabled

        # Внутренние переменные
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.ajax_requests = []
        self.current_url = None
        self.navigation_history = []
        self.visited_links = set()
        self.content_cache = {}
        self.last_headers = {}
        self.console_messages = []

        # Селекторы для LLM-оптимизированного контента
        # Адаптированы для русскоязычного контента
        self.semantic_selectors = {
            "main_content": [
                "main",
                "article",
                '[role="main"]',
                "#content",
                ".content",
                ".main-content",
                ".основной-контент",
                ".статья",
                ".содержание",
                ".текст",
                ".body",
                ".post",
                ".entry",
                ".page",
                ".page-content",
                '[itemprop="mainContentOfPage"]',
                '[itemprop="articleBody"]',
                "#main",
                "#article",
                ".main",
                ".article",
            ],
            "heading_elements": [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                ".headline",
                ".title",
                ".header",
                ".heading",
                ".subheading",
                ".subtitle",
                ".заголовок",
                ".подзаголовок",
            ],
            "navigation": [
                "nav",
                '[role="navigation"]',
                ".navigation",
                ".menu",
                ".navbar",
                ".nav",
                ".main-menu",
                ".навигация",
                ".меню",
                ".главное-меню",
                "#menu",
                "#nav",
                "#navigation",
                "#navbar",
                "header",
                ".header",
                "#header",
                "ul.menu",
                ".top-menu",
                ".primary-menu",
            ],
            "footer": [
                "footer",
                ".footer",
                "#footer",
                '[role="contentinfo"]',
                ".подвал",
                ".футер",
                ".нижняя-часть",
                ".bottom",
                ".site-footer",
                ".page-footer",
            ],
            "aside": [
                "aside",
                ".sidebar",
                ".widget-area",
                '[role="complementary"]',
                ".боковая-панель",
                ".виджеты",
                ".дополнительно",
                "#sidebar",
                ".right-column",
                ".left-column",
            ],
            "interactive_elements": [
                "button",
                '[role="button"]',
                "a.button",
                ".btn",
                '[type="button"]',
                '[type="submit"]',
                ".cta",
                ".call-to-action",
                ".кнопка",
                ".действие",
                ".ссылка-кнопка",
                "details",
                "summary",
                "dialog",
                '[role="dialog"]',
                ".modal",
                ".popup",
                ".модальное-окно",
                "form",
                '[role="form"]',
                ".form",
                ".форма",
                ".контактная-форма",
                ".заказ",
                ".опрос",
                "input",
                "select",
                "textarea",
                ".input",
                ".поле",
                ".ввод",
                ".выбор",
                ".список",
            ],
            "important_list_elements": [
                "ul.important",
                "ol.important",
                ".list-important",
                ".важный-список",
                ".features",
                ".benefits",
                ".steps",
                ".instructions",
                ".преимущества",
                ".шаги",
                ".инструкции",
                ".faq",
                ".часто-задаваемые-вопросы",
                ".вопросы-ответы",
                ".qa",
                "details",
                "summary",
                '[role="list"]',
                ".list-unstyled",
            ],
            "schema_elements": [
                "[itemscope]",
                "[itemprop]",
                "[itemtype]",
                'script[type="application/ld+json"]',
                'meta[property^="og:"]',
                'meta[name^="twitter:"]',
                "meta[itemprop]",
            ],
            "important_hidden_elements": [
                "noscript",
                '[aria-hidden="true"]',
                "[hidden]",
                ".visually-hidden",
                ".sr-only",
                ".d-none",
                ".hidden",
                '[style*="display: none"]',
                '[style*="display:none"]',
                '[style*="visibility: hidden"]',
                '[style*="visibility:hidden"]',
                ".скрытый",
                ".невидимый",
                ".visible-on-focus",
            ],
        }

        # Регулярные выражения для идентификации контента
        self.content_patterns = {
            "json_ld": re.compile(
                r'<script[^>]*type=[\'"]application/ld\+json[\'"][^>]*>(.*?)</script>', re.DOTALL
            ),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}"),
            "price": re.compile(
                r"\b(?:\d{1,3}[., ]?)*\d{1,3}(?:[.,]\d{2})?\s*(?:[₽руб$€£¥₴]|руб(?:лей)?|USD|EUR|RUB)\b",
                re.IGNORECASE,
            ),
            "date": re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
            "html_comment": re.compile(r"<!--.*?-->", re.DOTALL),
            "internal_link": re.compile(
                r'<a[^>]*href=[\'"](?!(?:https?:|tel:|mailto:|#|javascript:|\s*$))[^\'"]*[\'"][^>]*>'
            ),
            "external_link": re.compile(r'<a[^>]*href=[\'"](?:https?:)[^\'"]*[\'"][^>]*>'),
        }

        # Расширенные настройки из kwargs
        self.extra_headers = kwargs.get("extra_headers", {})
        self.cookies = kwargs.get("cookies", [])
        self.scenario = kwargs.get("scenario", [])
        self.max_depth = kwargs.get("max_depth", 1)
        self.max_pages = kwargs.get("max_pages", 10)
        self.follow_links = kwargs.get("follow_links", False)
        self.allowed_domains = kwargs.get("allowed_domains", [])
        self.excluded_patterns = kwargs.get("excluded_patterns", [])
        self.included_patterns = kwargs.get("included_patterns", [])
        self.proxy = kwargs.get("proxy", None)
        self.local_storage = kwargs.get("local_storage", {})
        self.session_storage = kwargs.get("session_storage", {})

        # Инициализация краулера
        self._init_crawler()

    def _init_crawler(self):
        """Инициализирует компоненты краулера."""
        # Инициализируем playwright если еще не инициализирован
        if self.playwright is None:
            self._start_browser()

        # Инициализируем параметры
        self.browser_content = None
        self.navigation_history = []
        self.extracted_links = set()
        self.processed_links = set()
        self.resources_info = {}
        self.requests_log = []
        self.dom_changes = []

        # Инициализируем компоненты для извлечения данных
        self.structured_data_extractor = None
        self.metadata_enhancer = None

        # Настройка логгера для сбора данных о работе краулера
        logger.info(
            f"Краулер инициализирован с параметрами: headless={self.headless}, "
            f"wait_for_network_idle={self.wait_for_network_idle}, wait_time={self.wait_time}"
        )

    # Улучшенная обработка исключения при определении языка
    def improved_language_detection_exception_handling(self, text_sample):
        try:
            lang_code = langdetect.detect(text_sample)
            return lang_code
        except langdetect.LangDetectException as e:
            logger.warning(f"Ошибка LangDetect при определении языка: {str(e)}")
            # Альтернативный способ определения языка - по частоте встречаемости характерных символов
            ru_chars = len(re.findall("[а-яА-Я]", text_sample))
            en_chars = len(re.findall("[a-zA-Z]", text_sample))

        if ru_chars > en_chars * 0.3:  # если русских символов больше 30% от английских
            self.logger.info("Язык определен альтернативным способом: ru")
            return "ru"
        elif en_chars > 0:
            self.logger.info("Язык определен альтернативным способом: en")
            return "en"

        # Если не удалось определить язык, используем язык по умолчанию
        self.logger.info(
            f"Не удалось определить язык, используется язык по умолчанию: {self.language}"
        )
        return self.language

    def crawl_url(self, url: str, depth: int = 0) -> Dict[str, Any]:
        """
        Выполняет краулинг SPA-страницы с оптимизацией для LLM.

        Args:
            url: URL для краулинга
            depth: Текущая глубина краулинга (для рекурсивного краулинга)

        Returns:
            Dict[str, Any]: Результаты краулинга с дополнительными данными для LLM
        """
        # Проверка наличия Playwright
        if not PLAYWRIGHT_AVAILABLE:
            return {"error": "Playwright не установлен", "url": url, "llm_language": self.language}

        # Инициализация Playwright и браузера, если еще не инициализированы
        self._start_browser()

        try:
            # Сохраняем текущий URL
            self.current_url = url
            self.navigation_history.append(url)

            # Отслеживаем AJAX-запросы, если включено
            if self.track_ajax_requests:
                self._setup_request_tracking()

            # Настраиваем прослушивание консоли
            self._setup_console_listener()

            # Переходим на страницу с расширенным ожиданием загрузки
            self._navigate_to_url(url)

            # Прокручиваем страницу для загрузки ленивого контента, если включено
            if self.scroll_for_lazy_loading:
                self._scroll_for_lazy_content()

            # Определяем язык страницы, если не указан явно
            detected_language = self._detect_language()

            # Выполняем сценарий взаимодействия, если указан
            if self.scenario:
                self._execute_scenario()

            # Извлекаем структурированные данные, если включено
            structured_data = {}
            if self.extract_structured_data:
                structured_data = self._extract_structured_data()

            # Извлекаем контент страницы с учетом семантической структуры
            content = self._extract_content_with_structure()

            # Оптимизируем контент для LLM, если включено
            if self.llm_content_optimization:
                content = self._optimize_content_for_llm(content)

            # Формируем результат
            result = {
                "url": url,
                "title": self._get_page_title(),
                "detected_language": detected_language,
                "content": content,
                "llm_optimized": self.llm_content_optimization,
                "structured_data": structured_data,
                "meta_data": self._extract_meta_data(),
                "timestamp": time.time(),
            }

            # Добавляем информацию об AJAX-запросах, если отслеживались
            if self.track_ajax_requests:
                result["ajax_requests"] = self._process_ajax_requests()

            # Добавляем консольные сообщения
            if self.console_messages:
                result["console_messages"] = self._filter_console_messages()

            # Рекурсивный краулинг по ссылкам, если включено
            if self.follow_links and depth < self.max_depth:
                result["linked_pages"] = self._crawl_links(depth + 1)

            return result
        except Exception as e:
            logger.error(f"Ошибка при краулинге URL {url}: {str(e)}")
            return {"error": str(e), "url": url, "llm_language": self.language}
        finally:
            # Не закрываем браузер, если нужно продолжить работу
            if depth == 0:
                self._close_browser()

    def _start_browser(self):
        """Запускает браузер и создает контекст."""
        if self.playwright is None:
            self.playwright = sync_playwright().start()

            # Выбираем тип браузера
            if self.browser_type == "firefox":
                self.browser = self.playwright.firefox.launch(
                    headless=self.headless, ignore_https_errors=self.ignore_https_errors
                )
            elif self.browser_type == "webkit":
                self.browser = self.playwright.webkit.launch(
                    headless=self.headless, ignore_https_errors=self.ignore_https_errors
                )
            else:  # По умолчанию chromium
                self.browser = self.playwright.chromium.launch(
                    headless=self.headless, ignore_https_errors=self.ignore_https_errors
                )

            # Создаем контекст браузера с расширенными параметрами
            context_params = {
                "viewport": self.viewport,
                "device_scale_factor": self.device_scale_factor,
                "user_agent": self.user_agent,
                "locale": self.language,
                "extra_http_headers": self.extra_headers,
            }

            # Добавляем прокси, если указан
            if self.proxy:
                context_params["proxy"] = self.proxy

            self.context = self.browser.new_context(**context_params)

            # Устанавливаем cookies, если указаны
            if self.cookies:
                self.context.add_cookies(self.cookies)

            # Создаем новую страницу
            self.page = self.context.new_page()

            # Настраиваем хранилище, если указано
            if self.local_storage:
                self._setup_storage("localStorage", self.local_storage)
            if self.session_storage:
                self._setup_storage("sessionStorage", self.session_storage)

    def _setup_storage(self, storage_type: str, data: Dict[str, str]):
        """
        Настраивает localStorage или sessionStorage.

        Args:
            storage_type: Тип хранилища ('localStorage' или 'sessionStorage')
            data: Данные для хранилища
        """
        script = f"() => {{ const storage = window.{storage_type}; "
        for key, value in data.items():
            script += f"storage.setItem('{key}', '{value}'); "
        script += "return Object.keys(storage).length; };"

        self.page.evaluate(script)

    def _setup_request_tracking(self):
        """Настраивает отслеживание сетевых запросов."""
        self.ajax_requests = []

        # Обработчик для отслеживания запросов
        def on_request(request):
            if request.resource_type in ["xhr", "fetch"]:
                self.ajax_requests.append(
                    {
                        "url": request.url,
                        "method": request.method,
                        "headers": request.headers,
                        "time": time.time(),
                        "resource_type": request.resource_type,
                        "response": None,
                    }
                )

        # Обработчик для отслеживания ответов
        def on_response(response):
            request = response.request
            if request.resource_type in ["xhr", "fetch"]:
                # Находим соответствующий запрос
                for ajax_request in self.ajax_requests:
                    if ajax_request["url"] == request.url and ajax_request["response"] is None:
                        # Сохраняем информацию о ответе
                        ajax_request["response"] = {
                            "status": response.status,
                            "status_text": response.status_text,
                            "headers": response.headers,
                            "time": time.time(),
                        }

                        # Пытаемся получить содержимое ответа, если это возможно
                        try:
                            # Для текстовых ответов
                            if "json" in response.headers.get("content-type", "").lower():
                                ajax_request["response"]["body"] = response.json()
                            elif "text" in response.headers.get("content-type", "").lower():
                                ajax_request["response"]["body"] = response.text()
                        except Exception as e:
                            logger.debug(f"Не удалось получить содержимое ответа: {str(e)}")

                        break

        # Устанавливаем обработчики
        self.page.on("request", on_request)
        self.page.on("response", on_response)

    def _setup_console_listener(self):
        """Настраивает прослушивание консольных сообщений."""
        self.console_messages = []

        def on_console(msg):
            self.console_messages.append(
                {"type": msg.type, "text": msg.text, "time": time.time(), "location": msg.location}
            )

        self.page.on("console", on_console)

    def _navigate_to_url(self, url: str):
        """
        Переходит на указанный URL с расширенной обработкой загрузки.

        Args:
            url: URL для перехода
        """
        # Устанавливаем таймауты
        self.page.set_default_timeout(self.max_navigation_time)

        try:
            # Переходим на страницу
            response = self.page.goto(
                url,
                wait_until="networkidle" if self.wait_for_network_idle else "load",
                timeout=self.max_navigation_time,
            )

            # Сохраняем заголовки ответа
            if response:
                self.last_headers = response.headers

                # Проверяем статус ответа
                if response.status >= 400:
                    logger.warning(
                        f"Получен статус ошибки при загрузке {url}: {response.status} {response.status_text}"
                    )

            # Дополнительное ожидание после загрузки страницы
            if self.wait_time > 0:
                self.page.wait_for_timeout(int(self.wait_time * 1000))

            # Проверяем интеллектуальное определение завершения загрузки
            self._wait_for_content_loaded()
        except PlaywrightTimeoutError:
            logger.warning(f"Таймаут при загрузке {url}")
        except Exception as e:
            logger.error(f"Ошибка при навигации на {url}: {str(e)}")

    def _wait_for_content_loaded(self):
        """
        Интеллектуальное определение завершения загрузки контента.
        Ожидает, пока страница не будет полностью загружена.
        """
        try:
            # Ожидаем загрузки основного содержимого
            for selector in self.semantic_selectors["main_content"]:
                try:
                    self.page.wait_for_selector(selector, timeout=5000)
                    logger.debug(f"Найден основной контент по селектору: {selector}")
                    break
                except:
                    continue

            # Проверяем стабилизацию DOM
            prev_dom_size = 0
            stable_count = 0
            max_checks = 5
            check_interval = 1000  # 1 секунда

            for _ in range(max_checks):
                # Получаем размер DOM
                dom_size = self.page.evaluate(
                    """() => {
                    return document.documentElement.outerHTML.length;
                }"""
                )

                logger.debug(f"Размер DOM: {dom_size}, предыдущий: {prev_dom_size}")

                # Если размер DOM не изменился значительно, увеличиваем счетчик стабильности
                if abs(dom_size - prev_dom_size) < dom_size * 0.01:  # Менее 1% изменений
                    stable_count += 1
                else:
                    stable_count = 0

                # Если DOM стабилен в течение достаточного количества проверок, считаем загрузку завершенной
                if stable_count >= 2:
                    logger.debug("DOM стабилизировался, загрузка завершена")
                    break

                prev_dom_size = dom_size
                self.page.wait_for_timeout(check_interval)
        except Exception as e:
            logger.error(f"Ошибка при ожидании загрузки контента: {str(e)}")

    def _scroll_for_lazy_content(self):
        """
        Прокручивает страницу для загрузки ленивого контента.
        """
        try:
            # Получаем высоту страницы
            page_height = self.page.evaluate(
                """() => {
                return Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.offsetHeight,
                    document.body.clientHeight,
                    document.documentElement.clientHeight
                );
            }"""
            )

            # Прокручиваем страницу постепенно
            view_port_height = self.viewport["height"]
            scroll_step = view_port_height // 2
            current_position = 0

            while current_position < page_height:
                # Прокручиваем на шаг вниз
                self.page.evaluate(f"window.scrollTo(0, {current_position});")
                current_position += scroll_step

                # Небольшая задержка для загрузки контента
                self.page.wait_for_timeout(300)

                # Проверяем наличие бесконечной прокрутки
                new_height = self.page.evaluate(
                    """() => {
                    return Math.max(
                        document.body.scrollHeight,
                        document.documentElement.scrollHeight,
                        document.body.offsetHeight,
                        document.documentElement.offsetHeight,
                        document.body.clientHeight,
                        document.documentElement.clientHeight
                    );
                }"""
                )

                # Если высота увеличилась, обновляем общую высоту
                if new_height > page_height:
                    page_height = new_height

                # Ограничиваем максимальную высоту прокрутки
                if current_position > 15000:  # Ограничиваем максимальную высоту прокрутки
                    break

            # Прокручиваем наверх
            self.page.evaluate("window.scrollTo(0, 0);")
            self.page.wait_for_timeout(300)
        except Exception as e:
            logger.error(f"Ошибка при прокрутке страницы: {str(e)}")

    def _execute_scenario(self):
        """
        Выполняет сценарий взаимодействия с страницей.
        Сценарий представляет собой список действий, которые нужно выполнить.
        """
        for step in self.scenario:
            try:
                action = step.get("action", "").lower()
                selector = step.get("selector", "")
                value = step.get("value", "")
                timeout = step.get("timeout", 5000)

                logger.debug(
                    f"Выполнение шага сценария: {action} на {selector} со значением {value}"
                )

                # Ожидаем появления элемента, если указан селектор
                if selector:
                    self.page.wait_for_selector(selector, timeout=timeout)

                # Выполняем действие
                if action == "click":
                    self.page.click(selector)
                elif action == "fill":
                    self.page.fill(selector, value)
                elif action == "type":
                    self.page.type(selector, value)
                elif action == "select":
                    self.page.select_option(selector, value)
                elif action == "check":
                    self.page.check(selector)
                elif action == "uncheck":
                    self.page.uncheck(selector)
                elif action == "press":
                    self.page.press(selector, value)
                elif action == "hover":
                    self.page.hover(selector)
                elif action == "wait":
                    self.page.wait_for_timeout(int(value) if value.isdigit() else timeout)
                elif action == "evaluate":
                    self.page.evaluate(value)
                elif action == "screenshot":
                    self.page.screenshot(path=value if value else "screenshot.png")

                # Ожидаем после действия
                wait_after = step.get("wait_after", 1000)
                if wait_after > 0:
                    self.page.wait_for_timeout(wait_after)
            except Exception as e:
                logger.error(f"Ошибка при выполнении шага сценария: {str(e)}")
                if step.get("critical", False):
                    raise

    def _detect_language(self) -> str:
        """
        Определяет язык страницы.

        Returns:
            str: Код языка
        """
        try:
            # Сначала проверяем HTML-атрибут lang
            lang_attr = self.page.evaluate(
                """() => {
                return document.documentElement.lang || 
                       document.querySelector('html').getAttribute('lang') ||
                       '';
            }"""
            )

            if lang_attr and len(lang_attr) >= 2:
                # Извлекаем первые два символа (код языка)
                lang_code = lang_attr.split("-")[0].lower()
                if lang_code in ["ru", "en", "fr", "de", "es", "it", "zh", "ja", "ko", "ar"]:
                    return lang_code

            # Затем проверяем мета-теги
            meta_lang = self.page.evaluate(
                """() => {
                const metaLang = document.querySelector('meta[name="language"], meta[http-equiv="content-language"]');
                return metaLang ? metaLang.getAttribute('content') : '';
            }"""
            )

            if meta_lang and len(meta_lang) >= 2:
                lang_code = meta_lang.split("-")[0].lower()
                if lang_code in ["ru", "en", "fr", "de", "es", "it", "zh", "ja", "ko", "ar"]:
                    return lang_code

            # Извлекаем текст для анализа языка
            text_sample = self.page.evaluate(
                """() => {
                const textNodes = Array.from(document.querySelectorAll('p, h1, h2, h3, h4, h5, h6'))
                    .map(el => el.textContent)
                    .join(' ')
                    .trim();
                return textNodes.substring(0, 1000);  // Берем первые 1000 символов
            }"""
            )

            if text_sample:
                # Определяем язык с помощью langdetect
                try:
                    lang_code = langdetect.detect(text_sample)
                    return lang_code
                except:
                    lang_code = self.improved_language_detection_exception_handling(text_sample)
                    return lang_code

            # По умолчанию возвращаем язык, указанный при инициализации
            return self.language
        except Exception as e:
            logger.error(f"Ошибка при определении языка: {str(e)}")
            return self.language

    def _extract_structured_data(self) -> Dict[str, Any]:
        """
        Извлекает структурированные данные со страницы.

        Returns:
            Dict[str, Any]: Структурированные данные
        """
        result = {"jsonld": [], "opengraph": {}, "twitter": {}, "microdata": [], "microformats": []}

        try:
            # Извлекаем JSON-LD
            jsonld_data = self.page.evaluate(
                """() => {
                const jsonLdScripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
                return jsonLdScripts.map(script => {
                    try {
                        return JSON.parse(script.textContent);
                    } catch (e) {
                        return { error: e.message, content: script.textContent };
                    }
                });
            }"""
            )

            if jsonld_data:
                result["jsonld"] = jsonld_data

            # Извлекаем Open Graph
            og_data = self.page.evaluate(
                """() => {
                const ogMeta = Array.from(document.querySelectorAll('meta[property^="og:"]'));
                const data = {};
                
                ogMeta.forEach(meta => {
                    const property = meta.getAttribute('property').replace('og:', '');
                    const content = meta.getAttribute('content');
                    data[property] = content;
                });
                
                return data;
            }"""
            )

            if og_data:
                result["opengraph"] = og_data

            # Извлекаем Twitter Card
            twitter_data = self.page.evaluate(
                """() => {
                const twitterMeta = Array.from(document.querySelectorAll('meta[name^="twitter:"]'));
                const data = {};
                
                twitterMeta.forEach(meta => {
                    const name = meta.getAttribute('name').replace('twitter:', '');
                    const content = meta.getAttribute('content');
                    data[name] = content;
                });
                
                return data;
            }"""
            )

            if twitter_data:
                result["twitter"] = twitter_data

            # Извлекаем Microdata
            microdata = self.page.evaluate(
                """() => {
                const extractMicrodata = (element) => {
                    const result = {
                        type: element.getAttribute('itemtype'),
                        properties: {}
                    };
                    
                    const props = Array.from(element.querySelectorAll('[itemprop]'));
                    props.forEach(prop => {
                        const name = prop.getAttribute('itemprop');
                        let value;
                        
                        if (prop.hasAttribute('content')) {
                            value = prop.getAttribute('content');
                        } else if (prop.hasAttribute('datetime')) {
                            value = prop.getAttribute('datetime');
                        } else {
                            value = prop.textContent.trim();
                        }
                        
                        result.properties[name] = value;
                    });
                    
                    return result;
                };
                
                return Array.from(document.querySelectorAll('[itemscope][itemtype]'))
                    .map(element => extractMicrodata(element));
            }"""
            )

            if microdata:
                result["microdata"] = microdata

            # Извлекаем Microformats
            microformats = self.page.evaluate(
                """() => {
                const elements = Array.from(document.querySelectorAll('[class*="h-"]'));
                return elements.map(el => {
                    const classes = Array.from(el.classList)
                        .filter(cls => cls.startsWith('h-'));
                    return {
                        type: classes,
                        element: el.outerHTML.substring(0, 200) + '...' // Сокращаем для краткости
                    };
                });
            }"""
            )

            if microformats:
                result["microformats"] = microformats
        except Exception as e:
            logger.error(f"Ошибка при извлечении структурированных данных: {str(e)}")

        return result

    def _extract_meta_data(self) -> Dict[str, Any]:
        """
        Извлекает мета-данные страницы.

        Returns:
            Dict[str, Any]: Мета-данные
        """
        try:
            meta_data = self.page.evaluate(
                """() => {
                const extractMetaContent = (selector) => {
                    const element = document.querySelector(selector);
                    return element ? element.getAttribute('content') : null;
                };
                
                return {
                    title: document.title,
                    description: extractMetaContent('meta[name="description"]'),
                    keywords: extractMetaContent('meta[name="keywords"]'),
                    author: extractMetaContent('meta[name="author"]'),
                    viewport: extractMetaContent('meta[name="viewport"]'),
                    robots: extractMetaContent('meta[name="robots"]'),
                    canonical: document.querySelector('link[rel="canonical"]') 
                        ? document.querySelector('link[rel="canonical"]').getAttribute('href') 
                        : null,
                    favicon: document.querySelector('link[rel="icon"], link[rel="shortcut icon"]') 
                        ? document.querySelector('link[rel="icon"], link[rel="shortcut icon"]').getAttribute('href') 
                        : null,
                    metaTags: Array.from(document.querySelectorAll('meta'))
                        .filter(meta => meta.hasAttribute('name') && meta.hasAttribute('content'))
                        .map(meta => ({
                            name: meta.getAttribute('name'),
                            content: meta.getAttribute('content')
                        }))
                };
            }"""
            )

            return meta_data
        except Exception as e:
            logger.error(f"Ошибка при извлечении мета-данных: {str(e)}")
            return {}

    def _get_page_title(self) -> str:
        """
        Получает заголовок страницы.

        Returns:
            str: Заголовок страницы
        """
        try:
            return self.page.title()
        except Exception as e:
            logger.error(f"Ошибка при получении заголовка страницы: {str(e)}")
            return ""

    def _extract_content_with_structure(self) -> Dict[str, Any]:
        """
        Извлекает контент страницы с учетом семантической структуры.

        Returns:
            Dict[str, Any]: Контент страницы с семантической структурой
        """
        result = {
            "title": self._get_page_title(),
            "main_content": "",
            "headings": [],
            "navigation": [],
            "footer": "",
            "aside": [],
            "interactive_elements": [],
            "important_lists": [],
            "hidden_content": [],
        }

        try:
            # Извлекаем основной контент
            for selector in self.semantic_selectors["main_content"]:
                content = self._extract_element_content(selector)
                if content and len(content) > 100:  # Проверяем, что контент достаточно большой
                    result["main_content"] = content
                    break

            # Если основной контент не найден, используем всё содержимое body
            if not result["main_content"]:
                result["main_content"] = self._extract_element_content("body")

            # Извлекаем заголовки
            result["headings"] = self._extract_headings()

            # Извлекаем навигацию
            for selector in self.semantic_selectors["navigation"]:
                nav_content = self._extract_element_content(selector)
                if nav_content:
                    result["navigation"].append({"selector": selector, "content": nav_content})

            # Извлекаем подвал
            for selector in self.semantic_selectors["footer"]:
                footer_content = self._extract_element_content(selector)
                if footer_content:
                    result["footer"] = footer_content
                    break

            # Извлекаем боковые панели
            for selector in self.semantic_selectors["aside"]:
                aside_content = self._extract_element_content(selector)
                if aside_content:
                    result["aside"].append({"selector": selector, "content": aside_content})

            # Извлекаем интерактивные элементы
            result["interactive_elements"] = self._extract_interactive_elements()

            # Извлекаем важные списки
            result["important_lists"] = self._extract_important_lists()

            # Извлекаем скрытый контент, если включено
            if self.extract_hidden_content:
                result["hidden_content"] = self._extract_hidden_content()
        except Exception as e:
            logger.error(f"Ошибка при извлечении контента с структурой: {str(e)}")

        return result

    def _extract_element_content(self, selector: str) -> str:
        """
        Извлекает текстовое содержимое элемента по селектору.

        Args:
            selector: CSS-селектор элемента

        Returns:
            str: Текстовое содержимое элемента
        """
        try:
            content = self.page.evaluate(
                f"""(selector) => {{
                const element = document.querySelector(selector);
                if (!element) return '';
                
                // Функция для очистки текста от лишних пробелов
                const cleanText = (text) => {{
                    return text.replace(/\s+/g, ' ').trim();
                }};
                
                // Извлекаем текстовое содержимое с учетом форматирования
                const extractFormattedText = (element) => {{
                    let result = '';
                    
                    // Рекурсивно обрабатываем дочерние элементы
                    for (const node of element.childNodes) {{
                        if (node.nodeType === Node.TEXT_NODE) {{
                            // Добавляем текст
                            const text = cleanText(node.textContent);
                            if (text) result += text + ' ';
                        }} else if (node.nodeType === Node.ELEMENT_NODE) {{
                            // Обрабатываем элементы в зависимости от тега
                            const tagName = node.tagName.toLowerCase();
                            
                            if (['script', 'style', 'noscript'].includes(tagName)) {{
                                continue;  // Пропускаем скрипты, стили и noscript
                            }}
                            
                            if (['br'].includes(tagName)) {{
                                result += '\n';  // Перенос строки для <br>
                            }} else if (['p', 'div', 'section', 'article', 'header', 'footer', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {{
                                // Блочные элементы
                                const innerText = extractFormattedText(node);
                                if (innerText) result += '\n' + innerText + '\n';
                            }} else if (['li'].includes(tagName)) {{
                                // Элементы списка
                                const innerText = extractFormattedText(node);
                                if (innerText) result += '\n• ' + innerText;
                            }} else if (['a'].includes(tagName)) {{
                                // Ссылки
                                const innerText = extractFormattedText(node);
                                const href = node.getAttribute('href');
                                if (innerText) {{
                                    result += innerText + (href ? ` [${{href}}]` : '');  # noqa: F821
                                }}
                            }} else {{
                                // Прочие элементы
                                const innerText = extractFormattedText(node);
                                if (innerText) result += innerText;
                            }}
                        }}
                    }}
                    
                    return result.trim();
                }};
                
                return extractFormattedText(element);
            }}""",
                selector,
            )

            return content
        except Exception as e:
            logger.error(f"Ошибка при извлечении содержимого элемента '{selector}': {str(e)}")
            return ""

    def _extract_headings(self) -> List[Dict[str, Any]]:
        """
        Извлекает заголовки страницы с учетом иерархии.

        Returns:
            List[Dict[str, Any]]: Список заголовков с их иерархией
        """
        try:
            headings = self.page.evaluate(
                """() => {
                return Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
                    .map(heading => {
                        return {
                            level: parseInt(heading.tagName.substring(1)),
                            text: heading.textContent.trim(),
                            id: heading.id || '',
                            classes: Array.from(heading.classList),
                            position: heading.getBoundingClientRect().top + window.pageYOffset
                        };
                    })
                    .filter(heading => heading.text.length > 0)
                    .sort((a, b) => a.position - b.position);
            }"""
            )

            return headings
        except Exception as e:
            logger.error(f"Ошибка при извлечении заголовков: {str(e)}")
            return []

    def _extract_interactive_elements(self) -> List[Dict[str, Any]]:
        """
        Извлекает интерактивные элементы страницы.

        Returns:
            List[Dict[str, Any]]: Список интерактивных элементов
        """
        try:
            # Создаем селектор из всех возможных интерактивных элементов
            interactive_selector = ", ".join(self.semantic_selectors["interactive_elements"])

            elements = self.page.evaluate(
                f"""(selector) => {{
                return Array.from(document.querySelectorAll(selector))
                    .map(element => {{
                        const tagName = element.tagName.toLowerCase();
                        const type = element.getAttribute('type') || '';
                        const text = element.textContent.trim();
                        const ariaLabel = element.getAttribute('aria-label') || '';
                        const placeholder = element.getAttribute('placeholder') || '';
                        
                        // Определяем тип элемента
                        let elementType = tagName;
                        if (tagName === 'input') elementType = type || 'text';
                        else if (element.hasAttribute('role')) elementType = element.getAttribute('role');
                        
                        return {{
                            type: elementType,
                            text: text || ariaLabel || placeholder,
                            is_visible: !!element.offsetWidth && !!element.offsetHeight,
                            is_enabled: !element.disabled,
                            position: {{
                                top: element.getBoundingClientRect().top + window.pageYOffset,
                                left: element.getBoundingClientRect().left + window.pageXOffset
                            }}
                        }};
                    }})
                    .filter(el => el.text.length > 0);
            }}""",
                interactive_selector,
            )

            return elements
        except Exception as e:
            logger.error(f"Ошибка при извлечении интерактивных элементов: {str(e)}")
            return []

    def _extract_important_lists(self) -> List[Dict[str, Any]]:
        """
        Извлекает важные списки на странице.

        Returns:
            List[Dict[str, Any]]: Список важных списков
        """
        try:
            # Создаем селектор из всех возможных важных списков
            list_selector = ", ".join(self.semantic_selectors["important_list_elements"])

            lists = self.page.evaluate(
                f"""(selector) => {{
                return Array.from(document.querySelectorAll(selector))
                    .map(list => {{
                        const tagName = list.tagName.toLowerCase();
                        const listType = tagName === 'ul' ? 'unordered' : 
                                        tagName === 'ol' ? 'ordered' : 'custom';
                        
                        return {{
                            type: listType,
                            class: Array.from(list.classList).join(' '),
                            items: Array.from(list.querySelectorAll('li'))
                                .map(item => item.textContent.trim())
                                .filter(text => text.length > 0)
                        }};
                    }})
                    .filter(list => list.items.length > 0);
            }}""",
                list_selector,
            )

            return lists
        except Exception as e:
            logger.error(f"Ошибка при извлечении важных списков: {str(e)}")
            return []

    def _extract_hidden_content(self) -> List[Dict[str, Any]]:
        """
        Извлекает скрытые элементы, важные для LLM.

        Returns:
            List[Dict[str, Any]]: Список скрытых элементов
        """
        try:
            # Создаем селектор из всех возможных скрытых элементов
            hidden_selector = ", ".join(self.semantic_selectors["important_hidden_elements"])

            hidden_elements = self.page.evaluate(
                f"""(selector) => {{
                return Array.from(document.querySelectorAll(selector))
                    .map(element => {{
                        const tagName = element.tagName.toLowerCase();
                        const text = element.textContent.trim();
                        
                        return {{
                            type: tagName,
                            text: text,
                            reason: "hidden",
                            selectors: Array.from(element.classList).map(cls => '.' + cls)
                        }};
                    }})
                    .filter(el => el.text.length > 0 && el.text.length < 1000);  // Ограничиваем размер
            }}""",
                hidden_selector,
            )

            return hidden_elements
        except Exception as e:
            logger.error(f"Ошибка при извлечении скрытых элементов: {str(e)}")
            return []

    def _optimize_content_for_llm(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизирует контент для LLM-анализа.

        Args:
            content: Контент страницы с семантической структурой

        Returns:
            Dict[str, Any]: Оптимизированный контент
        """
        try:
            optimized = content.copy()

            # Очищаем основной контент от ненужных элементов
            optimized["main_content"] = self._clean_text_for_llm(optimized["main_content"])

            # Оптимизируем заголовки
            for heading in optimized["headings"]:
                heading["text"] = self._clean_text_for_llm(heading["text"])

            # Оптимизируем навигацию
            for nav in optimized["navigation"]:
                nav["content"] = self._clean_text_for_llm(nav["content"])

            # Оптимизируем подвал
            optimized["footer"] = self._clean_text_for_llm(optimized["footer"])

            # Оптимизируем боковые панели
            for aside in optimized["aside"]:
                aside["content"] = self._clean_text_for_llm(aside["content"])

            # Оптимизируем интерактивные элементы
            for element in optimized["interactive_elements"]:
                element["text"] = self._clean_text_for_llm(element["text"])

            # Оптимизируем важные списки
            for list_item in optimized["important_lists"]:
                list_item["items"] = [self._clean_text_for_llm(item) for item in list_item["items"]]

            # Оптимизируем скрытый контент
            for hidden in optimized["hidden_content"]:
                hidden["text"] = self._clean_text_for_llm(hidden["text"])

            # Структурируем контент в формате, удобном для LLM
            optimized["llm_structured"] = self._create_llm_structure(optimized)

            # Добавляем метки для часто встречающихся сущностей
            optimized["entities"] = self._extract_entities(optimized)

            return optimized
        except Exception as e:
            logger.error(f"Ошибка при оптимизации контента для LLM: {str(e)}")
            return content

    def _clean_text_for_llm(self, text: str) -> str:
        """
        Очищает текст для LLM-анализа.

        Args:
            text: Исходный текст

        Returns:
            str: Очищенный текст
        """
        if not text:
            return ""

        # Удаляем HTML-комментарии
        text = re.sub(self.content_patterns["html_comment"], "", text)

        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r"\s+", " ", text)

        # Удаляем повторяющиеся символы
        text = re.sub(r"([.,!?:;])[.,!?:;]+", r"\1", text)

        # Удаляем слишком длинные строки без пробелов (вероятно, это код или мусор)
        text = re.sub(r"\S{100,}", "", text)

        return text.strip()

    def _create_llm_structure(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создает структуру контента, оптимизированную для LLM.

        Args:
            content: Контент страницы с семантической структурой

        Returns:
            Dict[str, Any]: Структурированный контент для LLM
        """
        # Создаем структуру документа для LLM
        document = {
            "title": content["title"],
            "headings": self._restructure_headings(content["headings"]),
            "main_content": content["main_content"],
            "summary": self._generate_summary(content),
            "key_points": self._extract_key_points(content),
            "lists": content["important_lists"],
            "metadata": {
                "has_navigation": len(content["navigation"]) > 0,
                "has_footer": bool(content["footer"]),
                "has_aside": len(content["aside"]) > 0,
                "has_interactive_elements": len(content["interactive_elements"]) > 0,
                "has_hidden_content": len(content["hidden_content"]) > 0,
            },
        }

        return document

    def _restructure_headings(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Реструктурирует заголовки в иерархическую структуру.

        Args:
            headings: Список заголовков

        Returns:
            Dict[str, Any]: Иерархическая структура заголовков
        """
        result = {"level_1": [], "level_2": [], "level_3_plus": []}

        for heading in headings:
            level = heading["level"]
            if level == 1:
                result["level_1"].append(heading)
            elif level == 2:
                result["level_2"].append(heading)
            else:
                result["level_3_plus"].append(heading)

        return result

    def _generate_summary(self, content: Dict[str, Any]) -> str:
        """
        Генерирует краткое содержание страницы на основе контента.

        Args:
            content: Контент страницы

        Returns:
            str: Краткое содержание
        """
        # Это упрощенная версия генерации саммари
        # В реальном приложении здесь может использоваться более сложная логика или LLM

        summary_parts = []

        # Добавляем заголовок
        if content["title"]:
            summary_parts.append(f"Страница: {content['title']}")

        # Добавляем основные заголовки
        if content["headings"]:
            h1_headings = [h["text"] for h in content["headings"] if h["level"] == 1]
            if h1_headings:
                summary_parts.append(f"Основной заголовок: {h1_headings[0]}")

            h2_headings = [h["text"] for h in content["headings"] if h["level"] == 2]
            if h2_headings:
                summary_parts.append(f"Подзаголовки: {', '.join(h2_headings[:5])}")
                if len(h2_headings) > 5:
                    summary_parts[-1] += " и другие"

        # Добавляем первый абзац основного контента (если он есть)
        if content["main_content"]:
            paragraphs = content["main_content"].split("\n\n")
            if paragraphs:
                first_paragraph = paragraphs[0].strip()
                if len(first_paragraph) > 200:
                    summary_parts.append(f"Начало содержания: {first_paragraph[:200]}...")
                else:
                    summary_parts.append(f"Начало содержания: {first_paragraph}")

        return "\n".join(summary_parts)

    def _extract_key_points(self, content: Dict[str, Any]) -> List[str]:
        """
        Извлекает ключевые моменты из контента.

        Args:
            content: Контент страницы

        Returns:
            List[str]: Список ключевых моментов
        """
        key_points = []

        # Добавляем H1 и H2 заголовки
        for heading in content["headings"]:
            if heading["level"] <= 2 and heading["text"]:
                key_points.append(heading["text"])

        # Добавляем элементы из важных списков
        for list_item in content["important_lists"]:
            for item in list_item["items"][:3]:  # Берем только первые три элемента
                if item and len(item) < 100:  # Ограничиваем длину
                    key_points.append(item)

        # Добавляем интерактивные элементы с текстом кнопок/действий
        for element in content["interactive_elements"]:
            if element["type"] in ["button", "submit", 'role="button"'] and element["text"]:
                if len(element["text"]) < 50:  # Ограничиваем длину
                    key_points.append(f"Действие: {element['text']}")

        # Ограничиваем количество ключевых моментов
        return key_points[:10]

    def _extract_entities(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает часто встречающиеся сущности из контента.

        Args:
            content: Контент страницы

        Returns:
            Dict[str, Any]: Словарь сущностей
        """
        entities = {"emails": [], "phones": [], "prices": [], "dates": []}

        # Объединяем весь текстовый контент
        all_text = content["main_content"]
        for heading in content["headings"]:
            all_text += " " + heading["text"]
        for nav in content["navigation"]:
            all_text += " " + nav["content"]
        all_text += " " + content["footer"]
        for aside in content["aside"]:
            all_text += " " + aside["content"]

        # Извлекаем email адреса
        emails = self.content_patterns["email"].findall(all_text)
        entities["emails"] = list(set(emails))[:10]  # Удаляем дубликаты и ограничиваем количество

        # Извлекаем телефонные номера
        phones = self.content_patterns["phone"].findall(all_text)
        entities["phones"] = list(set(phones))[:10]

        # Извлекаем цены
        prices = self.content_patterns["price"].findall(all_text)
        entities["prices"] = list(set(prices))[:10]

        # Извлекаем даты
        dates = self.content_patterns["date"].findall(all_text)
        entities["dates"] = list(set(dates))[:10]

        return entities

    def _process_ajax_requests(self) -> List[Dict[str, Any]]:
        """
        Обрабатывает AJAX-запросы и структурирует их.

        Returns:
            List[Dict[str, Any]]: Обработанные AJAX-запросы
        """
        processed_requests = []

        for request in self.ajax_requests:
            # Пропускаем запросы без ответов
            if not request.get("response"):
                continue

            # Формируем информацию о запросе
            processed_request = {
                "url": request["url"],
                "method": request["method"],
                "status": request["response"].get("status"),
                "content_type": request["response"].get("headers", {}).get("content-type", ""),
            }

            # Если в ответе есть JSON, добавляем его ключи
            if isinstance(request["response"].get("body"), dict):
                processed_request["data_keys"] = list(request["response"]["body"].keys())

            processed_requests.append(processed_request)

        return processed_requests

    def _filter_console_messages(self) -> List[Dict[str, Any]]:
        """
        Фильтрует и структурирует консольные сообщения.

        Returns:
            List[Dict[str, Any]]: Обработанные консольные сообщения
        """
        # Группируем сообщения по типу
        grouped_messages = defaultdict(list)
        for msg in self.console_messages:
            grouped_messages[msg["type"]].append(msg)

        # Формируем результат
        result = []
        for msg_type, messages in grouped_messages.items():
            # Для ошибок и предупреждений сохраняем все сообщения
            if msg_type in ["error", "warning"]:
                for msg in messages:
                    result.append({"type": msg_type, "text": msg["text"]})
            # Для прочих типов сохраняем только количество
            else:
                result.append({"type": msg_type, "count": len(messages)})

        return result

    def _crawl_links(self, depth: int) -> List[Dict[str, Any]]:
        """
        Выполняет рекурсивный краулинг по ссылкам страницы.

        Args:
            depth: Текущая глубина краулинга

        Returns:
            List[Dict[str, Any]]: Результаты краулинга связанных страниц
        """
        results = []

        try:
            # Получаем все ссылки на странице
            links = self.page.evaluate(
                """() => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => {
                        return {
                            href: a.href,
                            text: a.textContent.trim(),
                            isExternal: a.hostname !== window.location.hostname,
                            rel: a.rel
                        };
                    })
                    .filter(link => link.href && link.href !== '#' && !link.href.startsWith('javascript:') && !link.href.startsWith('mailto:') && !link.href.startsWith('tel:'));
            }"""
            )

            # Фильтруем ссылки
            filtered_links = []
            for link in links:
                url = link["href"]

                # Пропускаем уже посещенные URL
                if url in self.visited_links:
                    continue

                # Применяем фильтры
                if link["isExternal"] and not self.allowed_domains:
                    continue

                # Проверяем домен для внешних ссылок
                if link["isExternal"]:
                    parsed_url = urlparse(url)
                    if parsed_url.netloc not in self.allowed_domains:
                        continue

                # Проверяем шаблоны исключений
                if any(re.search(pattern, url) for pattern in self.excluded_patterns):
                    continue

                # Проверяем шаблоны включений
                if self.included_patterns and not any(
                    re.search(pattern, url) for pattern in self.included_patterns
                ):
                    continue

                filtered_links.append(link)

            # Ограничиваем количество ссылок
            filtered_links = filtered_links[: min(self.max_pages, len(filtered_links))]

            # Краулим каждую ссылку
            for link in filtered_links:
                url = link["href"]

                # Добавляем URL в список посещенных
                self.visited_links.add(url)

                # Краулим ссылку
                result = self.crawl_url(url, depth)

                # Добавляем результат
                results.append(
                    {
                        "url": url,
                        "text": link["text"],
                        "title": result.get("title", ""),
                        "error": result.get("error"),
                    }
                )

                # Если достигли максимального количества страниц, прерываем цикл
                if len(results) >= self.max_pages:
                    break
        except Exception as e:
            logger.error(f"Ошибка при краулинге ссылок: {str(e)}")

        return results

    def _close_browser(self):
        """Закрывает браузер и освобождает ресурсы."""
        if self.page:
            self.page.close()
            self.page = None

        if self.context:
            self.context.close()
            self.context = None

        if self.browser:
            self.browser.close()
            self.browser = None

        if self.playwright:
            self.playwright.stop()
            self.playwright = None


def improved_language_detection_exception_handling(self, text_sample):
    try:
        lang_code = langdetect.detect(text_sample)
        return lang_code
    except langdetect.LangDetectException as e:
        self.logger.warning(f"Ошибка LangDetect при определении языка: {str(e)}")
        # Альтернативный способ определения языка - по частоте встречаемости характерных символов
        ru_chars = len(re.findall("[а-яА-Я]", text_sample))
        en_chars = len(re.findall("[a-zA-Z]", text_sample))

        if ru_chars > en_chars * 0.3:  # если русских символов больше 30% от английских
            self.logger.info("Язык определен альтернативным способом: ru")
            return "ru"
        elif en_chars > 0:
            self.logger.info("Язык определен альтернативным способом: en")
            return "en"

        # Если не удалось определить язык, используем язык по умолчанию
        self.logger.info(
            f"Не удалось определить язык, используется язык по умолчанию: {self.language}"
        )
        return self.language


# Функция для создания экземпляра EnhancedSPACrawlerLLM с настройками по умолчанию
def create_enhanced_spa_crawler_llm(**kwargs) -> EnhancedSPACrawlerLLM:
    """
    Создает экземпляр EnhancedSPACrawlerLLM с настройками.

    Args:
        **kwargs: Параметры для инициализации краулера

    Returns:
        EnhancedSPACrawlerLLM: Экземпляр краулера
    """
    return EnhancedSPACrawlerLLM(**kwargs)


# noqa: F821
