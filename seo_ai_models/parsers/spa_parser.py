"""
Модуль для парсинга SPA-сайтов с поддержкой JavaScript и AJAX.
Интегрирует все улучшения в единый пакет.
"""

import logging
import asyncio
import time
import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPAParser:
    """
    Универсальный парсер для SPA-сайтов с поддержкой JavaScript и AJAX.
    """
    
    def __init__(
        self,
        headless: bool = True,
        wait_for_load: int = 5000,  # мс ожидания после события load
        wait_for_timeout: int = 30000,  # мс максимального ожидания для загрузки
        browser_type: str = "chromium",  # 'chromium', 'firefox', or 'webkit'
        user_agent: str = "SEOAIModels SPAParser/1.0",
        viewport_width: int = 1366,
        viewport_height: int = 768,
        cache_dir: str = ".cache",
        cache_enabled: bool = True,
        cache_max_age: int = 86400,  # 24 часа в секундах
        record_ajax: bool = True,
        analyze_ajax: bool = True,
        ajax_patterns: List[str] = None,
        content_tags: List[str] = None,
        block_tags: List[str] = None,
        exclude_classes: List[str] = None,
        exclude_ids: List[str] = None,
        max_retries: int = 2
    ):
        """
        Инициализация SPAParser.

        Args:
            headless: Запускать ли браузер в режиме headless
            wait_for_load: Время ожидания в мс после события load
            wait_for_timeout: Максимальное время ожидания в мс для загрузки страницы
            browser_type: Тип браузера для использования ('chromium', 'firefox', 'webkit')
            user_agent: User-Agent для запросов
            viewport_width: Ширина области просмотра браузера
            viewport_height: Высота области просмотра браузера
            cache_dir: Директория для хранения кэша
            cache_enabled: Включено ли кэширование
            cache_max_age: Максимальный возраст кэша в секундах
            record_ajax: Записывать ли AJAX-запросы
            analyze_ajax: Анализировать ли AJAX-ответы
            ajax_patterns: Паттерны URL для идентификации AJAX-запросов
            content_tags: HTML-теги, которые обычно содержат основной контент
            block_tags: Теги, которые представляют блочные элементы
            exclude_classes: CSS-классы для исключения из извлечения
            exclude_ids: HTML-идентификаторы для исключения из извлечения
            max_retries: Максимальное количество повторных попыток при ошибке
        """
        self.headless = headless
        self.wait_for_load = wait_for_load
        self.wait_for_timeout = wait_for_timeout
        self.browser_type = browser_type
        self.user_agent = user_agent
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.cache_max_age = cache_max_age
        self.record_ajax = record_ajax
        self.analyze_ajax = analyze_ajax
        self.max_retries = max_retries
        
        # Параметры для идентификации AJAX-запросов
        self.ajax_patterns = ajax_patterns or [
            '/api/', '/graphql', '/v1/', '/v2/', '/v3/', '/rest/',
            'json', 'data.', '.php', 'ajax', 'xhr', 'action='
        ]
        
        # Параметры для анализа контента
        self.content_tags = content_tags or [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
            'ul', 'ol', 'li', 'table', 'article', 'section', 'main'
        ]
        
        self.block_tags = block_tags or [
            'article', 'section', 'div', 'main', 'header', 'footer',
            'nav', 'aside', 'form', 'table'
        ]
        
        self.exclude_classes = exclude_classes or [
            'advertisement', 'ads', 'banner', 'menu', 'nav', 'sidebar',
            'footer', 'comment', 'cookie', 'popup', 'modal'
        ]
        
        self.exclude_ids = exclude_ids or [
            'advertisement', 'ads', 'banner', 'menu', 'nav', 'sidebar',
            'footer', 'comment', 'cookie', 'popup', 'modal'
        ]
        
        # Создаем директорию кэша, если нужно
        if self.cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Хранение последних AJAX-данных
        self._last_ajax_data = None
        
        # Признаки для определения SPA
        self.spa_framework_patterns = {
            "React": [
                'reactroot', 'react-root', 'data-reactroot', 'data-reactid',
                'react.development.js', 'react.production.min.js', 'react-dom'
            ],
            "Angular": [
                'ng-app', 'ng-controller', 'ng-model', 'ng-bind', 'ng-repeat',
                'angular.js', 'angular.min.js', '_nghost', '_ngcontent',
                'ng-version', '[ng'
            ],
            "Vue": [
                'v-app', 'v-bind', 'v-model', 'v-if', 'v-for', 'v-on',
                'vue.js', 'vue.min.js', 'data-v-', '[data-v-', 'vue-router',
                'vuex'
            ],
            "Next.js": [
                '__NEXT_DATA__', 'next/router', '_next/'
            ],
            "Nuxt.js": [
                '__NUXT__', 'nuxt-link', '_nuxt/'
            ]
        }
    
    def _get_cache_key(self, url: str, params: Dict = None) -> str:
        """
        Создает ключ кэша для URL и параметров.
        
        Args:
            url: URL для кэширования
            params: Дополнительные параметры, влияющие на содержимое
            
        Returns:
            str: Ключ кэша
        """
        # Создаем хэш на основе URL и параметров
        key_data = url
        if params:
            key_data += json.dumps(params, sort_keys=True)
            
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Получает путь к файлу кэша.
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            str: Путь к файлу кэша
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get_from_cache(self, url: str, params: Dict = None) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Получает данные из кэша.
        
        Args:
            url: URL для кэширования
            params: Дополнительные параметры
            
        Returns:
            Tuple[Optional[Dict[str, Any]], bool]: 
                - Данные из кэша (None, если не найдены)
                - Флаг, указывающий, актуален ли кэш
        """
        if not self.cache_enabled:
            return None, False
        
        cache_key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None, False
        
        # Проверяем возраст кэша
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age > self.cache_max_age:
            logger.info(f"Кэш для {url} устарел ({file_age:.0f} с > {self.cache_max_age} с)")
            return None, False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Загружены данные из кэша для {url}")
                return data, True
        except Exception as e:
            logger.error(f"Ошибка при загрузке кэша для {url}: {str(e)}")
            return None, False
    
    def save_to_cache(self, url: str, data: Dict[str, Any], params: Dict = None) -> bool:
        """
        Сохраняет данные в кэш.
        
        Args:
            url: URL для кэширования
            data: Данные для сохранения
            params: Дополнительные параметры
            
        Returns:
            bool: True при успешном сохранении
        """
        if not self.cache_enabled:
            return False
        
        cache_key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Данные для {url} сохранены в кэш")
                return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении кэша для {url}: {str(e)}")
            return False
    
    def clear_cache(self, url: str = None, params: Dict = None) -> bool:
        """
        Очищает кэш для URL или весь кэш.
        
        Args:
            url: URL для удаления из кэша (если None, очищает весь кэш)
            params: Дополнительные параметры
            
        Returns:
            bool: True при успешной очистке
        """
        if not self.cache_enabled:
            return False
        
        if url:
            # Очистка кэша для конкретного URL
            cache_key = self._get_cache_key(url, params)
            cache_path = self._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    logger.info(f"Кэш для {url} очищен")
                    return True
                except Exception as e:
                    logger.error(f"Ошибка при очистке кэша для {url}: {str(e)}")
                    return False
        else:
            # Очистка всего кэша
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path) and filename.endswith('.json'):
                        os.remove(file_path)
                        
                logger.info("Весь кэш очищен")
                return True
            except Exception as e:
                logger.error(f"Ошибка при очистке всего кэша: {str(e)}")
                return False
    
    def detect_is_spa(self, html_content: str) -> Dict[str, Any]:
        """
        Определяет, является ли сайт SPA на основе анализа HTML.
        
        Args:
            html_content: HTML-контент для анализа
            
        Returns:
            Dict[str, Any]: Информация о типе сайта
        """
        if not html_content:
            return {
                "is_spa": False,
                "confidence": 0,
                "detected_frameworks": []
            }
            
        # Получаем текст для анализа
        html_text = str(html_content).lower()
        
        # Проверяем на фреймворки
        detected_frameworks = []
        for framework, patterns in self.spa_framework_patterns.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in html_text:
                    if framework not in detected_frameworks:
                        detected_frameworks.append(framework)
                    break
                    
        # Другие признаки SPA
        spa_indicators = [
            'axios', 'fetch(', '.fetch(', 'XMLHttpRequest', 'jquery.ajax',
            'router', 'history.pushState', 'window.history',
            'template', 'mustache', 'handlebars',
            'spa', 'single-page-application', 'single-page-app'
        ]
        
        # Подсчет признаков SPA
        spa_indicators_found = []
        for indicator in spa_indicators:
            if indicator.lower() in html_text:
                spa_indicators_found.append(indicator)
        
        # Вычисляем оценку уверенности
        confidence = 0
        
        # Если найден хотя бы один фреймворк, высокая уверенность
        if detected_frameworks:
            confidence += 0.7
            
        # Если найдены другие признаки SPA, средняя уверенность
        if spa_indicators_found:
            confidence += min(0.3, 0.05 * len(spa_indicators_found))
            
        # Ограничиваем значение уверенности до 1.0
        confidence = min(1.0, confidence)
        
        # Определяем, является ли сайт SPA
        is_spa = confidence > 0.3 or len(detected_frameworks) > 0
        
        return {
            "is_spa": is_spa,
            "confidence": confidence,
            "detected_frameworks": detected_frameworks
        }
    
    async def _is_ajax_request(self, url: str) -> bool:
        """
        Проверяет, является ли URL AJAX-запросом.
        
        Args:
            url: URL для проверки
            
        Returns:
            bool: True, если URL является AJAX-запросом
        """
        # Игнорируем запросы к статическим файлам
        ignored_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', 
                             '.css', '.ico', '.svg', '.mp4', '.mp3']
        
        if any(url.lower().endswith(ext) for ext in ignored_extensions):
            return False
            
        # Проверяем на паттерны AJAX
        return any(pattern.lower() in url.lower() for pattern in self.ajax_patterns)
    
    async def _setup_ajax_tracking(self, page) -> List[Dict[str, Any]]:
        """
        Настраивает отслеживание AJAX-запросов на странице.
        
        Args:
            page: Объект Page из Playwright
            
        Returns:
            List[Dict[str, Any]]: Список перехваченных AJAX-запросов
        """
        if not self.record_ajax:
            return []
            
        ajax_calls = []
        json_responses = {}
        
        # Обработчик запросов
        async def on_request(request):
            url = request.url
            
            if await self._is_ajax_request(url):
                # Записываем информацию о запросе
                ajax_calls.append({
                    'url': url,
                    'method': request.method,
                    'headers': request.headers,
                    'timestamp': time.time(),
                    'response': None
                })
        
        # Обработчик ответов
        async def on_response(response):
            if not self.analyze_ajax:
                return
                
            url = response.url
            
            if await self._is_ajax_request(url):
                try:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type or 'application/javascript' in content_type:
                        # Получаем текст ответа
                        text = await response.text()
                        
                        # Пробуем распарсить JSON
                        try:
                            json_data = json.loads(text)
                            json_responses[url] = json_data
                            
                            # Обновляем записанный AJAX-вызов
                            for call in ajax_calls:
                                if call['url'] == url and call['response'] is None:
                                    call['response'] = {
                                        'status': response.status,
                                        'headers': dict(response.headers),
                                        'json': json_data
                                    }
                                    break
                        except json.JSONDecodeError:
                            pass
                        
                except Exception as e:
                    logger.error(f"Ошибка при обработке ответа от {url}: {str(e)}")
        
        # Устанавливаем обработчики
        page.on('request', on_request)
        page.on('response', on_response)
        
        return ajax_calls
    
    def _extract_ajax_data(self, ajax_calls: List[Dict[str, Any]], json_responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает структурированные данные из AJAX-ответов.
        
        Args:
            ajax_calls: Список AJAX-вызовов
            json_responses: Словарь JSON-ответов
            
        Returns:
            Dict[str, Any]: Структурированные данные
        """
        data = {
            'api_endpoints': [],
            'entities': {}
        }
        
        # Анализ API-вызовов
        if ajax_calls:
            data['api_endpoints'] = [
                {
                    'url': call['url'],
                    'method': call['method'],
                    'status': call.get('response', {}).get('status')
                }
                for call in ajax_calls if call.get('response')
            ]
        
        # Анализ структуры данных из ответов
        for url, json_data in json_responses.items():
            # Пытаемся определить тип данных и структуру
            if isinstance(json_data, dict):
                # Извлекаем верхнеуровневые ключи
                keys = list(json_data.keys())
                
                # Определяем, содержит ли ответ данные о сущностях
                if any(key in ['data', 'items', 'results', 'content', 'response'] for key in keys):
                    for key in ['data', 'items', 'results', 'content', 'response']:
                        if key in json_data and json_data[key]:
                            # Извлекаем потенциальные сущности
                            entity_data = json_data[key]
                            
                            if isinstance(entity_data, list) and entity_data:
                                # Если это список сущностей
                                entity_type = key
                                entity_sample = entity_data[0] if entity_data else {}
                                
                                if isinstance(entity_sample, dict):
                                    data['entities'][entity_type] = {
                                        'fields': list(entity_sample.keys()),
                                        'count': len(entity_data),
                                        'sample': entity_sample
                                    }
        
        return data
    
    def _is_excluded_element(self, element) -> bool:
        """
        Проверка, должен ли элемент быть исключен из извлечения.
        
        Args:
            element: BeautifulSoup Tag для проверки
            
        Returns:
            bool: True, если элемент должен быть исключен
        """
        if not hasattr(element, 'attrs'):
            return False
            
        # Проверка классов
        if element.get('class'):
            for cls in element.get('class', []):
                if any(excl.lower() in cls.lower() for excl in self.exclude_classes):
                    return True
                    
        # Проверка идентификаторов
        if element.get('id'):
            element_id = element.get('id', '').lower()
            if any(excl.lower() in element_id for excl in self.exclude_ids):
                return True
                
        return False
    
    def _extract_content(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Извлекает структурированный контент из HTML.
        
        Args:
            html_content: HTML-контент для парсинга
            url: URL контента (для ссылки)
            
        Returns:
            Dict: Информация об извлеченном контенте
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Удаление элементов script и style
        for script in soup(["script", "style", "noscript", "iframe"]):
            script.decompose()
            
        # Извлечение основного текстового контента
        all_text = soup.get_text(strip=True)
        
        # Извлечение заголовка
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            
        # Извлечение заголовков
        headings = {}
        for i in range(1, 7):
            heading_tags = soup.find_all(f'h{i}')
            headings[f'h{i}'] = [h.get_text(strip=True) for h in heading_tags if not self._is_excluded_element(h)]
            
        # Извлечение параграфов
        paragraphs = []
        for p in soup.find_all('p'):
            if not self._is_excluded_element(p):
                text = p.get_text(strip=True)
                if text:
                    paragraphs.append(text)
        
        # Извлечение списков
        lists = []
        for list_tag in soup.find_all(['ul', 'ol']):
            if not self._is_excluded_element(list_tag):
                list_items = []
                for li in list_tag.find_all('li'):
                    text = li.get_text(strip=True)
                    if text:
                        list_items.append(text)
                if list_items:
                    lists.append({
                        'type': list_tag.name,
                        'items': list_items
                    })
                    
        # Результат
        result = {
            'url': url,
            'title': title,
            'headings': headings,
            'content': {
                'all_text': all_text,
                'paragraphs': paragraphs,
                'lists': lists,
            },
            'metadata': {
                'text_length': len(all_text),
                'paragraph_count': len(paragraphs),
                'list_count': len(lists),
                'heading_counts': {key: len(values) for key, values in headings.items()},
            }
        }
        
        return result
    
    async def analyze_url(self, url: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Полный анализ URL с поддержкой SPA и AJAX.
        
        Args:
            url: URL для анализа
            use_cache: Использовать ли кэш
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        # Проверяем кэш, если включен
        if use_cache and self.cache_enabled:
            cached_data, is_valid = self.get_from_cache(url)
            if cached_data and is_valid:
                return cached_data
        
        result = {
            "url": url,
            "success": False,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # Рендерим страницу
            html_content, ajax_data, is_spa = await self._render_page(url)
            
            if not html_content:
                result["error"] = "Failed to render page content"
                return result
            
            # Анализ типа сайта
            spa_info = self.detect_is_spa(html_content)
            
            # Извлечение контента
            content_data = self._extract_content(html_content, url)
            
            # Сбор результатов
            result.update({
                "success": True,
                "content": content_data,
                "site_type": {
                    "is_spa": spa_info["is_spa"] or is_spa,
                    "confidence": spa_info["confidence"],
                    "detected_frameworks": spa_info["detected_frameworks"]
                },
                "processing_time": time.time() - start_time
            })
            
            # Добавляем данные AJAX, если они есть
            if ajax_data:
                result["ajax_data"] = ajax_data
            
            # Сохраняем в кэш, если включен
            if use_cache and self.cache_enabled:
                self.save_to_cache(url, result)
            
        except Exception as e:
            logger.error(f"Ошибка при анализе {url}: {str(e)}")
            result["error"] = str(e)
            
        return result
    
    async def _render_page(self, url: str, retry_count: int = 0) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool]:
        """
        Рендеринг страницы с JavaScript и перехватом AJAX.
        
        Args:
            url: URL для рендеринга
            retry_count: Текущее число повторных попыток
            
        Returns:
            Tuple[Optional[str], Optional[Dict[str, Any]], bool]:
                - HTML-контент (None при ошибке)
                - AJAX-данные (None, если нет)
                - Флаг, указывающий, является ли сайт SPA
        """
        logger.info(f"Рендеринг страницы ({retry_count+1}/{self.max_retries+1}): {url}")
        
        try:
            async with async_playwright() as playwright:
                # Выбор браузера
                if self.browser_type == "firefox":
                    browser_instance = playwright.firefox
                elif self.browser_type == "webkit":
                    browser_instance = playwright.webkit
                else:
                    browser_instance = playwright.chromium
                    
                # Запуск браузера
                browser = await browser_instance.launch(headless=self.headless)
                
                try:
                    context = await browser.new_context(
                        user_agent=self.user_agent,
                        viewport={'width': self.viewport_width, 'height': self.viewport_height}
                    )
                    
                    page = await context.new_page()
                    
                    # Настройка перехвата AJAX
                    ajax_calls = await self._setup_ajax_tracking(page)
                    json_responses = {}
                    
                    try:
                        # Используем load вместо networkidle для надежности
                        await page.goto(url, wait_until='load', timeout=self.wait_for_timeout)
                        
                        # Дополнительное ожидание для JavaScript
                        await page.wait_for_timeout(self.wait_for_load)
                        
                        # Выполнение дополнительных скриптов для раскрытия скрытого контента
                        await page.evaluate('''() => {
                            // Нажать на все кнопки "Показать больше" или похожие
                            const showMoreButtons = Array.from(document.querySelectorAll('button, a')).filter(
                                el => el.innerText && (
                                    el.innerText.toLowerCase().includes('show more') || 
                                    el.innerText.toLowerCase().includes('показать больше') ||
                                    el.innerText.toLowerCase().includes('load more') ||
                                    el.innerText.toLowerCase().includes('загрузить еще')
                                )
                            );
                            showMoreButtons.forEach(button => button.click());
                            
                            // Раскрыть все свернутые элементы
                            const expandableElements = Array.from(document.querySelectorAll('[aria-expanded="false"]'));
                            expandableElements.forEach(el => {
                                el.setAttribute('aria-expanded', 'true');
                                el.click();
                            });
                            
                            // Прокрутка для ленивой загрузки
                            window.scrollTo(0, document.body.scrollHeight / 2);
                            setTimeout(() => {
                                window.scrollTo(0, document.body.scrollHeight);
                            }, 500);
                        }''')
                        
                        # Дополнительное ожидание после скриптов
                        await page.wait_for_timeout(2000)
                        
                        # Получение HTML после всех манипуляций
                        html_content = await page.content()
                        
                        # Анализ AJAX-данных
                        ajax_data = None
                        if self.record_ajax and ajax_calls:
                            # Извлекаем JSON из ответов
                            for call in ajax_calls:
                                if call.get('response') and call['response'].get('json'):
                                    json_responses[call['url']] = call['response']['json']
                            
                            # Анализируем данные AJAX
                            ajax_data = {
                                'api_calls': ajax_calls,
                                'json_responses': json_responses,
                                'structured_data': self._extract_ajax_data(ajax_calls, json_responses)
                            }
                        
                        # Определяем, является ли сайт SPA
                        is_spa = False
                        try:
                            spa_check = await page.evaluate('''() => {
                                return {
                                    spa: Boolean(window.history && window.history.pushState) ||
                                         Boolean(document.querySelector('[ng-app],[data-reactroot],[id="app"],[id="root"]'))
                                };
                            }''')
                            is_spa = spa_check.get('spa', False) or bool(ajax_calls)
                        except Exception:
                            is_spa = bool(ajax_calls)
                        
                        return html_content, ajax_data, is_spa
                        
                    except Exception as e:
                        logger.error(f"Ошибка при рендеринге {url}: {str(e)}")
                        
                        # Пробуем повторно, если не достигнут лимит повторов
                        if retry_count < self.max_retries:
                            logger.info(f"Повторная попытка {retry_count+1}/{self.max_retries}")
                            return await self._render_page(url, retry_count+1)
                        
                        return None, None, False
                    
                    finally:
                        await page.close()
                        await context.close()
                
                finally:
                    await browser.close()
        
        except Exception as e:
            logger.error(f"Критическая ошибка при рендеринге {url}: {str(e)}")
            return None, None, False
    
    async def crawl_site(self, base_url: str, max_pages: int = 10, cache_enabled: bool = True) -> Dict[str, Any]:
        """
        Сканирует сайт, анализируя все страницы.
        
        Args:
            base_url: Начальный URL для сканирования
            max_pages: Максимальное количество страниц
            cache_enabled: Использовать ли кэш
            
        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        result = {
            "base_url": base_url,
            "success": False,
            "pages": {},
            "error": None
        }
        
        try:
            # Сначала определяем тип сайта
            logger.info(f"Определение типа сайта для {base_url}")
            initial_result = await self.analyze_url(base_url, use_cache=cache_enabled)
            
            if not initial_result["success"]:
                result["error"] = initial_result.get("error", "Failed to analyze base URL")
                return result
            
            # Сохраняем информацию о типе сайта
            result["site_type"] = initial_result["site_type"]
            result["pages"][base_url] = initial_result
            
            # Проверяем наличие внутренних ссылок для сканирования
            to_crawl = []
            if "content" in initial_result and "internal_links" in initial_result["content"]:
                to_crawl = initial_result["content"]["internal_links"][:max_pages-1]
            
            # Если нет внутренних ссылок, пробуем извлечь их из HTML
            if not to_crawl and "content" in initial_result:
                html_content = initial_result.get("html", "")
                if html_content:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        # Преобразуем относительные ссылки в абсолютные
                        if href.startswith('/'):
                            href = f"{base_url.rstrip('/')}{href}"
                        # Проверяем, что ссылка на том же домене
                        if base_url.split('/')[2] in href:
                            to_crawl.append(href)
                
                # Ограничиваем количество страниц
                to_crawl = list(set(to_crawl))[:max_pages-1]
            
            # Сканируем остальные страницы
            for i, url in enumerate(to_crawl):
                logger.info(f"Анализ страницы {i+2}/{max_pages}: {url}")
                page_result = await self.analyze_url(url, use_cache=cache_enabled)
                result["pages"][url] = page_result
            
            result["success"] = True
            result["total_pages"] = len(result["pages"])
            
        except Exception as e:
            logger.error(f"Ошибка при сканировании сайта {base_url}: {str(e)}")
            result["error"] = str(e)
            
        return result
    
    def analyze_url_sync(self, url: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Синхронная обертка для analyze_url.
        
        Args:
            url: URL для анализа
            use_cache: Использовать ли кэш
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        return asyncio.run(self.analyze_url(url, use_cache))
    
    def crawl_site_sync(self, base_url: str, max_pages: int = 10, cache_enabled: bool = True) -> Dict[str, Any]:
        """
        Синхронная обертка для crawl_site.
        
        Args:
            base_url: Начальный URL для сканирования
            max_pages: Максимальное количество страниц
            cache_enabled: Использовать ли кэш
            
        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        return asyncio.run(self.crawl_site(base_url, max_pages, cache_enabled))
