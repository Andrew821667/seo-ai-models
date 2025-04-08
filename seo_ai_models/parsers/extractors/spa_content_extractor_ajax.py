"""
Обновление SPAContentExtractor с поддержкой перехвата AJAX-запросов.
"""

from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor
from seo_ai_models.parsers.extractors.ajax_interceptor import AJAXInterceptor
from playwright.async_api import async_playwright

def update_spa_extractor_with_ajax(extractor_class):
    """
    Обновляет SPAContentExtractor для поддержки перехвата AJAX-запросов.
    
    Args:
        extractor_class: Класс SPAContentExtractor для обновления
        
    Returns:
        Обновленный класс с поддержкой AJAX
    """
    original_init = extractor_class.__init__
    original_render_page = extractor_class._render_page
    
    def __init_with_ajax(self, *args, **kwargs):
        # Извлекаем параметры для AJAX
        record_api_calls = kwargs.pop('record_api_calls', True)
        analyze_responses = kwargs.pop('analyze_responses', True)
        api_patterns = kwargs.pop('api_patterns', None)
        
        # Вызываем оригинальный конструктор
        original_init(self, *args, **kwargs)
        
        # Инициализируем AJAX-перехватчик
        self.ajax_interceptor = AJAXInterceptor(
            record_api_calls=record_api_calls,
            analyze_responses=analyze_responses,
            api_patterns=api_patterns
        )
        self.playwright = None
    
    async def render_page_with_ajax(self, url):
        """
        Рендерит страницу и перехватывает AJAX-запросы.
        
        Args:
            url: URL для рендеринга
            
        Returns:
            str: Отрендеренный HTML
        """
        try:
            # Инициализируем playwright
            self.playwright = await async_playwright().start()
            
            # Получаем браузер
            if self.browser_type == "firefox":
                browser_instance = self.playwright.firefox
            elif self.browser_type == "webkit":
                browser_instance = self.playwright.webkit
            else:
                browser_instance = self.playwright.chromium
                
            browser = await browser_instance.launch(headless=self.headless)
            
            try:
                # Создаем новый контекст и страницу
                context = await browser.new_context(viewport={'width': 1366, 'height': 768})
                page = await context.new_page()
                
                # Настраиваем перехват запросов на странице
                await self.ajax_interceptor.setup_request_interception(page)
                
                try:
                    # Ожидание загрузки страницы
                    await page.goto(url, wait_until='networkidle', timeout=self.wait_for_timeout)
                    await page.wait_for_timeout(self.wait_for_idle)
                    
                    # Получаем HTML после рендеринга
                    html_content = await page.content()
                    
                    # Собираем данные AJAX
                    api_calls = self.ajax_interceptor.get_api_calls()
                    structured_data = self.ajax_interceptor.extract_data_from_responses()
                    
                    # Готовим результат
                    result = {
                        'html': html_content,
                        'ajax_data': {
                            'api_calls': api_calls,
                            'structured_data': structured_data
                        }
                    }
                    
                    return result
                    
                except Exception as e:
                    raise Exception(f"Error during page rendering: {str(e)}")
                finally:
                    await page.close()
                    await context.close()
            finally:
                await browser.close()
        finally:
            if self.playwright:
                await self.playwright.stop()
    
    async def extract_content_with_ajax_async(self, url):
        """
        Извлекает контент и AJAX-данные из URL.
        
        Args:
            url: URL для извлечения
            
        Returns:
            Dict: Контент и данные AJAX
        """
        try:
            # Рендерим страницу с перехватом AJAX
            result = await self.render_page_with_ajax(url)
            
            if not result or 'html' not in result:
                return {
                    'url': url,
                    'error': 'Failed to render page with AJAX'
                }
            
            # Извлекаем контент из HTML
            content_data = self.extract_content(result['html'], url)
            
            # Добавляем AJAX-данные к результату
            content_data['ajax_data'] = result.get('ajax_data', {})
            
            return content_data
            
        except Exception as e:
            return {
                'url': url,
                'error': str(e)
            }
    
    def extract_content_with_ajax(self, url):
        """
        Синхронная обертка для извлечения контента и AJAX-данных.
        
        Args:
            url: URL для извлечения
            
        Returns:
            Dict: Контент и данные AJAX
        """
        import asyncio
        return asyncio.run(self.extract_content_with_ajax_async(url))
    
    # Создаем новый класс с дополнительными методами
    class SPAContentExtractorWithAJAX(extractor_class):
        __init__ = __init_with_ajax
        render_page_with_ajax = render_page_with_ajax
        extract_content_with_ajax_async = extract_content_with_ajax_async
        extract_content_with_ajax = extract_content_with_ajax
        extract_content_from_url_with_ajax = extract_content_with_ajax
    
    return SPAContentExtractorWithAJAX
