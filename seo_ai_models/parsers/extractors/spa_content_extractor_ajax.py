"""
Обновление SPAContentExtractor с поддержкой перехвата AJAX-запросов.
"""

from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor
from seo_ai_models.parsers.extractors.ajax_interceptor import AJAXInterceptor

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
    
    async def render_page_with_ajax(self, url):
        """
        Рендерит страницу и перехватывает AJAX-запросы.
        
        Args:
            url: URL для рендеринга
            
        Returns:
            str: Отрендеренный HTML
        """
        # Используем оригинальный метод с небольшой модификацией
        async with self._get_browser() as browser:
            context = await browser.new_context(viewport={'width': 1366, 'height': 768})
            page = await context.new_page()
            
            try:
                # Устанавливаем перехват AJAX перед загрузкой страницы
                await self.ajax_interceptor.setup_request_interception(page)
                
                # Загружаем страницу и выполняем стандартные действия
                await page.goto(url, wait_until='networkidle', timeout=self.wait_for_timeout)
                await page.wait_for_timeout(self.wait_for_idle)
                
                # Ждем селекторы
                for selector in self.wait_for_selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=1000)
                        break
                    except Exception:
                        continue
                
                # Выполняем дополнительные скрипты
                await page.evaluate('''() => {
                    // Стандартный код для раскрытия контента
                    const showMoreButtons = Array.from(document.querySelectorAll('button, a')).filter(
                        el => el.innerText && (
                            el.innerText.toLowerCase().includes('show more') || 
                            el.innerText.toLowerCase().includes('показать больше') ||
                            el.innerText.toLowerCase().includes('load more') ||
                            el.innerText.toLowerCase().includes('загрузить еще')
                        )
                    );
                    showMoreButtons.forEach(button => button.click());
                    
                    const expandableElements = Array.from(document.querySelectorAll('[aria-expanded="false"]'));
                    expandableElements.forEach(el => {
                        el.setAttribute('aria-expanded', 'true');
                        el.click();
                    });
                }''')
                
                # Дополнительное время для обработки AJAX-запросов
                await page.wait_for_timeout(2000)
                
                # Получаем окончательный HTML
                html = await page.content()
                
                # Сохраняем AJAX-данные
                self._last_ajax_data = {
                    'api_calls': self.ajax_interceptor.get_api_calls(),
                    'json_responses': self.ajax_interceptor.get_json_responses(),
                    'structured_data': self.ajax_interceptor.extract_data_from_responses()
                }
                
                return html
                
            except Exception as e:
                self._last_error = str(e)
                return None
                
            finally:
                await page.close()
    
    # Заменяем методы в классе
    extractor_class.__init__ = __init_with_ajax
    
    # Не заменяем _render_page напрямую, а добавляем новый метод
    extractor_class._render_page_with_ajax = render_page_with_ajax
    
    # Добавляем метод для получения данных AJAX
    def get_ajax_data(self):
        """
        Получает данные, перехваченные из AJAX-запросов.
        
        Returns:
            Dict: Данные AJAX или None
        """
        return getattr(self, '_last_ajax_data', None)
    
    extractor_class.get_ajax_data = get_ajax_data
    
    # Обновляем extract_content_from_url_async для использования AJAX
    original_extract_async = extractor_class.extract_content_from_url_async
    
    async def extract_content_with_ajax_async(self, url):
        """
        Извлекает контент с перехватом AJAX-запросов.
        
        Args:
            url: URL для анализа
            
        Returns:
            Dict: Результат с контентом и AJAX-данными
        """
        # Используем новый метод рендеринга с AJAX
        html_content = await self._render_page_with_ajax(url)
        
        if not html_content:
            return {
                "url": url,
                "error": getattr(self, '_last_error', "Failed to render page content")
            }
        
        # Получаем стандартный результат
        result = self.extract_content(html_content, url)
        
        # Добавляем данные AJAX, если они есть
        ajax_data = self.get_ajax_data()
        if ajax_data:
            result['ajax_data'] = ajax_data
        
        return result
    
    extractor_class.extract_content_from_url_async = extract_content_with_ajax_async
    
    return extractor_class
