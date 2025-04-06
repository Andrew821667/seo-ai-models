"""
Финальный патч для объединения всех исправлений.
"""

import sys
import asyncio
import logging
from playwright.async_api import async_playwright

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импортируем оригинальные классы
from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor
from seo_ai_models.parsers.extractors.spa_content_extractor_ajax import update_spa_extractor_with_ajax
from seo_ai_models.parsers.extractors.spa_content_extractor_fix import fix_spa_content_extractor
from seo_ai_models.parsers.extractors.ajax_interceptor import AJAXInterceptor

# Применяем все исправления
Fixed_SPAContentExtractor = fix_spa_content_extractor(SPAContentExtractor)

# Создаем патч для AJAX-экстрактора с прямой реализацией (без зависимостей)
class SPAContentExtractorWithAJAX(Fixed_SPAContentExtractor):
    """
    SPA Content Extractor с поддержкой перехвата AJAX-запросов.
    """
    
    def __init__(
        self,
        content_tags=None,
        block_tags=None,
        exclude_classes=None,
        exclude_ids=None,
        wait_for_idle=2000,
        wait_for_timeout=10000,
        wait_for_selectors=None,
        headless=True,
        browser_type="chromium",
        record_api_calls=True,
        analyze_responses=True,
        api_patterns=None
    ):
        """
        Инициализация SPAContentExtractorWithAJAX.
        """
        super().__init__(
            content_tags=content_tags,
            block_tags=block_tags,
            exclude_classes=exclude_classes,
            exclude_ids=exclude_ids,
            wait_for_idle=wait_for_idle,
            wait_for_timeout=wait_for_timeout,
            wait_for_selectors=wait_for_selectors,
            headless=headless,
            browser_type=browser_type
        )
        
        # Инициализируем AJAX-перехватчик
        self.ajax_interceptor = AJAXInterceptor(
            record_api_calls=record_api_calls,
            analyze_responses=analyze_responses,
            api_patterns=api_patterns
        )
        
        # Для хранения последних данных
        self._last_ajax_data = None
        self._last_error = None
    
    async def _render_page_with_ajax(self, url):
        """
        Рендерит страницу с перехватом AJAX-запросов.
        
        Args:
            url: URL для рендеринга
            
        Returns:
            str: Отрендеренный HTML
        """
        logger.info(f"Рендеринг SPA с перехватом AJAX: {url}")
        
        async with self.PlaywrightContext() as playwright:
            async with self.BrowserContext(self, playwright) as browser:
                context = await browser.new_context(viewport={'width': 1366, 'height': 768})
                page = await context.new_page()
                
                try:
                    # Устанавливаем перехват AJAX
                    await self.ajax_interceptor.setup_request_interception(page)
                    
                    # Стандартные действия по рендерингу
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
                    
                    # Дополнительное время для AJAX
                    await page.wait_for_timeout(2000)
                    
                    # Получаем HTML
                    html = await page.content()
                    
                    # Сохраняем AJAX-данные
                    self._last_ajax_data = {
                        'api_calls': self.ajax_interceptor.get_api_calls(),
                        'json_responses': self.ajax_interceptor.get_json_responses(),
                        'structured_data': self.ajax_interceptor.extract_data_from_responses()
                    }
                    
                    return html
                    
                except Exception as e:
                    logger.error(f"Ошибка при AJAX-анализе {url}: {str(e)}")
                    self._last_error = str(e)
                    return None
                    
                finally:
                    await page.close()
                    await context.close()
    
    async def extract_content_from_url_async(self, url):
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
                "error": self._last_error or "Failed to render page content"
            }
        
        # Получаем стандартный результат
        result = self.extract_content(html_content, url)
        
        # Добавляем данные AJAX, если они есть
        if self._last_ajax_data:
            result['ajax_data'] = self._last_ajax_data
        
        return result
    
    def get_ajax_data(self):
        """
        Получает данные, перехваченные из AJAX-запросов.
        
        Returns:
            Dict: Данные AJAX или None
        """
        return self._last_ajax_data

# Экспортируем исправленные классы
__all__ = [
    'Fixed_SPAContentExtractor',
    'SPAContentExtractorWithAJAX'
]

# Переопределяем имя для доступа извне
Fixed_SPAContentExtractorWithAJAX = SPAContentExtractorWithAJAX
