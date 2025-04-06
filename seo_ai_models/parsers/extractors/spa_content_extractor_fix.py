"""
Исправление для SPAContentExtractor.
"""

import logging
import asyncio
from playwright.async_api import async_playwright

from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_spa_content_extractor(extractor_class):
    """
    Исправляет SPAContentExtractor, добавляя недостающие методы.
    
    Args:
        extractor_class: Класс SPAContentExtractor для исправления
        
    Returns:
        Исправленный класс
    """
    class PlaywrightContext:
        """
        Контекстный менеджер для работы с Playwright.
        """
        def __init__(self):
            self.playwright = None
            
        async def __aenter__(self):
            self.playwright = await async_playwright().start()
            return self.playwright
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.playwright:
                await self.playwright.stop()
    
    class BrowserContext:
        """
        Контекстный менеджер для работы с браузером.
        """
        def __init__(self, spa_extractor, playwright):
            self.spa_extractor = spa_extractor
            self.playwright = playwright
            self.browser = None
            
        async def __aenter__(self):
            # Выбор браузера в зависимости от настройки
            if self.spa_extractor.browser_type == "firefox":
                browser_instance = self.playwright.firefox
            elif self.spa_extractor.browser_type == "webkit":
                browser_instance = self.playwright.webkit
            else:
                browser_instance = self.playwright.chromium  # По умолчанию
                
            # Запускаем браузер
            self.browser = await browser_instance.launch(headless=self.spa_extractor.headless)
            return self.browser
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.browser:
                await self.browser.close()
    
    # Исправляем _render_page
    async def _render_page_fixed(self, url):
        """
        Исправленная версия _render_page с корректным управлением ресурсами.
        
        Args:
            url: URL для рендеринга
            
        Returns:
            str: Отрендеренный HTML
        """
        logger.info(f"Рендеринг страницы с JavaScript: {url}")
        
        async with PlaywrightContext() as playwright:
            async with BrowserContext(self, playwright) as browser:
                context = await browser.new_context(viewport={'width': 1366, 'height': 768})
                page = await context.new_page()
                
                try:
                    # Переход на страницу и ожидание загрузки
                    await page.goto(url, wait_until='networkidle', timeout=self.wait_for_timeout)
                    
                    # Дополнительная задержка для полной загрузки динамического контента
                    await page.wait_for_timeout(self.wait_for_idle)
                    
                    # Ожидание наличия ключевых селекторов контента (любого из списка)
                    for selector in self.wait_for_selectors:
                        try:
                            await page.wait_for_selector(selector, timeout=1000)
                            break  # Если найден хотя бы один селектор, выходим из цикла
                        except Exception:
                            continue  # Если не найден, пробуем следующий селектор
                    
                    # Получение полного HTML после рендеринга
                    html_content = await page.content()
                    
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
                    }''')
                    
                    # Ожидание дополнительного контента после раскрытия
                    await page.wait_for_timeout(1000)
                    
                    # Получение окончательного HTML после всех манипуляций
                    final_html = await page.content()
                    
                    return final_html
                    
                except Exception as e:
                    logger.error(f"Ошибка при рендеринге {url}: {str(e)}")
                    return None
                
                finally:
                    await page.close()
                    await context.close()
    
    # Также исправляем метод render_page_with_ajax в spa_content_extractor_ajax.py
    async def _render_page_with_ajax_fixed(self, url):
        """
        Исправленная версия _render_page_with_ajax.
        
        Args:
            url: URL для рендеринга
            
        Returns:
            str: Отрендеренный HTML
        """
        logger.info(f"Рендеринг SPA с перехватом AJAX: {url}")
        
        async with PlaywrightContext() as playwright:
            async with BrowserContext(self, playwright) as browser:
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
                    logger.error(f"Ошибка при AJAX-анализе {url}: {str(e)}")
                    self._last_error = str(e)
                    return None
                    
                finally:
                    await page.close()
                    await context.close()
    
    # Добавляем классы и методы в extractor_class
    extractor_class.PlaywrightContext = PlaywrightContext
    extractor_class.BrowserContext = BrowserContext
    extractor_class._render_page = _render_page_fixed
    
    # Проверяем и заменяем метод render_page_with_ajax в ajax-версии
    if hasattr(extractor_class, '_render_page_with_ajax'):
        extractor_class._render_page_with_ajax = _render_page_with_ajax_fixed
    
    return extractor_class
