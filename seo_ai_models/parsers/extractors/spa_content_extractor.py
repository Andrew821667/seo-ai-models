"""
SPA Content Extractor модуль для проекта SEO AI Models.
Предоставляет функциональность для извлечения текста и структуры из JavaScript-рендеринга.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag, NavigableString
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeoutError

from seo_ai_models.parsers.extractors.content_extractor import ContentExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPAContentExtractor(ContentExtractor):
    """
    Извлекает значимый контент из SPA-страниц с поддержкой JavaScript,
    включая текст, заголовки, параграфы и другие структурные элементы.
    """
    
    def __init__(
        self,
        content_tags: List[str] = None,
        block_tags: List[str] = None,
        exclude_classes: List[str] = None,
        exclude_ids: List[str] = None,
        wait_for_idle: int = 2000,  # мс ожидания загрузки страницы после событий idle
        wait_for_timeout: int = 5000,  # мс максимального ожидания загрузки
        wait_for_selectors: List[str] = None,  # CSS-селекторы для ожидания
        headless: bool = True,
        browser_type: str = "chromium"
    ):
        """
        Инициализация SPAContentExtractor.

        Args:
            content_tags: HTML-теги, которые обычно содержат основной контент
            block_tags: Теги, которые представляют блочные элементы (для извлечения структуры)
            exclude_classes: CSS-классы для исключения из извлечения
            exclude_ids: HTML-идентификаторы для исключения из извлечения
            wait_for_idle: Время ожидания в мс после событий 'networkidle'
            wait_for_timeout: Максимальное время ожидания в мс для загрузки страницы
            wait_for_selectors: CSS-селекторы для ожидания перед извлечением контента
            headless: Запускать ли браузер в режиме headless
            browser_type: Тип браузера для использования ('chromium', 'firefox', 'webkit')
        """
        super().__init__(content_tags, block_tags, exclude_classes, exclude_ids)
        
        self.wait_for_idle = wait_for_idle
        self.wait_for_timeout = wait_for_timeout
        self.wait_for_selectors = wait_for_selectors or [
            'main', 'article', '#content', '.content', 'body'
        ]
        self.headless = headless
        self.browser_type = browser_type
        
    async def _render_page(self, url: str) -> Optional[str]:
        """
        Рендеринг страницы с выполнением JavaScript и получение HTML.
        
        Args:
            url: URL для рендеринга
            
        Returns:
            Optional[str]: Отрендеренный HTML или None при ошибке
        """
        logger.info(f"Рендеринг страницы с JavaScript: {url}")
        
        async with async_playwright() as p:
            # Выбор браузера в зависимости от настройки
            if self.browser_type == "firefox":
                browser_instance = p.firefox
            elif self.browser_type == "webkit":
                browser_instance = p.webkit
            else:
                browser_instance = p.chromium  # По умолчанию
                
            browser = await browser_instance.launch(headless=self.headless)
            
            try:
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
                        except PlaywrightTimeoutError:
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
                    
            finally:
                await browser.close()
    
    async def extract_content_from_url_async(self, url: str) -> Dict[str, Any]:
        """
        Асинхронный метод извлечения структурированного контента из URL с поддержкой SPA.
        
        Args:
            url: URL для анализа
            
        Returns:
            Dict: Информация об извлеченном контенте
        """
        try:
            html_content = await self._render_page(url)
            if not html_content:
                return {
                    "url": url,
                    "error": "Failed to render page content"
                }
                
            return self.extract_content(html_content, url)
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении контента из {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e)
            }
    
    def extract_content_from_url(self, url: str) -> Dict[str, Any]:
        """
        Синхронный метод извлечения структурированного контента из URL с поддержкой SPA.
        
        Args:
            url: URL для анализа
            
        Returns:
            Dict: Информация об извлеченном контенте
        """
        return asyncio.run(self.extract_content_from_url_async(url))
