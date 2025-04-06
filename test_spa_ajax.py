"""
Тест для парсинга SPA с поддержкой AJAX.
"""

import asyncio
import logging
import sys
import json

from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor
from seo_ai_models.parsers.extractors.spa_content_extractor_fix import fix_spa_content_extractor
from seo_ai_models.parsers.extractors.ajax_interceptor import AJAXInterceptor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Исправляем SPAContentExtractor
FixedSPAContentExtractor = fix_spa_content_extractor(SPAContentExtractor)

async def test_render_page(url):
    """
    Тестирует рендеринг SPA-страницы.
    
    Args:
        url: URL для тестирования
    """
    logger.info(f"Тестирование рендеринга SPA для {url}")
    
    extractor = FixedSPAContentExtractor(
        headless=True,
        wait_for_idle=2000,
        wait_for_timeout=10000
    )
    
    # Рендеринг страницы
    html = await extractor._render_page(url)
    
    if html:
        logger.info(f"Успешно отрендерена страница длиной {len(html)} символов")
        # Извлекаем заголовок страницы
        if "<title>" in html and "</title>" in html:
            title = html.split("<title>")[1].split("</title>")[0]
            logger.info(f"Заголовок страницы: {title}")
    else:
        logger.error("Не удалось отрендерить страницу")

def main():
    if len(sys.argv) < 2:
        print("Использование: python test_spa_ajax.py URL")
        return 1
        
    url = sys.argv[1]
    asyncio.run(test_render_page(url))
    return 0

if __name__ == "__main__":
    sys.exit(main())
