"""
Тест для перехвата AJAX-запросов в SPA-приложениях.
"""

import asyncio
import logging
import sys
import json
import time
from pprint import pprint

from seo_ai_models.parsers.extractors.spa_content_extractor import SPAContentExtractor
from seo_ai_models.parsers.extractors.spa_content_extractor_fix import fix_spa_content_extractor
from seo_ai_models.parsers.extractors.ajax_interceptor import AJAXInterceptor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Исправляем SPAContentExtractor
FixedSPAContentExtractor = fix_spa_content_extractor(SPAContentExtractor)

class SpaAjaxAnalyzer:
    """
    Анализатор SPA-приложений с поддержкой AJAX-перехвата.
    """
    
    def __init__(self, headless=True, wait_timeout=10000, wait_idle=2000):
        self.headless = headless
        self.wait_timeout = wait_timeout
        self.wait_idle = wait_idle
        
        # Создаем AJAX-перехватчик
        self.ajax_interceptor = AJAXInterceptor(
            record_api_calls=True,
            analyze_responses=True
        )
        
        # Создаем экстрактор с исправлениями
        self.extractor = FixedSPAContentExtractor(
            headless=headless,
            wait_for_idle=wait_idle,
            wait_for_timeout=wait_timeout
        )
    
    async def analyze_page(self, url):
        """
        Анализирует SPA-страницу с перехватом AJAX-запросов.
        
        Args:
            url: URL для анализа
            
        Returns:
            dict: Результаты анализа
        """
        logger.info(f"Анализ SPA-страницы с AJAX: {url}")
        start_time = time.time()
        
        # Рендерим страницу через Playwright
        async with self.extractor.PlaywrightContext() as playwright:
            async with self.extractor.BrowserContext(self.extractor, playwright) as browser:
                context = await browser.new_context(viewport={'width': 1366, 'height': 768})
                page = await context.new_page()
                
                try:
                    # Устанавливаем перехват AJAX перед загрузкой страницы
                    await self.ajax_interceptor.setup_request_interception(page)
                    
                    # Загружаем страницу и ожидаем загрузки
                    await page.goto(url, wait_until='networkidle', timeout=self.wait_timeout)
                    await page.wait_for_timeout(self.wait_idle)
                    
                    # Выполняем действия для активации дополнительных AJAX-запросов
                    await page.evaluate('''() => {
                        // Нажимаем на кнопки "Показать больше"
                        const showMoreButtons = Array.from(document.querySelectorAll('button, a')).filter(
                            el => el.innerText && (
                                el.innerText.toLowerCase().includes('show more') || 
                                el.innerText.toLowerCase().includes('load more')
                            )
                        );
                        showMoreButtons.forEach(button => button.click());
                        
                        // Прокручиваем страницу для активации ленивой загрузки
                        window.scrollTo(0, document.body.scrollHeight / 2);
                        setTimeout(() => {
                            window.scrollTo(0, document.body.scrollHeight);
                        }, 500);
                    }''')
                    
                    # Дополнительное ожидание для завершения AJAX-запросов
                    await page.wait_for_timeout(3000)
                    
                    # Получаем HTML страницы
                    html = await page.content()
                    
                    # Получаем заголовок страницы
                    title = await page.title()
                    
                    # Получаем данные AJAX-перехватчика
                    api_calls = self.ajax_interceptor.get_api_calls()
                    json_responses = self.ajax_interceptor.get_json_responses()
                    structured_data = self.ajax_interceptor.extract_data_from_responses()
                    
                    elapsed = time.time() - start_time
                    
                    return {
                        "success": True,
                        "url": url,
                        "title": title,
                        "html_length": len(html),
                        "elapsed_time": elapsed,
                        "ajax_data": {
                            "api_calls": api_calls,
                            "json_responses": json_responses,
                            "structured_data": structured_data
                        }
                    }
                
                except Exception as e:
                    logger.error(f"Ошибка при анализе {url}: {str(e)}")
                    return {
                        "success": False,
                        "url": url,
                        "error": str(e)
                    }
                
                finally:
                    await page.close()
                    await context.close()

def main():
    if len(sys.argv) < 2:
        print("Использование: python test_ajax_support.py URL [output_file]")
        return 1
        
    url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = SpaAjaxAnalyzer()
    result = asyncio.run(analyzer.analyze_page(url))
    
    if result["success"]:
        print(f"Успешный анализ URL: {url}")
        print(f"Заголовок: {result['title']}")
        print(f"Длина HTML: {result['html_length']} символов")
        print(f"Время анализа: {result['elapsed_time']:.2f} сек")
        
        # Выводим информацию об AJAX-запросах
        ajax_data = result.get("ajax_data", {})
        api_calls = ajax_data.get("api_calls", [])
        
        if api_calls:
            print(f"\nПерехвачено {len(api_calls)} AJAX-запросов:")
            for i, call in enumerate(api_calls[:5], 1):
                print(f"  {i}. {call['method']} {call['url']}")
                
            if len(api_calls) > 5:
                print(f"  ...и еще {len(api_calls) - 5} запросов")
        else:
            print("\nAJAX-запросы не обнаружены")
        
        # Информация о структурированных данных
        structured_data = ajax_data.get("structured_data", {})
        if structured_data and 'entities' in structured_data:
            entities = structured_data['entities']
            if entities:
                print("\nОбнаружены сущности:")
                for entity_type, info in entities.items():
                    print(f"  - {entity_type}: {info['count']} записей")
                    print(f"    Поля: {', '.join(info['fields'][:5])}")
                    if len(info['fields']) > 5:
                        print(f"    ...и еще {len(info['fields']) - 5} полей")
        
        # Сохраняем результат в файл, если указано
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nРезультаты сохранены в {output_file}")
    else:
        print(f"Ошибка при анализе {url}: {result.get('error', 'Неизвестная ошибка')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
