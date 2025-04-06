"""
Простой тест для проверки исправлений.
"""

import asyncio
import logging
import sys
import time

# Используем только исправленные классы
from final_fix import Fixed_SPAContentExtractor, Fixed_SPAContentExtractorWithAJAX

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_spa(url, use_ajax=False):
    """
    Анализирует SPA-страницу.
    
    Args:
        url: URL для анализа
        use_ajax: Использовать ли перехват AJAX
    """
    logger.info(f"Анализ SPA-страницы: {url}")
    start_time = time.time()
    
    if use_ajax:
        extractor = Fixed_SPAContentExtractorWithAJAX(
            headless=True,
            wait_for_idle=2000,
            wait_for_timeout=10000
        )
    else:
        extractor = Fixed_SPAContentExtractor(
            headless=True,
            wait_for_idle=2000,
            wait_for_timeout=10000
        )
    
    # Запускаем анализ
    result = await extractor.extract_content_from_url_async(url)
    
    elapsed = time.time() - start_time
    logger.info(f"Анализ завершен за {elapsed:.2f} сек")
    
    # Выводим основную информацию
    print(f"URL: {url}")
    print(f"Время анализа: {elapsed:.2f} сек")
    
    if 'title' in result:
        print(f"Заголовок: {result['title']}")
    
    if 'content' in result and 'all_text' in result['content']:
        text_length = len(result['content']['all_text'])
        print(f"Общая длина текста: {text_length} символов")
    
    if use_ajax and 'ajax_data' in result:
        ajax_data = result['ajax_data']
        api_calls = ajax_data.get('api_calls', [])
        
        if api_calls:
            print(f"\nПерехвачено {len(api_calls)} AJAX-запросов:")
            for i, call in enumerate(api_calls[:3], 1):
                print(f"  {i}. {call['method']} {call['url']}")
                
            if len(api_calls) > 3:
                print(f"  ...и еще {len(api_calls) - 3} запросов")
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Использование: python simple_spa_test.py URL [--ajax]")
        return 1
        
    url = sys.argv[1]
    use_ajax = "--ajax" in sys.argv
    
    print(f"Тестирование {'с AJAX-перехватом' if use_ajax else 'без AJAX-перехвата'}")
    asyncio.run(analyze_spa(url, use_ajax))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
