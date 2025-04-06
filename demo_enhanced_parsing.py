"""
Демонстрация улучшенного парсера с поддержкой SPA, AJAX и кэширования.
"""

import argparse
import logging
import json
import sys
import os
import time

# Патчим модули перед импортом
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
from seo_ai_models.parsers.extractors.meta_extractor_update import update_meta_extractor
update_meta_extractor(MetaExtractor)

# Импортируем интеграцию улучшений
from seo_ai_models.parsers.integrate_all_improvements import integrate_all_improvements

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Демонстрация улучшенного парсера")
    parser.add_argument("url", help="URL для анализа")
    parser.add_argument("--force-spa", action="store_true", help="Принудительно использовать режим SPA")
    parser.add_argument("--output", help="Файл для сохранения результатов")
    parser.add_argument("--cache-dir", default=".cache", help="Директория для кэша")
    parser.add_argument("--clear-cache", action="store_true", help="Очистить кэш перед анализом")
    parser.add_argument("--use-cache", action="store_true", help="Использовать кэш")
    parser.add_argument("--ajax", action="store_true", help="Анализировать AJAX-запросы")
    
    args = parser.parse_args()
    
    try:
        # Получаем обновленные компоненты
        improvements = integrate_all_improvements()
        
        # Создаем директорию кэша
        os.makedirs(args.cache_dir, exist_ok=True)
        
        # Используем улучшенный конвейер парсинга
        if args.use_cache:
            # Создаем конвейер с кэшированием
            pipeline = improvements['CachedAdaptiveParsingPipeline'](
                force_spa_mode=args.force_spa,
                cache_enabled=True,
                cache_dir=args.cache_dir,
                cache_max_age=86400  # 24 часа
            )
            
            if args.clear_cache:
                pipeline.clear_cache()
                logger.info("Кэш очищен")
        else:
            # Создаем обычный улучшенный конвейер
            pipeline = improvements['get_enhanced_pipeline'](
                force_spa_mode=args.force_spa
            )
        
        # Анализируем URL
        logger.info(f"Анализ URL: {args.url}")
        start_time = time.time()
        
        if args.ajax:
            # Используем SPAContentExtractorWithAJAX для анализа
            extractor = improvements['SPAContentExtractorWithAJAX'](
                headless=True,
                wait_for_idle=2000,
                wait_for_timeout=10000
            )
            result = extractor.extract_content_from_url(args.url)
        else:
            # Используем стандартный анализ
            result = pipeline.analyze_url(args.url)
        
        elapsed = time.time() - start_time
        
        # Выводим результаты
        print(f"URL: {args.url}")
        print(f"Время анализа: {elapsed:.2f} сек")
        
        if hasattr(pipeline, 'detect_site_type'):
            site_type = pipeline.detect_site_type(args.url)
            print(f"Тип сайта: {'SPA' if site_type.get('is_spa', False) else 'Обычный'}")
            
            if 'detected_frameworks' in site_type and site_type['detected_frameworks']:
                print(f"Обнаруженные фреймворки: {', '.join(site_type['detected_frameworks'])}")
        
        if isinstance(result, dict) and result.get('content'):
            content = result['content']
            
            # Выводим заголовок
            if 'title' in content:
                print(f"\nЗаголовок: {content['title']}")
            
            # Количество заголовков на странице
            if 'headings' in content:
                heading_count = sum(len(values) for key, values in content['headings'].items())
                print(f"Количество заголовков на странице: {heading_count}")
            
            # Общая длина текста
            if 'content' in content and 'all_text' in content['content']:
                text_length = len(content['content']['all_text'])
                print(f"Общая длина текста: {text_length} символов")
            
            # AJAX-данные, если есть
            if 'ajax_data' in result:
                ajax_data = result['ajax_data']
                
                api_calls = ajax_data.get('api_calls', [])
                if api_calls:
                    print(f"\nОбнаружено {len(api_calls)} AJAX-запросов:")
                    for i, call in enumerate(api_calls[:3], 1):  # Показываем первые 3
                        print(f"  {i}. {call['method']} {call['url']}")
                    
                    if len(api_calls) > 3:
                        print(f"  ...и еще {len(api_calls) - 3} запросов")
                
                structured_data = ajax_data.get('structured_data', {})
                if structured_data and 'entities' in structured_data:
                    entities = structured_data['entities']
                    if entities:
                        print("\nОбнаружены сущности:")
                        for entity_type, info in entities.items():
                            print(f"  - {entity_type}: {info['count']} записей")
        
        # Сохраняем результат в файл, если указано
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены в {args.output}")
        
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
