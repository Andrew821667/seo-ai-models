"""
Простой тест для проверки функциональности парсинга SPA.
"""

import argparse
import logging
import sys

# Патчим модули перед импортом
from patch_parsers import update_meta_extractor
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor
update_meta_extractor(MetaExtractor)

from seo_ai_models.parsers.adaptive_parsing_pipeline import AdaptiveParsingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Тест парсинга SPA-сайтов")
    parser.add_argument("url", help="URL для анализа")
    parser.add_argument("--force-spa", action="store_true", help="Принудительно использовать режим SPA")
    
    args = parser.parse_args()
    
    try:
        pipeline = AdaptiveParsingPipeline(force_spa_mode=args.force_spa)
        
        # Определение типа сайта
        logger.info(f"Определение типа сайта для {args.url}")
        site_type = pipeline.detect_site_type(args.url)
        
        print(f"URL: {args.url}")
        print(f"Тип сайта: {'SPA' if site_type['is_spa'] else 'Обычный'}")
        print(f"Уверенность: {site_type.get('confidence', 0):.2f}")
        
        if 'detected_frameworks' in site_type:
            frameworks = site_type.get('detected_frameworks', [])
            if frameworks:
                print(f"Обнаруженные фреймворки: {', '.join(frameworks)}")
        
        # Анализ URL
        logger.info(f"Анализ содержимого {args.url}")
        
        # Сначала извлекаем только метаданные, чтобы не перегружать вывод
        pipeline.meta_extractor.extract_from_url = lambda url: {"title": "Test", "description": "Test"}
        
        result = pipeline.analyze_url(args.url)
        
        if result["success"]:
            print("\nУспешно проанализирован URL!")
            
            if "content" in result and result["content"]:
                content = result["content"]
                
                # Вывод заголовка
                print(f"Заголовок: {content.get('title', 'Не найден')}")
                
                # Вывод основного содержимого (первый параграф)
                paragraphs = content.get('content', {}).get('paragraphs', [])
                if paragraphs and len(paragraphs) > 0:
                    print(f"\nНачало контента: {paragraphs[0][:100]}...")
                
                # Количество заголовков
                headings = content.get('headings', {})
                heading_count = sum(len(values) for key, values in headings.items())
                print(f"Количество заголовков: {heading_count}")
                
                # Общая длина текста
                all_text = content.get('content', {}).get('all_text', '')
                text_length = len(all_text) if all_text else 0
                print(f"Общая длина текста: {text_length} символов")
        else:
            print(f"Ошибка при анализе URL: {result.get('error', 'неизвестная ошибка')}")
    
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
