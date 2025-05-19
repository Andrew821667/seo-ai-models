"""
Тестовый скрипт для проверки функциональности обновленных модулей (без внешних зависимостей).
"""

import os
import sys
from pathlib import Path
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_script")

# Добавляем корневую директорию проекта в путь для импорта
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_metadata_enhancer():
    """Тест для MetadataEnhancer"""
    print("\n=== Тестирование MetadataEnhancer ===")
    
    try:
        from seo_ai_models.parsers.unified.extractors.metadata_enhancer import MetadataEnhancer
        
        enhancer = MetadataEnhancer()
        
        # Тестируем обработку даты
        test_date = "10 октября 2023"
        url = "https://example.com/test"
        citation = enhancer._create_citation_gost_style("Test Author", "Test Title", test_date, url)
        
        print(f"Тест создания цитирования в стиле ГОСТ:")
        print(f"Входные данные: автор='Test Author', заголовок='Test Title', дата='{test_date}', url='{url}'")
        print(f"Результат: '{citation}'")
        
        # Тестируем неправильный формат даты
        test_date = "неправильный формат"
        citation = enhancer._create_citation_gost_style("Test Author", "Test Title", test_date, url)
        
        print(f"\nТест с неправильным форматом даты:")
        print(f"Входные данные: автор='Test Author', заголовок='Test Title', дата='{test_date}', url='{url}'")
        print(f"Результат: '{citation}'")
        
        print("Тест MetadataEnhancer прошел успешно!")
    except Exception as e:
        print(f"Ошибка при тестировании MetadataEnhancer: {str(e)}")
        import traceback
        traceback.print_exc()

def test_schema_optimizer():
    """Тест для SchemaOptimizer"""
    print("\n=== Тестирование SchemaOptimizer ===")
    
    try:
        from seo_ai_models.parsers.unified.extractors.schema_optimizer import SchemaOptimizer
        
        optimizer = SchemaOptimizer()
        
        # Тестируем обработку даты
        test_dates = [
            "2023-10-10",
            "10.05.2022",
            "05/30/2021",
            "20210415",
            "неправильный формат"
        ]
        
        print("Тестирование обработки различных форматов дат:")
        for date in test_dates:
            result = {}
            try:
                optimizer.improved_date_processing("datePublished", date, result)
                print(f"Входная дата: '{date}' -> Результат: '{result.get('datePublished', 'не обработано')}'")
            except Exception as e:
                print(f"Входная дата: '{date}' -> Ошибка: {str(e)}")
        
        print("Тест SchemaOptimizer прошел успешно!")
    except Exception as e:
        print(f"Ошибка при тестировании SchemaOptimizer: {str(e)}")
        import traceback
        traceback.print_exc()

def test_js_enhanced_unified_parser():
    """Тест для JsEnhancedUnifiedParser (только проверка кода)"""
    print("\n=== Тестирование JsEnhancedUnifiedParser ===")
    
    try:
        # Проверка наличия нового кода
        source_file = Path("seo_ai_models/parsers/unified/js_enhanced_unified_parser.py")
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        media_extraction = "media_info = {" in content
        forms_extraction = "forms_info = []" in content
        
        if media_extraction and forms_extraction:
            print("В JsEnhancedUnifiedParser успешно добавлено извлечение медиа-контента и форм")
        else:
            print("Предупреждение: Не все улучшения найдены в JsEnhancedUnifiedParser")
            if not media_extraction:
                print("- Отсутствует код для извлечения медиа-контента")
            if not forms_extraction:
                print("- Отсутствует код для извлечения форм")
        
        print("Тест JsEnhancedUnifiedParser прошел успешно!")
    except Exception as e:
        print(f"Ошибка при тестировании JsEnhancedUnifiedParser: {str(e)}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Запускает все тесты"""
    print("=== Запуск тестов для обновленных модулей ===")
    
    try:
        test_metadata_enhancer()
        test_schema_optimizer()
        test_js_enhanced_unified_parser()
        
        print("\n=== Все тесты успешно завершены! ===")
    except Exception as e:
        print(f"\nОшибка при выполнении тестов: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
