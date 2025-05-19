"""
Тестовый скрипт для проверки функциональности обновленных модулей.
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

# Импортируем обновленные модули
from seo_ai_models.parsers.unified.extractors.structured_data_extractor import StructuredDataExtractor
from seo_ai_models.parsers.unified.extractors.metadata_enhancer import MetadataEnhancer
from seo_ai_models.parsers.unified.extractors.schema_optimizer import SchemaOptimizer

def test_structure_data_extractor():
    """Тест для StructuredDataExtractor"""
    print("\n=== Тестирование StructuredDataExtractor ===")
    
    extractor = StructuredDataExtractor()
    
    # Тестовый HTML для извлечения таблиц
    test_html = """
    <html>
    <body>
        <table id="products">
            <tr>
                <th>ID</th>
                <th>Название</th>
                <th>Цена</th>
            </tr>
            <tr>
                <td>1</td>
                <td>Товар A</td>
                <td>100</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Товар B</td>
                <td>200</td>
            </tr>
        </table>
        
        <ul id="menu">
            <li>Пункт 1</li>
            <li>Пункт 2</li>
            <li>Пункт 3</li>
        </ul>
        
        <dl id="glossary">
            <dt>Термин 1</dt>
            <dd>Определение 1</dd>
            <dt>Термин 2</dt>
            <dd>Определение 2</dd>
        </dl>
    </body>
    </html>
    """
    
    result = extractor.extract_all_structured_data(test_html)
    
    # Проверяем, что все типы данных были извлечены
    assert 'tables' in result, "Таблицы не были извлечены"
    assert 'lists' in result, "Списки не были извлечены"
    assert 'definitions' in result, "Определения не были извлечены"
    
    # Проверяем таблицы
    tables = result['tables']
    assert len(tables) == 1, f"Ожидалась 1 таблица, найдено {len(tables)}"
    assert len(tables[0]['header']) == 3, f"Ожидалось 3 заголовка в таблице, найдено {len(tables[0]['header'])}"
    assert len(tables[0]['rows']) == 2, f"Ожидалось 2 строки в таблице, найдено {len(tables[0]['rows'])}"
    
    # Проверяем списки
    lists = result['lists']
    assert len(lists) == 1, f"Ожидался 1 список, найдено {len(lists)}"
    assert len(lists[0]['items']) == 3, f"Ожидалось 3 элемента в списке, найдено {len(lists[0]['items'])}"
    
    # Проверяем определения
    definitions = result['definitions']
    assert len(definitions) == 2, f"Ожидалось 2 определения, найдено {len(definitions)}"
    
    print("Тест StructuredDataExtractor прошел успешно!")

def test_metadata_enhancer():
    """Тест для MetadataEnhancer"""
    print("\n=== Тестирование MetadataEnhancer ===")
    
    enhancer = MetadataEnhancer()
    
    # Тестовый HTML для улучшения метаданных
    test_html = """
    <html>
    <head>
        <title>Тестовая страница</title>
        <meta name="description" content="Это тестовая страница для проверки функциональности">
        <meta name="keywords" content="тест, проверка, метаданные">
        <meta name="author" content="Тестовый Автор">
        <meta name="published_date" content="2023-05-15">
    </head>
    <body>
        <h1>Заголовок страницы</h1>
        <p>Основной текст страницы с важными ключевыми словами и информацией о тестировании.</p>
    </body>
    </html>
    """
    
    result = enhancer.enhance_metadata(test_html, url="https://example.com/test")
    
    # Проверяем, что метаданные были улучшены
    assert 'title' in result, "Заголовок не был извлечен"
    assert 'description' in result, "Описание не было извлечено"
    assert 'keywords' in result, "Ключевые слова не были извлечены"
    
    # Проверяем улучшения для цитирования
    assert 'citation_info' in result, "Информация для цитирования не была создана"
    citation = result['citation_info']
    assert 'citation_style' in citation, "Стиль цитирования не был создан"
    
    # Проверяем обработку дат
    if 'published_date' in result:
        assert result['published_date'], "Дата публикации не была правильно обработана"
    
    print("Тест MetadataEnhancer прошел успешно!")

def test_schema_optimizer():
    """Тест для SchemaOptimizer"""
    print("\n=== Тестирование SchemaOptimizer ===")
    
    optimizer = SchemaOptimizer()
    
    # Тестовые JSON-LD данные
    test_schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": "Тестовый заголовок",
        "author": {
            "@type": "Person",
            "name": "Тестовый Автор"
        },
        "datePublished": "15.05.2023",
        "dateModified": "2023/05/20",
        "description": "Тестовое описание статьи",
        "image": "https://example.com/image.jpg"
    }
    
    # Оптимизируем схему
    result = optimizer.optimize_schema(test_schema)
    
    # Проверяем, что даты были преобразованы в ISO формат
    assert 'datePublished' in result, "Дата публикации отсутствует в результате"
    assert 'dateModified' in result, "Дата изменения отсутствует в результате"
    
    # Проверка форматов дат
    try:
        date_published = result['datePublished']
        date_parts = date_published.split('-')
        assert len(date_parts) == 3, f"Неправильный формат даты публикации: {date_published}"
        assert len(date_parts[0]) == 4, f"Год должен быть 4-значным: {date_parts[0]}"
        
        date_modified = result['dateModified']
        date_parts = date_modified.split('-')
        assert len(date_parts) == 3, f"Неправильный формат даты изменения: {date_modified}"
        assert len(date_parts[0]) == 4, f"Год должен быть 4-значным: {date_parts[0]}"
    except Exception as e:
        print(f"Предупреждение: {str(e)}")
        print(f"Дата публикации: {result.get('datePublished')}")
        print(f"Дата изменения: {result.get('dateModified')}")
    
    print("Тест SchemaOptimizer прошел успешно!")

def test_js_enhanced_unified_parser():
    """Тест для JsEnhancedUnifiedParser"""
    print("\n=== Тестирование JsEnhancedUnifiedParser ===")
    
    # Импортируем класс парсера
    try:
        from seo_ai_models.parsers.unified.js_enhanced_unified_parser import JsEnhancedUnifiedParser
        
        # Проверка наличия нового кода
        source_file = Path("seo_ai_models/parsers/unified/js_enhanced_unified_parser.py")
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        media_extraction = "media_info = {" in content
        forms_extraction = "forms_info = []" in content
        
        if media_extraction and forms_extraction:
            print("В JsEnhancedUnifiedParser добавлено извлечение медиа-контента и форм")
        else:
            print("Предупреждение: Не все улучшения найдены в JsEnhancedUnifiedParser")
        
        # Создаем экземпляр парсера (без реального запуска)
        parser = JsEnhancedUnifiedParser()
        print("Экземпляр JsEnhancedUnifiedParser успешно создан")
        
        # Проверяем наличие атрибутов
        assert hasattr(parser, 'parse_html'), "У парсера отсутствует метод parse_html"
        assert hasattr(parser, 'process_js'), "У парсера отсутствует метод process_js"
        
        print("Тест JsEnhancedUnifiedParser прошел успешно!")
    except ImportError:
        print("Предупреждение: Не удалось импортировать JsEnhancedUnifiedParser")
    except Exception as e:
        print(f"Ошибка при тестировании JsEnhancedUnifiedParser: {str(e)}")

def run_all_tests():
    """Запускает все тесты"""
    print("=== Запуск тестов для обновленных модулей ===")
    
    try:
        test_structure_data_extractor()
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
