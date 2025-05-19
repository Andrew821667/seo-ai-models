
"""
Тестовый скрипт для проверки обогащенных парсеров, созданных в Фазе 4, Этап 3.
"""

import sys
import os
from bs4 import BeautifulSoup
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импортируем компоненты для тестирования
sys.path.append(os.getcwd())

from seo_ai_models.parsers.unified.extractors.schema_optimizer import SchemaOptimizer
from seo_ai_models.parsers.unified.extractors.metadata_enhancer import MetadataEnhancer
from seo_ai_models.parsers.unified.extractors.structured_data_extractor import StructuredDataExtractor

# Тестовый HTML контент
TEST_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Тестовая страница</title>
    <meta name="description" content="Описание тестовой страницы для проверки обогащенных парсеров">
    <meta name="keywords" content="тест, парсеры, LLM, оптимизация">
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": "Тестовая статья",
        "datePublished": "2025-05-19T12:00:00+00:00",
        "author": {
            "@type": "Person",
            "name": "Иван Петров"
        }
    }
    </script>
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#">Главная</a></li>
                <li><a href="#">О нас</a></li>
                <li><a href="#">Контакты</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <article>
            <h1>Заголовок статьи</h1>
            <div class="author">Автор: Иван Петров</div>
            <div class="date">Опубликовано: 19.05.2025</div>
            
            <p>Это первый абзац статьи, который содержит важную информацию о теме. 
            Здесь мы рассматриваем основные аспекты и приводим ключевые факты.</p>
            
            <h2>Подзаголовок первого раздела</h2>
            <p>Более подробное описание первого аспекта темы. Этот абзац содержит 
            дополнительные детали и примеры.</p>
            
            <ul>
                <li>Первый пункт списка</li>
                <li>Второй пункт списка</li>
                <li>Третий пункт списка</li>
            </ul>
            
            <h2>Подзаголовок второго раздела</h2>
            <p>Описание второго аспекта темы. Здесь приводятся аргументы и факты,
            подтверждающие основную мысль статьи.</p>
            
            <table>
                <caption>Таблица с данными</caption>
                <thead>
                    <tr>
                        <th>Название</th>
                        <th>Значение</th>
                        <th>Примечание</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Первый элемент</td>
                        <td>10</td>
                        <td>Базовое значение</td>
                    </tr>
                    <tr>
                        <td>Второй элемент</td>
                        <td>20</td>
                        <td>Улучшенное значение</td>
                    </tr>
                </tbody>
            </table>
            
            <blockquote>
                <p>Это цитата из важного источника, которая подтверждает основную мысль статьи.</p>
                <cite>Известный автор</cite>
            </blockquote>
            
            <h2>Часто задаваемые вопросы</h2>
            <dl class="faq">
                <dt>Вопрос 1: Что такое обогащенные парсеры?</dt>
                <dd>Ответ 1: Обогащенные парсеры - это компоненты, которые извлекают и улучшают структурированные данные из веб-страниц для лучшего понимания LLM.</dd>
                
                <dt>Вопрос 2: Для чего нужна оптимизация под LLM?</dt>
                <dd>Ответ 2: Оптимизация под LLM позволяет улучшить распознавание и цитирование контента в больших языковых моделях.</dd>
            </dl>
            
            <h2>Заключение</h2>
            <p>В данной статье мы рассмотрели основные аспекты темы и привели ключевые аргументы.
            Надеемся, что представленная информация была полезной и познавательной.</p>
        </article>
    </main>
    
    <footer>
        <p>&copy; 2025 Тестовый сайт. Все права защищены.</p>
    </footer>
</body>
</html>
"""

def test_schema_optimizer():
    """Тестирование SchemaOptimizer."""
    logger.info("Тестирование SchemaOptimizer...")
    
    optimizer = SchemaOptimizer(language='ru')
    
    # Извлечение структурированных данных
    structured_data = optimizer.extract_structured_data(TEST_HTML)
    logger.info(f"Извлечено структурированных данных: {len(structured_data)}")
    
    # Оптимизация схемы
    optimized_data = optimizer.optimize_schema(structured_data)
    logger.info(f"Оптимизированные данные: {json.dumps(optimized_data, ensure_ascii=False, indent=2)}")
    
    # Оптимизация HTML
    optimized_html = optimizer.optimize_html_for_llm(TEST_HTML)
    logger.info(f"Размер оптимизированного HTML: {len(optimized_html)} байт")
    
    logger.info("Тестирование SchemaOptimizer завершено успешно!")

def test_metadata_enhancer():
    """Тестирование MetadataEnhancer."""
    logger.info("Тестирование MetadataEnhancer...")
    
    enhancer = MetadataEnhancer(language='ru')
    
    # Улучшение метаданных
    enhanced_metadata = enhancer.enhance_metadata(TEST_HTML, url="https://example.com/test")
    logger.info(f"Улучшенные метаданные: {json.dumps(enhanced_metadata, ensure_ascii=False, indent=2)}")
    
    # Применение улучшенных метаданных
    enhanced_html = enhancer.apply_enhanced_metadata(TEST_HTML, enhanced_metadata)
    logger.info(f"Размер HTML с улучшенными метаданными: {len(enhanced_html)} байт")
    
    logger.info("Тестирование MetadataEnhancer завершено успешно!")

def test_structured_data_extractor():
    """Тестирование StructuredDataExtractor."""
    logger.info("Тестирование StructuredDataExtractor...")
    
    extractor = StructuredDataExtractor(language='ru')
    
    # Извлечение всех структурированных данных
    structured_data = extractor.extract_all_structured_data(TEST_HTML, url="https://example.com/test")
    
    logger.info(f"Извлеченные типы данных: {structured_data.get('types', [])}")
    if 'tables' in structured_data:
        logger.info(f"Извлечено таблиц: {len(structured_data['tables'])}")
    if 'lists' in structured_data:
        logger.info(f"Извлечено списков: {len(structured_data['lists'])}")
    if 'faq' in structured_data:
        logger.info(f"Извлечено FAQ: {len(structured_data['faq'])}")
    
    logger.info(f"Структурированные данные: {json.dumps(structured_data, ensure_ascii=False, indent=2)}")
    
    logger.info("Тестирование StructuredDataExtractor завершено успешно!")

def main():
    """Основная функция для запуска тестов."""
    logger.info("Начинаем тестирование компонентов Фазы 4, Этап 3...")
    
    # Тестируем SchemaOptimizer
    test_schema_optimizer()
    
    # Тестируем MetadataEnhancer
    test_metadata_enhancer()
    
    # Тестируем StructuredDataExtractor
    test_structured_data_extractor()
    
    logger.info("Все тесты успешно завершены!")

if __name__ == "__main__":
    main()
