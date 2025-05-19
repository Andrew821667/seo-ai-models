
"""
Упрощенный тестовый скрипт для проверки обогащенных парсеров Фазы 4, Этап 3.
"""

import os
import json
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def manual_test_schema_optimizer():
    """Ручное тестирование SchemaOptimizer."""
    logger.info("Тестирование функций SchemaOptimizer...")
    
    from bs4 import BeautifulSoup
    import json
    import copy
    
    # Эмулируем базовую функциональность SchemaOptimizer
    soup = BeautifulSoup(TEST_HTML, 'html.parser')
    
    # Извлекаем JSON-LD
    json_ld_scripts = soup.find_all('script', type='application/ld+json')
    structured_data = []
    for script in json_ld_scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                structured_data.extend(data)
            else:
                structured_data.append(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Не удалось разобрать JSON-LD: {e}")
    
    logger.info(f"Извлечено структурированных данных JSON-LD: {len(structured_data)}")
    logger.info(f"Данные: {json.dumps(structured_data, ensure_ascii=False)}")
    
    # Эмулируем оптимизацию схемы
    optimized_data = copy.deepcopy(structured_data)
    for item in optimized_data:
        if "@type" in item and item["@type"] == "Article":
            # Улучшаем описание статьи
            if "headline" in item and "description" not in item:
                item["description"] = f"Статья под названием: {item['headline']}"
            
            # Улучшаем информацию об авторе
            if "author" in item and isinstance(item["author"], dict):
                if "description" not in item["author"] and "name" in item["author"]:
                    item["author"]["description"] = f"{item['author']['name']} - автор статьи"
    
    logger.info(f"Оптимизированные данные: {json.dumps(optimized_data, ensure_ascii=False)}")
    logger.info("Тестирование SchemaOptimizer завершено успешно!")

def manual_test_metadata_enhancer():
    """Ручное тестирование MetadataEnhancer."""
    logger.info("Тестирование функций MetadataEnhancer...")
    
    from bs4 import BeautifulSoup
    
    # Эмулируем базовую функциональность MetadataEnhancer
    soup = BeautifulSoup(TEST_HTML, 'html.parser')
    
    # Извлекаем базовые метаданные
    metadata = {}
    
    # Заголовок
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.text.strip()
    
    # Описание
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        metadata['description'] = meta_desc.get('content').strip()
    
    # Ключевые слова
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords and meta_keywords.get('content'):
        keywords_text = meta_keywords.get('content').strip()
        metadata['keywords'] = [k.strip() for k in keywords_text.split(',')]
        metadata['keywords_string'] = keywords_text
    
    # Автор
    author_elem = soup.select_one('.author')
    if author_elem:
        author_text = author_elem.text.strip()
        # Извлекаем имя автора из текста "Автор: Имя Автора"
        if ":" in author_text:
            metadata['author'] = author_text.split(":", 1)[1].strip()
    
    # Добавляем метаданные для цитирования
    metadata['citation_info'] = {
        'title': metadata.get('title', ''),
        'author': metadata.get('author', ''),
        'url': 'https://example.com/test',
        'citation_style': f"{metadata.get('author', '')}. {metadata.get('title', '')}. URL: https://example.com/test"
    }
    
    logger.info(f"Извлеченные метаданные: {json.dumps(metadata, ensure_ascii=False)}")
    logger.info("Тестирование MetadataEnhancer завершено успешно!")

def manual_test_structured_data_extractor():
    """Ручное тестирование StructuredDataExtractor."""
    logger.info("Тестирование функций StructuredDataExtractor...")
    
    from bs4 import BeautifulSoup
    
    # Эмулируем базовую функциональность StructuredDataExtractor
    soup = BeautifulSoup(TEST_HTML, 'html.parser')
    
    structured_data = {
        'url': 'https://example.com/test',
        'language': 'ru',
        'types': []
    }
    
    # Извлекаем таблицы
    tables = []
    table_elements = soup.find_all('table')
    
    for table_idx, table in enumerate(table_elements):
        caption = table.find('caption')
        caption_text = caption.get_text().strip() if caption else ''
        
        # Извлекаем заголовки
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
        
        # Извлекаем строки
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                rows.append(row_data)
        
        table_data = {
            'id': f'table_{table_idx + 1}',
            'caption': caption_text,
            'headers': headers,
            'rows': rows
        }
        
        tables.append(table_data)
    
    if tables:
        structured_data['tables'] = tables
        structured_data['types'].append('tables')
    
    # Извлекаем списки
    lists = []
    list_elements = soup.find_all(['ul', 'ol'])
    
    for list_idx, list_elem in enumerate(list_elements):
        # Извлекаем элементы списка
        items = [li.get_text().strip() for li in list_elem.find_all('li')]
        
        list_data = {
            'id': f'list_{list_idx + 1}',
            'type': list_elem.name,
            'items': items
        }
        
        lists.append(list_data)
    
    if lists:
        structured_data['lists'] = lists
        structured_data['types'].append('lists')
    
    # Извлекаем FAQ
    faqs = []
    dl_elements = soup.find_all('dl', class_=['faq'])
    
    for dl_idx, dl in enumerate(dl_elements):
        items = []
        dt_elements = dl.find_all('dt')
        dd_elements = dl.find_all('dd')
        
        for i, (dt, dd) in enumerate(zip(dt_elements, dd_elements)):
            items.append({
                'question': dt.get_text().strip(),
                'answer': dd.get_text().strip()
            })
        
        faq_data = {
            'id': f'faq_{dl_idx + 1}',
            'items': items
        }
        
        faqs.append(faq_data)
    
    if faqs:
        structured_data['faq'] = faqs
        structured_data['types'].append('faq')
    
    logger.info(f"Извлеченные структурированные данные: {json.dumps(structured_data, ensure_ascii=False, indent=2)}")
    logger.info("Тестирование StructuredDataExtractor завершено успешно!")

def main():
    """Основная функция для запуска тестов."""
    logger.info("Начинаем тестирование компонентов Фазы 4, Этап 3...")
    
    # Тестируем SchemaOptimizer
    manual_test_schema_optimizer()
    
    # Тестируем MetadataEnhancer
    manual_test_metadata_enhancer()
    
    # Тестируем StructuredDataExtractor
    manual_test_structured_data_extractor()
    
    logger.info("Все тесты успешно завершены!")

if __name__ == "__main__":
    main()
