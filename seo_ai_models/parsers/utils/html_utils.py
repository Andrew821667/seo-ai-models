"""
HTML утилиты для SEO AI Models парсеров.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from bs4 import BeautifulSoup, Tag

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_html(html: str) -> str:
    """
    Очистка HTML путем удаления скриптов, стилей и комментариев.
    
    Args:
        html: HTML-контент для очистки
        
    Returns:
        str: Очищенный HTML
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Удаление элементов script и style
    for element in soup(['script', 'style', 'iframe', 'noscript']):
        element.decompose()
        
    # Удаление комментариев
    for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
        comment.extract()
        
    return str(soup)

def extract_text_from_html(html: str) -> str:
    """
    Извлечение обычного текста из HTML-контента.
    
    Args:
        html: HTML-контент
        
    Returns:
        str: Извлеченный текст
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Удаление элементов script и style
    for element in soup(['script', 'style', 'iframe', 'noscript']):
        element.decompose()
        
    # Получение текста и нормализация пробелов
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def find_main_content_area(soup: BeautifulSoup) -> Optional[Tag]:
    """
    Поиск главной области контента в HTML-документе.
    Использует эвристики для определения основного контента.
    
    Args:
        soup: BeautifulSoup объект
        
    Returns:
        Optional[Tag]: Элемент основного контента или None
    """
    # Поиск распространенных элементов контейнеров контента
    content_elements = []
    
    # Оценка на основе типа элемента и атрибутов
    scores = {
        'main': 10,
        'article': 9,
        'section': 7,
        'div.content': 8,
        'div.main-content': 8,
        'div.post': 8,
        'div.entry': 8,
        'div.blog-post': 8,
    }
    
    # Поиск потенциальных элементов контента
    for tag_name in ['main', 'article', 'section', 'div']:
        for element in soup.find_all(tag_name):
            score = 0
            
            # Базовая оценка по тегу
            if tag_name in scores:
                score += scores[tag_name]
                
            # Оценка по классам
            if 'class' in element.attrs:
                classes = ' '.join(element.get('class', []))
                for class_pattern, class_score in [
                    ('content', 3), 
                    ('main', 3), 
                    ('article', 3), 
                    ('post', 2), 
                    ('entry', 2), 
                    ('body', 1),
                    ('text', 1)
                ]:
                    if class_pattern in classes.lower():
                        score += class_score
                        
            # Оценка по ID
            if 'id' in element.attrs:
                element_id = element['id'].lower()
                for id_pattern, id_score in [
                    ('content', 4), 
                    ('main', 4), 
                    ('article', 3), 
                    ('post', 2), 
                    ('entry', 2), 
                    ('body', 1)
                ]:
                    if id_pattern in element_id:
                        score += id_score
                        
            # Оценка по контенту
            text_length = len(element.get_text(strip=True))
            if text_length > 200:
                score += 2
                
            # Оценка по количеству параграфов
            p_count = len(element.find_all('p'))
            score += min(p_count // 2, 5)  # Максимум 5 баллов из параграфов
            
            # Добавление в потенциальные элементы контента
            if score > 0:
                content_elements.append((element, score, text_length))
                
    # Сортировка по оценке и длине текста
    content_elements.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Возвращение лучшего кандидата, если он найден
    if content_elements:
        return content_elements[0][0]
        
    return None

def extract_visible_text(element: Tag) -> str:
    """
    Извлечение видимого текста из элемента BeautifulSoup.
    
    Args:
        element: BeautifulSoup Tag
        
    Returns:
        str: Видимый текст
    """
    texts = []
    for text in element.stripped_strings:
        texts.append(text)
    return ' '.join(texts)

def is_boilerplate(element: Tag) -> bool:
    """
    Проверка, является ли элемент, вероятно, шаблонным контентом.
    
    Args:
        element: BeautifulSoup Tag
        
    Returns:
        bool: True, если элемент, вероятно, является шаблоном
    """
    # Проверка ID и классов элемента
    boilerplate_patterns = [
        'header', 'footer', 'sidebar', 'menu', 'nav', 'navigation',
        'comment', 'widget', 'ad', 'banner', 'share', 'social',
        'related', 'popular', 'recommended', 'promo'
    ]
    
    # Проверка id элемента
    if element.get('id'):
        if any(pattern in element['id'].lower() for pattern in boilerplate_patterns):
            return True
            
    # Проверка классов элемента
    if element.get('class'):
        classes = ' '.join(element.get('class', []))
        if any(pattern in classes.lower() for pattern in boilerplate_patterns):
            return True
            
    return False
