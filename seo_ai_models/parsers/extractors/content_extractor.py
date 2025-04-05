"""
Content Extractor модуль для проекта SEO AI Models.
Предоставляет функциональность для извлечения текста и структуры из HTML.
"""

import logging
import re
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag, NavigableString

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentExtractor:
    """
    Извлекает значимый контент из HTML-страниц, включая текст, заголовки,
    параграфы и другие структурные элементы.
    """
    
    def __init__(
        self,
        content_tags: List[str] = None,
        block_tags: List[str] = None,
        exclude_classes: List[str] = None,
        exclude_ids: List[str] = None
    ):
        """
        Инициализация ContentExtractor.

        Args:
            content_tags: HTML-теги, которые обычно содержат основной контент
            block_tags: Теги, которые представляют блочные элементы (для извлечения структуры)
            exclude_classes: CSS-классы для исключения из извлечения
            exclude_ids: HTML-идентификаторы для исключения из извлечения
        """
        self.content_tags = content_tags or [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
            'ul', 'ol', 'li', 'table', 'article', 'section', 'main'
        ]
        
        self.block_tags = block_tags or [
            'article', 'section', 'div', 'main', 'header', 'footer',
            'nav', 'aside', 'form', 'table'
        ]
        
        self.exclude_classes = exclude_classes or [
            'advertisement', 'ads', 'banner', 'menu', 'nav', 'sidebar',
            'footer', 'comment', 'cookie', 'popup', 'modal'
        ]
        
        self.exclude_ids = exclude_ids or [
            'advertisement', 'ads', 'banner', 'menu', 'nav', 'sidebar',
            'footer', 'comment', 'cookie', 'popup', 'modal'
        ]
        
        # Индикаторы качества контента
        self.text_density_threshold = 0.5  # Отношение текста к HTML
        self.min_content_length = 150      # Минимальная длина контента, чтобы считаться значимым
        
    def _is_excluded_element(self, element: Tag) -> bool:
        """
        Проверка, должен ли элемент быть исключен из извлечения.
        
        Args:
            element: BeautifulSoup Tag для проверки
            
        Returns:
            bool: True, если элемент должен быть исключен
        """
        if not hasattr(element, 'attrs'):
            return False
            
        # Проверка классов
        if element.get('class'):
            for cls in element.get('class', []):
                if any(excl.lower() in cls.lower() for excl in self.exclude_classes):
                    return True
                    
        # Проверка идентификаторов
        if element.get('id'):
            element_id = element.get('id', '').lower()
            if any(excl.lower() in element_id for excl in self.exclude_ids):
                return True
                
        return False
    
    def _calculate_text_density(self, element: Tag) -> float:
        """
        Расчет плотности текста для элемента (отношение текста к HTML).
        
        Args:
            element: BeautifulSoup Tag для анализа
            
        Returns:
            float: Отношение плотности текста
        """
        html_length = len(str(element))
        if html_length == 0:
            return 0
            
        text_length = len(element.get_text(strip=True))
        return text_length / html_length
    
    def _clean_text(self, text: str) -> str:
        """
        Очистка извлеченного текста путем удаления лишних пробелов и нормализации пробелов.
        
        Args:
            text: Текст для очистки
            
        Returns:
            str: Очищенный текст
        """
        # Заменяем несколько пробелов одним
        text = re.sub(r'\s+', ' ', text)
        # Удаляем начальные/конечные пробелы
        text = text.strip()
        # Удаляем более двух последовательных переносов строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _extract_element_structure(self, element: Tag, depth: int = 0) -> Dict[str, Any]:
        """
        Извлечение структурной информации об элементе.
        
        Args:
            element: BeautifulSoup Tag для анализа
            depth: Текущая глубина рекурсии
            
        Returns:
            Dict: Информация о структуре элемента
        """
        if isinstance(element, NavigableString):
            return None
            
        if self._is_excluded_element(element):
            return None
            
        tag_name = element.name
        
        # Пропускаем неконтентные теги на глубине > 0
        if depth > 0 and tag_name not in self.content_tags + self.block_tags:
            return None
            
        # Базовая информация об элементе
        element_info = {
            "tag": tag_name,
            "text": self._clean_text(element.get_text(strip=True)),
            "attributes": {}
        }
        
        # Добавляем выбранные атрибуты
        important_attrs = ['id', 'class', 'href', 'src', 'alt', 'title']
        for attr in important_attrs:
            if element.has_attr(attr):
                element_info["attributes"][attr] = element[attr]
                
        # Добавляем дочерние элементы
        children = []
        for child in element.children:
            if isinstance(child, Tag):
                child_structure = self._extract_element_structure(child, depth + 1)
                if child_structure:
                    children.append(child_structure)
                    
        if children:
            element_info["children"] = children
            
        return element_info
    
    def extract_content(self, html_content: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Извлечение структурированного контента из HTML.
        
        Args:
            html_content: HTML-контент для парсинга
            url: URL контента (для ссылки)
            
        Returns:
            Dict: Информация об извлеченном контенте
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Удаление элементов script и style
        for script in soup(["script", "style", "noscript", "iframe"]):
            script.decompose()
            
        # Удаление комментариев
        for comment in soup.find_all(text=lambda text: isinstance(text, NavigableString) and text.strip().startswith('<!--')):
            comment.extract()
            
        # Извлечение основного текстового контента
        all_text = soup.get_text(strip=True)
        
        # Извлечение заголовка
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            
        # Извлечение заголовков
        headings = {}
        for i in range(1, 7):
            heading_tags = soup.find_all(f'h{i}')
            headings[f'h{i}'] = [self._clean_text(h.get_text(strip=True)) for h in heading_tags]
            
        # Попытка найти область основного контента
        main_content_element = None
        potential_elements = []
        
        # Поиск общих контейнеров контента
        for tag in ['main', 'article', 'div', 'section']:
            elements = soup.find_all(tag)
            for element in elements:
                if self._is_excluded_element(element):
                    continue
                    
                text_length = len(element.get_text(strip=True))
                text_density = self._calculate_text_density(element)
                
                if text_length > self.min_content_length and text_density > self.text_density_threshold:
                    potential_elements.append({
                        'element': element,
                        'text_length': text_length,
                        'text_density': text_density,
                        'score': text_length * text_density
                    })
                    
        if potential_elements:
            # Сортировка по оценке (длина текста * плотность текста)
            potential_elements.sort(key=lambda x: x['score'], reverse=True)
            main_content_element = potential_elements[0]['element']
        
        # Извлечение параграфов
        paragraphs = []
        if main_content_element:
            p_tags = main_content_element.find_all('p')
        else:
            p_tags = soup.find_all('p')
            
        for p in p_tags:
            if not self._is_excluded_element(p):
                text = self._clean_text(p.get_text(strip=True))
                if text:
                    paragraphs.append(text)
        
        # Извлечение списков
        lists = []
        for list_tag in soup.find_all(['ul', 'ol']):
            if not self._is_excluded_element(list_tag):
                list_items = []
                for li in list_tag.find_all('li'):
                    text = self._clean_text(li.get_text(strip=True))
                    if text:
                        list_items.append(text)
                if list_items:
                    lists.append({
                        'type': list_tag.name,
                        'items': list_items
                    })
        
        # Извлечение структурной информации
        structure = None
        if main_content_element:
            structure = self._extract_element_structure(main_content_element)
            
        # Сборка результата
        domain = ""
        path = ""
        if url:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            path = parsed_url.path
            
        # Построение окончательного результата извлечения контента
        result = {
            'url': url,
            'domain': domain,
            'path': path,
            'title': title,
            'headings': headings,
            'content': {
                'all_text': all_text,
                'paragraphs': paragraphs,
                'lists': lists,
            },
            'metadata': {
                'text_length': len(all_text),
                'paragraph_count': len(paragraphs),
                'list_count': len(lists),
                'heading_counts': {key: len(values) for key, values in headings.items()},
            }
        }
        
        if structure:
            result['structure'] = structure
            
        return result
