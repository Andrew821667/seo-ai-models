
"""
StructuredDataExtractor - компонент для извлечения структурированных 
данных из контента для LLM-анализа с поддержкой русского языка.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from urllib.parse import urljoin
from datetime import datetime

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class StructuredDataExtractor:
    """
    Извлекает структурированные данные из HTML-контента для LLM-анализа.
    
    Поддерживает извлечение и анализ различных типов структурированных данных,
    включая таблицы, списки, определения и сложные структуры с учетом
    специфики русскоязычного контента.
    """
    
    def __init__(self, 
                 extract_tables: bool = True,
                 extract_lists: bool = True,
                 extract_definitions: bool = True,
                 extract_code_blocks: bool = True,
                 extract_quotes: bool = True,
                 extract_faq: bool = True,
                 language: str = 'ru'):
        """
        Инициализация экстрактора структурированных данных.
        
        Args:
            extract_tables: Извлекать таблицы.
            extract_lists: Извлекать списки.
            extract_definitions: Извлекать определения.
            extract_code_blocks: Извлекать блоки кода.
            extract_quotes: Извлекать цитаты.
            extract_faq: Извлекать FAQ (вопросы и ответы).
            language: Язык контента ('ru' для русского, 'en' для английского).
        """
        self.extract_tables = extract_tables
        self.extract_lists = extract_lists
        self.extract_definitions = extract_definitions
        self.extract_code_blocks = extract_code_blocks
        self.extract_quotes = extract_quotes
        self.extract_faq = extract_faq
        self.language = language
    
    def extract_all_structured_data(self, html: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Извлекает все типы структурированных данных из HTML.
        
        Args:
            html: HTML-контент страницы.
            url: URL страницы.
            
        Returns:
            Dict[str, Any]: Извлеченные структурированные данные.
        """
        soup = BeautifulSoup(html, 'html.parser')
        structured_data = {
            'url': url,
            'extracted_at': datetime.now().isoformat(),
            'language': self.language,
            'types': []
        }
        
        # Извлекаем различные типы данных
        if self.extract_tables:
            tables = self._extract_tables(soup)
            if tables:
                structured_data['tables'] = tables
                structured_data['types'].append('tables')
        
        if self.extract_lists:
            lists = self._extract_lists(soup)
            if lists:
                structured_data['lists'] = lists
                structured_data['types'].append('lists')
        
        if self.extract_definitions:
            definitions = self._extract_definitions(soup)
            if definitions:
                structured_data['definitions'] = definitions
                structured_data['types'].append('definitions')
        
        if self.extract_code_blocks:
            code_blocks = self._extract_code_blocks(soup)
            if code_blocks:
                structured_data['code_blocks'] = code_blocks
                structured_data['types'].append('code_blocks')
        
        if self.extract_quotes:
            quotes = self._extract_quotes(soup)
            if quotes:
                structured_data['quotes'] = quotes
                structured_data['types'].append('quotes')
        
        if self.extract_faq:
            faq = self._extract_faq(soup)
            if faq:
                structured_data['faq'] = faq
                structured_data['types'].append('faq')
        
        # Дополнительно извлекаем JSON-LD
        json_ld = self._extract_json_ld(soup)
        if json_ld:
            structured_data['json_ld'] = json_ld
            structured_data['types'].append('json_ld')
        
        # Добавляем метаданные
        structured_data['metadata'] = {
            'total_items': sum(len(structured_data.get(t, [])) for t in structured_data.get('types', []) if t != 'json_ld')
        }
        
        return structured_data
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из HTML.
        
        Args:
            soup: BeautifulSoup объект.
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных таблиц.
        """
        tables = []
        table_elements = soup.find_all('table')
        
        for table_idx, table in enumerate(table_elements):
            # Проверяем, не является ли таблица частью другой структуры
            if table.parent and table.parent.name in ['td', 'th']:
                continue
            
            caption = table.find('caption')
            caption_text = caption.get_text().strip() if caption else ''
            
            # Определяем, есть ли заголовки
            thead = table.find('thead')
            has_header = False
            header_cells = []
            
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    header_cells = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                    if header_cells:
                        has_header = True
            else:
                # Проверяем первую строку - если в ней есть th, считаем её заголовком
                first_row = table.find('tr')
                if first_row:
                    ths = first_row.find_all('th')
                    if ths:
                        header_cells = [th.get_text().strip() for th in ths]
                        has_header = True
            
            # Извлекаем данные таблицы
            rows = []
            for row_idx, row in enumerate(table.find_all('tr')):
                # Пропускаем строку, если она является заголовком
                if has_header and row_idx == 0 and not thead:
                    continue
                
                cells = []
                for cell in row.find_all(['td', 'th']):
                    # Извлекаем содержимое ячейки
                    cell_content = cell.get_text().strip()
                    cells.append(cell_content)
                
                if cells:  # Добавляем строку только если она не пустая
                    rows.append(cells)
            
            # Создаем структуру таблицы
            table_data = {
                'id': f'table_{table_idx + 1}',
                'has_header': has_header,
                'rows_count': len(rows),
                'columns_count': len(header_cells) if has_header else (len(rows[0]) if rows else 0)
            }
            
            if caption_text:
                table_data['caption'] = caption_text
            
            if has_header:
                table_data['headers'] = header_cells
            
            table_data['rows'] = rows
            
            tables.append(table_data)
        
        return tables
    
    def _extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает списки из HTML.
        
        Args:
            soup: BeautifulSoup объект.
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных списков.
        """
        lists = []
        list_elements = soup.find_all(['ul', 'ol', 'dl'])
        
        for list_idx, list_elem in enumerate(list_elements):
            # Пропускаем вложенные списки
            if list_elem.parent and list_elem.parent.name in ['li', 'dd']:
                continue
            
            # Определяем тип списка
            list_type = list_elem.name
            
            # Извлекаем элементы списка
            items = []
            
            if list_type in ['ul', 'ol']:
                for item in list_elem.find_all('li', recursive=False):
                    # Извлекаем текст элемента
                    item_text = item.get_text().strip()
                    items.append(item_text)
            
            elif list_type == 'dl':
                for dt, dd in zip(list_elem.find_all('dt'), list_elem.find_all('dd')):
                    # Извлекаем текст термина и определения
                    term = dt.get_text().strip()
                    definition = dd.get_text().strip()
                    items.append({'term': term, 'definition': definition})
            
            # Создаем структуру списка
            list_data = {
                'id': f'list_{list_idx + 1}',
                'type': list_type,
                'items_count': len(items),
                'items': items
            }
            
            lists.append(list_data)
        
        return lists
    
    def _extract_definitions(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает определения из HTML.
        
        Args:
            soup: BeautifulSoup объект.
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных определений.
        """
        definitions = []
        
        # Ищем списки определений (dl/dt/dd)
        dl_elements = soup.find_all('dl')
        for dl_idx, dl in enumerate(dl_elements):
            dl_items = []
            for dt, dd in zip(dl.find_all('dt'), dl.find_all('dd')):
                dl_items.append({
                    'term': dt.get_text().strip(),
                    'definition': dd.get_text().strip()
                })
            
            if dl_items:
                definitions.append({
                    'id': f'def_dl_{dl_idx + 1}',
                    'type': 'definition_list',
                    'items': dl_items
                })
        
        return definitions
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает блоки кода из HTML.
        
        Args:
            soup: BeautifulSoup объект.
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных блоков кода.
        """
        code_blocks = []
        
        # Ищем стандартные блоки кода
        pre_elements = soup.find_all('pre')
        for pre_idx, pre in enumerate(pre_elements):
            code = pre.find('code')
            if code:
                code_content = code.get_text().strip()
            else:
                code_content = pre.get_text().strip()
            
            if code_content:
                code_blocks.append({
                    'id': f'code_{pre_idx + 1}',
                    'type': 'code_block',
                    'language': 'unknown',  # В реальном коде определение языка
                    'content': code_content
                })
        
        return code_blocks
    
    def _extract_quotes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает цитаты из HTML.
        
        Args:
            soup: BeautifulSoup объект.
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных цитат.
        """
        quotes = []
        
        # Ищем блоки цитат
        blockquote_elements = soup.find_all('blockquote')
        for quote_idx, quote in enumerate(blockquote_elements):
            quote_content = quote.get_text().strip()
            cite = quote.get('cite')
            
            if quote_content:
                quote_data = {
                    'id': f'quote_{quote_idx + 1}',
                    'type': 'blockquote',
                    'content': quote_content
                }
                
                if cite:
                    quote_data['source'] = cite
                
                quotes.append(quote_data)
        
        return quotes
    
    def _extract_faq(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает FAQ (вопросы и ответы) из HTML.
        
        Args:
            soup: BeautifulSoup объект.
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных FAQ.
        """
        faq_items = []
        
        # Ищем структуры вопрос-ответ
        # Вариант 1: dl с dt (вопрос) и dd (ответ)
        dl_elements = soup.find_all('dl', class_=['faq', 'qa', 'question-answer'])
        for dl_idx, dl in enumerate(dl_elements):
            for dt, dd in zip(dl.find_all('dt'), dl.find_all('dd')):
                faq_items.append({
                    'id': f'faq_dl_{dl_idx}_{len(faq_items) + 1}',
                    'question': dt.get_text().strip(),
                    'answer': dd.get_text().strip()
                })
        
        # Вариант 2: Структуры с классами или атрибутами
        faq_containers = soup.find_all(['div', 'section'], class_=['faq', 'qa', 'question-answer'])
        for container_idx, container in enumerate(faq_containers):
            # Ищем вопросы и ответы внутри
            questions = container.find_all(['h2', 'h3', 'h4', 'div', 'p'], class_=['question', 'q'])
            answers = container.find_all(['div', 'p'], class_=['answer', 'a'])
            
            for i, (q, a) in enumerate(zip(questions, answers)):
                faq_items.append({
                    'id': f'faq_container_{container_idx}_{i + 1}',
                    'question': q.get_text().strip(),
                    'answer': a.get_text().strip()
                })
        
        return faq_items
    
    def _extract_json_ld(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Извлекает структурированные данные JSON-LD из HTML.
        
        Args:
            soup: BeautifulSoup объект.
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных данных JSON-LD.
        """
        json_ld_data = []
        
        # Ищем скрипты JSON-LD
        json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    json_ld_data.extend(data)
                else:
                    json_ld_data.append(data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Не удалось разобрать JSON-LD: {e}")
        
        return json_ld_data
