
"""
MetadataEnhancer - компонент для улучшения метаданных с целью повышения 
цитируемости русскоязычного контента в LLM.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Union, Set
from urllib.parse import urljoin, urlparse
import hashlib
from datetime import datetime

from bs4 import BeautifulSoup
from seo_ai_models.parsers.unified.extractors.meta_extractor import MetaExtractor

logger = logging.getLogger(__name__)


class MetadataEnhancer:
    """
    Компонент для улучшения метаданных для повышения цитируемости русскоязычного контента в LLM.
    
    Анализирует и обогащает метаданные страницы для лучшего распознавания 
    и цитирования русскоязычного контента LLM моделями.
    """
    
    def __init__(self, 
                 improve_titles: bool = True,
                 enhance_descriptions: bool = True,
                 optimize_keywords: bool = True,
                 add_authorship: bool = True,
                 add_timestamps: bool = True,
                 add_citations: bool = True,
                 language: str = 'ru'):
        """
        Инициализация улучшителя метаданных.
        
        Args:
            improve_titles: Улучшать заголовки.
            enhance_descriptions: Улучшать описания.
            optimize_keywords: Оптимизировать ключевые слова.
            add_authorship: Добавлять информацию об авторстве.
            add_timestamps: Добавлять метки времени.
            add_citations: Добавлять информацию о цитировании.
            language: Язык контента ('ru' для русского, 'en' для английского).
        """
        self.improve_titles = improve_titles
        self.enhance_descriptions = enhance_descriptions
        self.optimize_keywords = optimize_keywords
        self.add_authorship = add_authorship
        self.add_timestamps = add_timestamps
        self.add_citations = add_citations
        self.language = language
        
        self.meta_extractor = MetaExtractor()
        
        # Общие термины на русском языке, которые могут быть добавлены в метаданные для LLM
        self.semantic_terms = {
            'статья': ['контент', 'источник', 'ссылка', 'информация', 'факты'],
            'исследование': ['исследование', 'научная работа', 'анализ', 'результаты', 'методология'],
            'инструкция': ['руководство', 'инструкция', 'пошаговое руководство', 'этапы', 'обучение'],
            'новости': ['репортаж', 'текущие события', 'последние', 'освещение'],
            'обзор': ['оценка', 'анализ', 'мнение', 'критика', 'рейтинг'],
            'мнение': ['точка зрения', 'взгляд', 'позиция', 'убеждение', 'аргумент'],
            'определение': ['значение', 'объяснение', 'описание', 'концепция', 'термин']
        }
    
    def enhance_metadata(self, html: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Улучшает метаданные страницы для лучшей цитируемости в LLM.
        
        Args:
            html: HTML-контент страницы.
            url: URL страницы.
            
        Returns:
            Dict[str, Any]: Улучшенные метаданные.
        """
        # Извлекаем базовые метаданные
        original_metadata = self.meta_extractor.extract_meta_information(html, url)
        
        # Создаем копию для улучшений
        enhanced_metadata = original_metadata.copy()
        enhanced_metadata['language'] = self.language
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Улучшаем заголовки
        if self.improve_titles:
            enhanced_metadata = self._enhance_titles(enhanced_metadata, soup)
        
        # Улучшаем описания
        if self.enhance_descriptions:
            enhanced_metadata = self._enhance_descriptions(enhanced_metadata, soup)
        
        # Оптимизируем ключевые слова
        if self.optimize_keywords:
            enhanced_metadata = self._optimize_keywords(enhanced_metadata, soup)
        
        # Добавляем информацию об авторстве
        if self.add_authorship:
            enhanced_metadata = self._enhance_authorship(enhanced_metadata, soup)
        
        # Добавляем метки времени
        if self.add_timestamps:
            enhanced_metadata = self._enhance_timestamps(enhanced_metadata, soup)
        
        # Добавляем информацию о цитировании
        if self.add_citations:
            enhanced_metadata = self._enhance_citations(enhanced_metadata, soup, url)
        
        # Добавляем дополнительные метаданные для LLM
        enhanced_metadata['llm_optimized'] = True
        enhanced_metadata['llm_enhancement_version'] = '1.0'
        enhanced_metadata['llm_optimized_for'] = 'ru' if self.language == 'ru' else 'en'
        
        return enhanced_metadata
    
    def _enhance_titles(self, metadata: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Улучшает заголовки для лучшей цитируемости.
        Адаптировано для русскоязычного контента.
        
        Args:
            metadata: Текущие метаданные.
            soup: BeautifulSoup объект.
            
        Returns:
            Dict[str, Any]: Обновленные метаданные.
        """
        title = metadata.get('title', '')
        
        # Если заголовка нет, пытаемся извлечь его из HTML
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.text.strip()
            else:
                h1 = soup.find('h1')
                if h1:
                    title = h1.text.strip()
        
        # Улучшаем заголовок, если он есть
        if title:
            # Если заголовок содержит разделитель, вероятно это "Название | Сайт"
            if ' | ' in title:
                parts = title.split(' | ')
                # Ставим основную часть первой
                title = parts[0].strip()
                
            # Если заголовок содержит разделитель "-", вероятно это "Название - Сайт"
            elif ' - ' in title:
                parts = title.split(' - ')
                # Ставим основную часть первой
                title = parts[0].strip()
            
            # Для русского языка делаем только первую букву заголовка заглавной
            if self.language == 'ru':
                if title:
                    title = title[0].upper() + title[1:]
        
        metadata['title'] = title
        metadata['enhanced_title'] = title
        
        return metadata
    
    def _enhance_descriptions(self, metadata: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Улучшает описания для лучшей цитируемости.
        Адаптировано для русскоязычного контента.
        
        Args:
            metadata: Текущие метаданные.
            soup: BeautifulSoup объект.
            
        Returns:
            Dict[str, Any]: Обновленные метаданные.
        """
        description = metadata.get('description', '')
        
        # Если описания нет, пытаемся извлечь его из HTML
        if not description:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc.get('content').strip()
            else:
                # Ищем первый абзац или блок текста
                first_p = soup.find('p')
                if first_p:
                    description = first_p.text.strip()
        
        # Улучшаем описание, если оно есть
        if description:
            # Удаляем лишние пробелы, переносы строк и другие проблемы форматирования
            description = re.sub(r'\s+', ' ', description).strip()
            
            # Капитализируем первую букву
            if description:
                description = description[0].upper() + description[1:]
        
        metadata['description'] = description
        metadata['enhanced_description'] = description
        
        return metadata
    
    def _optimize_keywords(self, metadata: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Оптимизирует ключевые слова для лучшей цитируемости.
        Адаптировано для русскоязычного контента.
        
        Args:
            metadata: Текущие метаданные.
            soup: BeautifulSoup объект.
            
        Returns:
            Dict[str, Any]: Обновленные метаданные.
        """
        keywords = metadata.get('keywords', [])
        
        # Если ключевых слов нет, пытаемся извлечь их из HTML
        if not keywords:
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords and meta_keywords.get('content'):
                keywords_text = meta_keywords.get('content').strip()
                # Разделяем по запятым
                keywords = [k.strip() for k in keywords_text.split(',')]
        
        # Обновляем метаданные
        if keywords:
            metadata['keywords'] = keywords
            metadata['enhanced_keywords'] = keywords
            
            # Добавляем строковое представление для совместимости
            metadata['keywords_string'] = ', '.join(keywords)
        
        return metadata
    
    def _enhance_authorship(self, metadata: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Добавляет информацию об авторстве для лучшей цитируемости.
        Адаптировано для русскоязычного контента.
        
        Args:
            metadata: Текущие метаданные.
            soup: BeautifulSoup объект.
            
        Returns:
            Dict[str, Any]: Обновленные метаданные.
        """
        author = metadata.get('author', '')
        
        # Если информации об авторе нет, пытаемся извлечь её из HTML
        if not author:
            # Проверяем мета-теги
            meta_author = soup.find('meta', attrs={'name': 'author'})
            if meta_author and meta_author.get('content'):
                author = meta_author.get('content').strip()
            else:
                # Ищем по популярным классам и атрибутам
                # Адаптация для русскоязычных сайтов
                author_selectors = [
                    '.author', '.byline', '[rel="author"]', '.article-author',
                    '.post-author', '.entry-author', '[itemprop="author"]',
                    '.автор', '.имя-автора', '.публикатор'
                ]
                for selector in author_selectors:
                    author_elem = soup.select_one(selector)
                    if author_elem:
                        author = author_elem.text.strip()
                        break
        
        if author:
            metadata['author'] = author
            metadata['enhanced_author'] = author
        
        return metadata
    
    def _enhance_timestamps(self, metadata: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Добавляет метки времени для лучшей цитируемости.
        Адаптировано для русскоязычного контента.
        
        Args:
            metadata: Текущие метаданные.
            soup: BeautifulSoup объект.
            
        Returns:
            Dict[str, Any]: Обновленные метаданные.
        """
        published_date = metadata.get('published_date', '')
        
        # Если даты публикации нет, используем текущую
        if not published_date:
            published_date = datetime.now().isoformat()
            
        metadata['published_date'] = published_date
        
        return metadata
    
    def _enhance_citations(self, metadata: Dict[str, Any], soup: BeautifulSoup, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Добавляет информацию о цитировании для LLM.
        Адаптировано для русскоязычного контента.
        
        Args:
            metadata: Текущие метаданные.
            soup: BeautifulSoup объект.
            url: URL страницы.
            
        Returns:
            Dict[str, Any]: Обновленные метаданные.
        """
        citation_info = {}
        
        # Базовая информация для цитирования
        title = metadata.get('title', '')
        author = metadata.get('author', '')
        published_date = metadata.get('published_date', '')
        
        if title:
            citation_info['title'] = title
        
        if author:
            citation_info['author'] = author
        
        if published_date:
            citation_info['published_date'] = published_date
        
        if url:
            citation_info['url'] = url
        
        # Добавляем стили цитирования для русскоязычного контента
        if self.language == 'ru':
            gost = ''
            if author:
                gost += f"{author}. "
            
            if title:
                gost += f"{title}. "
            
            if published_date:
                try:
                    year = re.search(r'\d{4}', published_date).group(0)
                    gost += f"— {year}. "
                except:
                self.logger.warning(f"Не удалось извлечь год из даты: {published_date}")
                # Проверка на другие форматы даты
                try:
                    # Проверка на формат DD.MM.YYYY или MM/DD/YYYY
                    date_match = re.search(r'\b\d{1,2}[/.]\d{1,2}[/.]\d{4}\b', published_date)
                    if date_match:
                        date_str = date_match.group(0)
                        year = date_str.split('/')[-1].split('.')[-1]
                        gost += f"— {year}. "
                    else:
                        # Если ничего не найдено, добавляем информацию о доступе
                        accessed_date = datetime.now().strftime("%d.%m.%Y")
                        gost += f"— (дата обращения: {accessed_date}). ""
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке даты: {str(e)}")
                    # Если все не удалось, указываем только дату обращения
                    accessed_date = datetime.now().strftime("%d.%m.%Y")
                    gost += f"— (дата обращения: {accessed_date}). "
            
            # URL уже добавлен в improved_date_extraction
            
            citation_info['citation_style'] = gost.strip()
        
        metadata['citation_info'] = citation_info
        
        return metadata
    
    def apply_enhanced_metadata(self, html: str, metadata: Dict[str, Any]) -> str:
        """
        Применяет улучшенные метаданные к HTML-странице.
        
        Args:
            html: Исходный HTML.
            metadata: Улучшенные метаданные.
            
        Returns:
            str: HTML с улучшенными метаданными.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Проверяем наличие head
        if not soup.head:
            soup.html.insert(0, soup.new_tag('head'))
        
        # Обновляем базовые мета-теги
        meta_updates = {
            'title': metadata.get('title', ''),
            'description': metadata.get('description', ''),
            'keywords': metadata.get('keywords_string', ''),
            'author': metadata.get('author', ''),
            'language': metadata.get('language', 'ru')
        }
        
        for name, content in meta_updates.items():
            if not content:
                continue
                
            meta = soup.find('meta', attrs={'name': name})
            if meta:
                meta['content'] = content
            else:
                new_meta = soup.new_tag('meta', attrs={'name': name, 'content': content})
                soup.head.append(new_meta)
        
        # Обновляем тег title
        if 'title' in meta_updates and meta_updates['title']:
            title_tag = soup.find('title')
            if title_tag:
                title_tag.string = meta_updates['title']
            else:
                new_title = soup.new_tag('title')
                new_title.string = meta_updates['title']
                soup.head.append(new_title)
        
        return str(soup)

def improved_date_extraction(self, published_date, gost, url):
    if published_date:
        try:
            # Попытка найти год в формате 4 цифр
            year = re.search(r'\d{4}', published_date).group(0)
            gost += f"— {year}. "
        except AttributeError:
            # Если не удалось найти 4 цифры, пробуем другие форматы
            self.logger.warning(f"Не удалось извлечь год из даты: {published_date}")
            
            # Проверка на наличие 2 цифр, которые могут быть годом (например, '22 для 2022)
            year_match = re.search(r'\b\d{2}\b', published_date)
            if year_match:
                year_short = year_match.group(0)
                current_century = datetime.now().year // 100
                year = f"{current_century}{year_short}"
                self.logger.info(f"Извлечен короткий год: {year_short}, преобразован в: {year}")
                gost += f"— {year}. "
            
            # Проверка на текстовое представление даты
            months_ru = ['январ', 'феврал', 'март', 'апрел', 'ма[йя]', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']
            months_en = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
            
            month_pattern_ru = '|'.join(months_ru)
            month_pattern_en = '|'.join(months_en)
            
            date_match_ru = re.search(fr'({month_pattern_ru}).*?(\d{{4}})', published_date, re.IGNORECASE)
            date_match_en = re.search(fr'({month_pattern_en}).*?(\d{{4}})', published_date, re.IGNORECASE)
            
            if date_match_ru:
                year = date_match_ru.group(2)
                self.logger.info(f"Извлечен год из русской текстовой даты: {year}")
                gost += f"— {year}. "
            elif date_match_en:
                year = date_match_en.group(2)
                self.logger.info(f"Извлечен год из английской текстовой даты: {year}")
                gost += f"— {year}. "
            else:
                # Если ничего не найдено, добавляем информацию о доступе
                accessed_date = datetime.now().strftime("%d.%m.%Y")
                gost += f"— (дата обращения: {accessed_date}). ""
    
    if url:
        gost += f"URL: {url}"
    
    return gost.strip()

