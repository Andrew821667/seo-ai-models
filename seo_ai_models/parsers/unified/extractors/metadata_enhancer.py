# -*- coding: utf-8 -*-
"""
MetadataEnhancer - Модуль для улучшения метаданных страницы для лучшей цитируемости в LLM.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union
from bs4 import BeautifulSoup
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataEnhancer:
    """Класс для улучшения метаданных страницы для лучшей цитируемости в LLM."""
    
    def __init__(self, 
                 improve_titles: bool = True,
                 enhance_descriptions: bool = True,
                 optimize_keywords: bool = True):
        """
        Инициализирует модуль MetadataEnhancer.
        
        Args:
            improve_titles: Улучшать ли заголовки
            enhance_descriptions: Улучшать ли описания
            optimize_keywords: Оптимизировать ли ключевые слова
        """
        self.improve_titles = improve_titles
        self.enhance_descriptions = enhance_descriptions
        self.optimize_keywords = optimize_keywords
        self.logger = logger
    
    def _create_citation_gost_style(self, author: str, title: str, published_date: str, url: str) -> str:
        """
        Создает цитирование в стиле ГОСТ.
        
        Args:
            author: Автор материала
            title: Название материала
            published_date: Дата публикации
            url: URL материала
            
        Returns:
            str: Цитирование в стиле ГОСТ
        """
        gost = ""
        
        if author:
            gost += f"{author} "
        
        if title:
            gost += f"{title} "
        
        if published_date:
            try:
                year = re.search(r'\d{4}', published_date).group(0)
                gost += f"— {year}. "
            except:
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
                else:
                    # Если ничего не найдено, добавляем информацию о доступе
                    accessed_date = datetime.now().strftime("%d.%m.%Y")
                    gost += f"— (дата обращения: {accessed_date}). "
        
        if url:
            gost += f"URL: {url}"
        
        return gost.strip()
    
    def enhance_metadata(self, html: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Улучшает метаданные страницы для лучшей цитируемости в LLM.
        
        Args:
            html: HTML-код страницы
            url: URL страницы (опционально)
            
        Returns:
            Dict[str, Any]: Улучшенные метаданные
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Извлекаем базовые метаданные
        metadata = {
            'title': self._extract_title(soup),
            'description': self._extract_description(soup),
            'keywords': self._extract_keywords(soup),
            'author': self._extract_author(soup),
            'published_date': self._extract_published_date(soup),
            'language': self._extract_language(soup),
            'canonical_url': self._extract_canonical_url(soup, url)
        }
        
        # Улучшаем метаданные
        if self.improve_titles:
            metadata = self._enhance_titles(metadata, soup)
        
        if self.enhance_descriptions:
            metadata = self._enhance_descriptions(metadata, soup)
        
        if self.optimize_keywords:
            metadata = self._optimize_keywords(metadata, soup)
        
        # Создаем информацию для цитирования
        citation_info = {
            'citation_style': 'gost',  # По умолчанию используем ГОСТ
            'citation_apa': self._create_citation_apa_style(
                metadata.get('author', ''),
                metadata.get('title', ''),
                metadata.get('published_date', ''),
                url or metadata.get('canonical_url', '')
            ),
            'citation_mla': self._create_citation_mla_style(
                metadata.get('author', ''),
                metadata.get('title', ''),
                metadata.get('published_date', ''),
                url or metadata.get('canonical_url', '')
            ),
            'citation_gost': self._create_citation_gost_style(
                metadata.get('author', ''),
                metadata.get('title', ''),
                metadata.get('published_date', ''),
                url or metadata.get('canonical_url', '')
            )
        }
        
        metadata['citation_info'] = citation_info
        
        return metadata
