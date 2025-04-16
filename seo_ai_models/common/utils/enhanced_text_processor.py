
""" Улучшенный TextProcessor для более качественной обработки HTML-контента.
Расширяет возможности стандартного TextProcessor добавляя специфическую логику для работы с JavaScript-генерируемым контентом.
"""

from typing import Dict, List, Tuple, Set, Optional, Union
import re
import os
import string
from collections import Counter
from bs4 import BeautifulSoup
from seo_ai_models.common.utils.text_processing import TextProcessor

class EnhancedTextProcessor(TextProcessor):
    """
    Улучшенная версия TextProcessor с дополнительными возможностями для обработки
    HTML-контента и JavaScript-генерируемого текста.
    """

    def __init__(self, language=None):
        """Инициализация процессора текста."""
        super().__init__()
        self.language = language
    
    def process_html_content(self, html_content: str) -> Dict[str, any]:
        """
        Глубокий анализ HTML-контента с извлечением различных метрик.
        
        Args:
            html_content: HTML-контент для анализа
            
        Returns:
            Dict[str, any]: Извлеченные метрики и данные
        """
        if not html_content:
            return {}
            
        # Используем BeautifulSoup для более надежного парсинга
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Базовые метрики
        text_content = soup.get_text()
        
        # Анализ языка
        self.detect_language(text_content)
        
        # Анализируем структуру
        structure = self.analyze_html_structure(soup)
        
        # Извлекаем ключевые слова
        keywords = self.extract_keywords(text_content)
        
        # Анализируем читабельность
        readability = self.calculate_readability(text_content)
        
        return {
            "language": self.language,
            "structure": structure,
            "keywords": keywords,
            "readability": readability,
            "word_count": len(text_content.split()),
            "character_count": len(text_content)
        }
        
    def analyze_html_structure(self, soup: BeautifulSoup) -> Dict[str, any]:
        """
        Анализирует структуру HTML-документа.
        
        Args:
            soup: объект BeautifulSoup
            
        Returns:
            Dict: структурные метрики документа
        """
        headers = {f"h{i}": len(soup.find_all(f'h{i}')) for i in range(1, 7)}
        paragraphs = len(soup.find_all('p'))
        links = len(soup.find_all('a'))
        images = len(soup.find_all('img'))
        lists = len(soup.find_all(['ul', 'ol']))
        tables = len(soup.find_all('table'))
        
        return {
            "headers": headers,
            "total_headers": sum(headers.values()),
            "paragraphs": paragraphs,
            "links": links,
            "images": images,
            "lists": lists,
            "tables": tables
        }
    
    def detect_language(self, text: str) -> str:
        """
        Определяет язык текста.
        
        Args:
            text: анализируемый текст
            
        Returns:
            str: код языка
        """
        # Заглушка - в реальном приложении здесь использовалась бы
        # библиотека определения языка
        self.language = self.language or "en"
        return self.language
