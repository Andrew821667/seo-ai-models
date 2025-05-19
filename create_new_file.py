import os

# Создаем правильный файл metadata_enhancer.py
metadata_content = """# -*- coding: utf-8 -*-
\"\"\"
MetadataEnhancer - Модуль для улучшения метаданных страницы для лучшей цитируемости в LLM.
\"\"\"

import re
import logging
from typing import Dict, List, Optional, Any, Union
from bs4 import BeautifulSoup
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataEnhancer:
    \"\"\"Класс для улучшения метаданных страницы для лучшей цитируемости в LLM.\"\"\"
    
    def __init__(self, 
                 improve_titles: bool = True,
                 enhance_descriptions: bool = True,
                 optimize_keywords: bool = True):
        \"\"\"
        Инициализирует модуль MetadataEnhancer.
        
        Args:
            improve_titles: Улучшать ли заголовки
            enhance_descriptions: Улучшать ли описания
            optimize_keywords: Оптимизировать ли ключевые слова
        \"\"\"
        self.improve_titles = improve_titles
        self.enhance_descriptions = enhance_descriptions
        self.optimize_keywords = optimize_keywords
        self.logger = logger
    
    def _create_citation_gost_style(self, author: str, title: str, published_date: str, url: str) -> str:
        \"\"\"
        Создает цитирование в стиле ГОСТ.
        
        Args:
            author: Автор материала
            title: Название материала
            published_date: Дата публикации
            url: URL материала
            
        Returns:
            str: Цитирование в стиле ГОСТ
        \"\"\"
        gost = ""
        
        if author:
            gost += f"{author} "
        
        if title:
            gost += f"{title} "
        
        if published_date:
            try:
                year = re.search(r'\\d{4}', published_date).group(0)
                gost += f"— {year}. "
            except:
                # Если не удалось найти 4 цифры, пробуем другие форматы
                self.logger.warning(f"Не удалось извлечь год из даты: {published_date}")
                
                # Проверка на наличие 2 цифр, которые могут быть годом (например, '22 для 2022)
                year_match = re.search(r'\\b\\d{2}\\b', published_date)
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
        \"\"\"
        Улучшает метаданные страницы для лучшей цитируемости в LLM.
        
        Args:
            html: HTML-код страницы
            url: URL страницы (опционально)
            
        Returns:
            Dict[str, Any]: Улучшенные метаданные
        \"\"\"
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
"""

# Создаем правильный файл schema_optimizer.py
schema_content = """# -*- coding: utf-8 -*-
\"\"\"
SchemaOptimizer - Модуль для оптимизации Schema.org разметки для лучшей цитируемости в LLM.
\"\"\"

import re
import json
import logging
import requests
import dateutil.parser
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class SchemaOptimizer:
    \"\"\"Класс для оптимизации Schema.org разметки.\"\"\"
    
    def __init__(self, schema_url: str = "https://schema.org/version/latest/schemaorg-current-https.jsonld"):
        \"\"\"
        Инициализирует валидатор схем.
        
        Args:
            schema_url: URL схемы Schema.org
        \"\"\"
        self.schema_url = schema_url
        self.schema_types = []
        self.schema_properties = {}
        self.logger = logger
        
        # Загружаем схему
        self.load_schema()
    
    def improved_date_processing(self, prop, value, result):
        \"\"\"
        Улучшенная обработка дат в схеме.
        
        Args:
            prop: Имя свойства
            value: Значение свойства
            result: Словарь результатов
            
        Returns:
            None
        \"\"\"
        try:
            # Попытка разбора различных форматов даты
            parsed_date = dateutil.parser.parse(value)
            result[prop] = parsed_date.strftime('%Y-%m-%d')
        except Exception as e:
            self.logger.warning(f"Не удалось преобразовать дату '{value}' для свойства {prop}: {str(e)}")
            
            # Попытка извлечь дату с использованием регулярных выражений
            try:
                # YYYY-MM-DD или YYYY/MM/DD
                if re.search(r'\\d{4}[/-]\\d{1,2}[/-]\\d{1,2}', value):
                    parts = re.split(r'[/-]', value)
                    if len(parts) == 3 and len(parts[0]) == 4:  # год идет первым
                        year, month, day = parts
                        formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        result[prop] = formatted_date
                        return
                
                # DD.MM.YYYY или MM/DD/YYYY
                elif re.search(r'\\d{1,2}[/.]\\d{1,2}[/.]\\d{4}', value):
                    parts = re.split(r'[/.]', value)
                    if len(parts) == 3 and len(parts[2]) == 4:  # год идет последним
                        if '.' in value:  # вероятно DD.MM.YYYY
                            day, month, year = parts
                        else:  # вероятно MM/DD/YYYY
                            month, day, year = parts
                        formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        result[prop] = formatted_date
                        return
            
                # Если не удалось разобрать дату, оставляем оригинальное значение
                result[prop] = value
            except Exception as e:
                self.logger.error(f"Ошибка при обработке даты: {str(e)}")
                result[prop] = value
    
    def optimize_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Оптимизирует схему Schema.org для лучшей цитируемости в LLM.
        
        Args:
            schema_data: Данные схемы Schema.org
            
        Returns:
            Dict[str, Any]: Оптимизированная схема
        \"\"\"
        if not schema_data:
            return {}
        
        result = schema_data.copy()
        
        # Валидируем и оптимизируем общие свойства
        self._optimize_common_properties(result)
        
        # Обрабатываем разные типы схем
        schema_type = result.get('@type')
        if not schema_type:
            return result
        
        if isinstance(schema_type, list):
            schema_type = schema_type[0]
        
        # Обрабатываем различные типы схем
        if schema_type == 'Article' or schema_type == 'NewsArticle' or schema_type == 'BlogPosting':
            self._optimize_article(result)
        elif schema_type == 'Product':
            self._optimize_product(result)
        elif schema_type == 'Organization':
            self._optimize_organization(result)
        elif schema_type == 'Person':
            self._optimize_person(result)
        elif schema_type == 'Event':
            self._optimize_event(result)
        elif schema_type == 'WebPage' or schema_type == 'WebSite':
            self._optimize_webpage(result)
        
        return result
    
    def extract_schema_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        \"\"\"
        Извлекает схемы Schema.org из HTML-кода.
        
        Args:
            html_content: HTML-код страницы
            
        Returns:
            List[Dict[str, Any]]: Список схем Schema.org
        \"\"\"
        schemas = []
        
        # Извлекаем JSON-LD
        jsonld_pattern = re.compile(r'''<script[^>]*type=["']application/ld\+json["'][^>]*>(.*?)</script>''', re.DOTALL)
        matches = jsonld_pattern.findall(html_content)
        
        for match in matches:
            try:
                schema = json.loads(match)
                schemas.append(schema)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Ошибка при разборе JSON-LD: {str(e)}")
        
        # Дополнительно можно добавить извлечение микроданных и RDFa
        
        return schemas
"""

# Создаем новые файлы
with open("seo_ai_models/parsers/unified/extractors/metadata_enhancer.py", 'w', encoding='utf-8') as f:
    f.write(metadata_content)

with open("seo_ai_models/parsers/unified/extractors/schema_optimizer.py", 'w', encoding='utf-8') as f:
    f.write(schema_content)

print("Новые файлы успешно созданы")
