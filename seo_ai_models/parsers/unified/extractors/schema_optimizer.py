# -*- coding: utf-8 -*-
"""
SchemaOptimizer - Модуль для оптимизации Schema.org разметки для лучшей цитируемости в LLM.
"""

import re
import json
import logging
import requests
import dateutil.parser
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class SchemaOptimizer:
    """Класс для оптимизации Schema.org разметки."""
    
    def __init__(self, schema_url: str = "https://schema.org/version/latest/schemaorg-current-https.jsonld"):
        """
        Инициализирует валидатор схем.
        
        Args:
            schema_url: URL схемы Schema.org
        """
        self.schema_url = schema_url
        self.schema_types = []
        self.schema_properties = {}
        self.logger = logger
        
        # Загружаем схему
        self.load_schema()
    
    def load_schema(self):
        """Загружает и обрабатывает схему Schema.org."""
        try:
            response = requests.get(self.schema_url, timeout=10)
            if response.status_code == 200:
                schema_data = response.json()
                
                # Извлекаем типы и свойства
                if '@graph' in schema_data:
                    for item in schema_data['@graph']:
                        if item.get('@type') == 'rdfs:Class':
                            self.schema_types.append(item.get('@id'))
                        elif item.get('@type') == 'rdf:Property':
                            property_id = item.get('@id')
                            domain = item.get('schema:domainIncludes', [])
                            
                            if not isinstance(domain, list):
                                domain = [domain]
                            
                            for d in domain:
                                if isinstance(d, dict) and '@id' in d:
                                    domain_id = d['@id']
                                    if domain_id not in self.schema_properties:
                                        self.schema_properties[domain_id] = []
                                    self.schema_properties[domain_id].append(property_id)
                
                self.logger.info(f"Схема Schema.org успешно загружена. Найдено {len(self.schema_types)} типов и {len(self.schema_properties)} доменов свойств.")
            else:
                self.logger.warning(f"Не удалось загрузить схему Schema.org. Код ответа: {response.status_code}")
                self._set_base_types()
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке схемы Schema.org: {str(e)}")
            self._set_base_types()
    def _set_base_types(self):
        """Устанавливает базовые типы Schema.org при неудачной загрузке схемы."""
        base_types = [
            "Thing", "Action", "CreativeWork", "Event", "Organization", 
            "Person", "Place", "Product", "WebPage", "WebSite"
        ]
        self.schema_types = base_types
        
        # Базовые свойства для наиболее распространенных типов
        self.schema_properties = {
            "Article": ["headline", "author", "datePublished", "dateModified", "publisher", "description", "image"],
            "NewsArticle": ["headline", "author", "datePublished", "dateModified", "publisher", "description", "image"],
            "BlogPosting": ["headline", "author", "datePublished", "dateModified", "publisher", "description", "image"],
            "Product": ["name", "description", "brand", "offers", "sku", "image", "review", "aggregateRating"],
            "Organization": ["name", "description", "logo", "url", "address", "telephone", "email"],
            "Person": ["name", "jobTitle", "telephone", "email", "address", "image"],
            "Event": ["name", "startDate", "endDate", "location", "organizer", "description"],
            "WebPage": ["name", "description", "url", "author", "datePublished", "dateModified"],
            "WebSite": ["name", "description", "url", "publisher"]
        }
        
        self.logger.info(f"Установлены базовые типы Schema.org: {', '.join(base_types)}")
    def improved_date_processing(self, prop, value, result):
        """
        Улучшенная обработка дат в схеме.
        
        Args:
            prop: Имя свойства
            value: Значение свойства
            result: Словарь результатов
            
        Returns:
            None
        """
        try:
            # Попытка разбора различных форматов даты
            parsed_date = dateutil.parser.parse(value)
            result[prop] = parsed_date.strftime('%Y-%m-%d')
        except Exception as e:
            self.logger.warning(f"Не удалось преобразовать дату '{value}' для свойства {prop}: {str(e)}")
            
            # Попытка извлечь дату с использованием регулярных выражений
            try:
                # YYYY-MM-DD или YYYY/MM/DD
                if re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', value):
                    parts = re.split(r'[/-]', value)
                    if len(parts) == 3 and len(parts[0]) == 4:  # год идет первым
                        year, month, day = parts
                        formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        result[prop] = formatted_date
                        return
                
                # DD.MM.YYYY или MM/DD/YYYY
                elif re.search(r'\d{1,2}[/.]\d{1,2}[/.]\d{4}', value):
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
        """
        Оптимизирует схему Schema.org для лучшей цитируемости в LLM.
        
        Args:
            schema_data: Данные схемы Schema.org
            
        Returns:
            Dict[str, Any]: Оптимизированная схема
        """
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
        """
        Извлекает схемы Schema.org из HTML-кода.
        
        Args:
            html_content: HTML-код страницы
            
        Returns:
            List[Dict[str, Any]]: Список схем Schema.org
        """
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
    def _optimize_common_properties(self, schema):
        """Оптимизирует общие свойства схемы."""
        # Оптимизируем даты
        date_properties = ['datePublished', 'dateModified', 'dateCreated', 'startDate', 'endDate', 'validFrom', 'validThrough']
        
        for prop in date_properties:
            if prop in schema and schema[prop]:
                try:
                    self.improved_date_processing(prop, schema[prop], schema)
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке даты {prop}: {str(e)}")
        
        # Оптимизируем URL
        url_properties = ['url', 'sameAs', 'contentUrl', 'thumbnailUrl']
        
        for prop in url_properties:
            if prop in schema and schema[prop]:
                # Убедимся, что URL абсолютный
                if isinstance(schema[prop], str) and not schema[prop].startswith(('http://', 'https://')):
                    schema[prop] = f"https://{schema[prop]}"
        
        return schema

    def _optimize_article(self, schema):
        """Оптимизирует схему статьи."""
        # Убедимся, что заголовок есть
        if 'headline' not in schema or not schema['headline']:
            if 'name' in schema:
                schema['headline'] = schema['name']
        
        # Убедимся, что описание есть
        if 'description' not in schema or not schema['description']:
            if 'articleBody' in schema:
                # Берем первые 200 символов текста как описание
                schema['description'] = schema['articleBody'][:200] + '...'
        
        # Оптимизируем автора
        if 'author' in schema:
            if isinstance(schema['author'], dict):
                if '@type' not in schema['author']:
                    schema['author']['@type'] = 'Person'
                
                if 'name' not in schema['author'] or not schema['author']['name']:
                    schema['author']['name'] = 'Unknown Author'
            elif isinstance(schema['author'], str):
                author_name = schema['author']
                schema['author'] = {
                    '@type': 'Person',
                    'name': author_name
                }
        
        return schema

    def _optimize_product(self, schema):
        """Оптимизирует схему продукта."""
        # Убедимся, что цена указана правильно
        if 'offers' in schema and isinstance(schema['offers'], dict):
            if 'price' in schema['offers'] and isinstance(schema['offers']['price'], str):
                try:
                    # Попытка преобразовать строку в число
                    schema['offers']['price'] = float(schema['offers']['price'].replace(',', '.'))
                except ValueError:
                    pass
            
            # Убедимся, что валюта указана
            if 'priceCurrency' not in schema['offers']:
                schema['offers']['priceCurrency'] = 'USD'
        
        return schema

    def _optimize_organization(self, schema):
        """Оптимизирует схему организации."""
        # Оптимизируем логотип
        if 'logo' in schema and isinstance(schema['logo'], str):
            logo_url = schema['logo']
            schema['logo'] = {
                '@type': 'ImageObject',
                'url': logo_url
            }
        
        return schema

    def _optimize_person(self, schema):
        """Оптимизирует схему человека."""
        # Оптимизируем имя
        if 'name' in schema and isinstance(schema['name'], str):
            # Если имя содержит только инициалы, развернем их
            name = schema['name']
            if re.match(r'^[A-ZА-Я]\.\s*[A-ZА-Я]\.\s*[A-ZА-Яa-zа-я]+$', name):
                schema['name'] = f"{name.split('.')[-1].strip()} {'.'.join(name.split('.')[:2])}."
        
        return schema

    def _optimize_event(self, schema):
        """Оптимизирует схему события."""
        # Оптимизируем местоположение
        if 'location' in schema and isinstance(schema['location'], str):
            location_name = schema['location']
            schema['location'] = {
                '@type': 'Place',
                'name': location_name
            }
        
        return schema

    def _optimize_webpage(self, schema):
        """Оптимизирует схему веб-страницы."""
        # Убедимся, что указан язык
        if 'inLanguage' not in schema:
            schema['inLanguage'] = 'en'
        
        return schema

