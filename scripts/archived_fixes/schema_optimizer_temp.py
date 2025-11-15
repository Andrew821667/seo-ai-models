
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SchemaOptimizer - модуль для оптимизации схем разметки данных (Schema.org, JSON-LD, Microdata)
с учетом требований LLM-систем.

Модуль предоставляет функциональность для анализа, валидации и оптимизации
структурированных данных на веб-страницах.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class SchemaValidator:
    """Класс для валидации схем разметки данных."""
    
    def __init__(self, schema_url: str = "https://schema.org/version/latest/schemaorg-current-https.jsonld"):
        """
        Инициализирует валидатор схем.
        
        Args:
            schema_url: URL файла схемы Schema.org
        """
        self.schema_url = schema_url
        self.schema_types = {}
        self.schema_properties = {}
        self.loaded = False
        self.relationships = {}  # Структура для отслеживания отношений между типами
        
        # Загружаем схему при инициализации
        self.load_schema()
    
    def load_schema(self):
        """Загружает и обрабатывает схему Schema.org."""
        try:
            response = requests.get(self.schema_url, timeout=10)
            response.raise_for_status()
            
            schema_data = response.json()
            
            # Обрабатываем схему
            if "@graph" in schema_data:
                for item in schema_data["@graph"]:
                    if item.get("@type") == "rdfs:Class":
                        type_id = item["@id"]
                        self.schema_types[type_id] = {
                            "label": item.get("rdfs:label", ""),
                            "comment": item.get("rdfs:comment", ""),
                            "subClassOf": item.get("rdfs:subClassOf", [])
                        }
                        
                        # Добавляем отношения между типами
                        if "rdfs:subClassOf" in item:
                            parent_types = item["rdfs:subClassOf"]
                            if isinstance(parent_types, list):
                                for parent in parent_types:
                                    parent_id = parent.get("@id") if isinstance(parent, dict) else parent
                                    if parent_id not in self.relationships:
                                        self.relationships[parent_id] = []
                                    if type_id not in self.relationships[parent_id]:
                                        self.relationships[parent_id].append(type_id)
                            else:
                                parent_id = parent_types.get("@id") if isinstance(parent_types, dict) else parent_types
                                if parent_id not in self.relationships:
                                    self.relationships[parent_id] = []
                                if type_id not in self.relationships[parent_id]:
                                    self.relationships[parent_id].append(type_id)
                    
                    elif item.get("@type") == "rdf:Property":
                        self.schema_properties[item["@id"]] = {
                            "label": item.get("rdfs:label", ""),
                            "comment": item.get("rdfs:comment", ""),
                            "domain": item.get("schema:domainIncludes", []),
                            "range": item.get("schema:rangeIncludes", [])
                        }
            
            self.loaded = True
            logger.info(f"Schema.org schema loaded: {len(self.schema_types)} types, {len(self.schema_properties)} properties")
        except Exception as e:
            logger.error(f"Error loading Schema.org schema: {str(e)}")
            # Устанавливаем минимальные базовые типы
            self._set_base_types()
    
    def _set_base_types(self):
        """Устанавливает базовые типы Schema.org при неудачной загрузке схемы."""
        base_types = [
            "Thing", "Action", "CreativeWork", "Event", "Organization", 
            "Person", "Place", "Product", "BreadcrumbList", "Review",
            "WebSite", "WebPage", "Article", "BlogPosting", "NewsArticle",
            "ItemList", "ListItem", "FAQPage", "Question", "Answer",
            "HowTo", "Recipe", "LocalBusiness", "Offer", "AggregateRating"
        ]
        
        # Устанавливаем иерархию типов
        hierarchy = {
            "Thing": [],
            "CreativeWork": ["Thing"],
            "Article": ["Thing", "CreativeWork"],
            "BlogPosting": ["Thing", "CreativeWork", "Article"],
            "NewsArticle": ["Thing", "CreativeWork", "Article"],
            "WebPage": ["Thing", "CreativeWork"],
            "WebSite": ["Thing", "CreativeWork"],
            "FAQPage": ["Thing", "CreativeWork", "WebPage"],
            "HowTo": ["Thing", "CreativeWork"],
            "Recipe": ["Thing", "CreativeWork", "HowTo"],
            "Person": ["Thing"],
            "Organization": ["Thing"],
            "LocalBusiness": ["Thing", "Organization"],
            "Place": ["Thing"],
            "Product": ["Thing"],
            "Event": ["Thing"],
            "Action": ["Thing"],
            "Review": ["Thing", "CreativeWork"],
            "ItemList": ["Thing", "CreativeWork"],
            "ListItem": ["Thing"],
            "Question": ["Thing", "CreativeWork"],
            "Answer": ["Thing", "CreativeWork"],
            "Offer": ["Thing"],
            "AggregateRating": ["Thing"]
        }
        
        # Создаем типы и их отношения
        for type_name in base_types:
            self.schema_types[f"schema:{type_name}"] = {
                "label": type_name,
                "comment": f"Schema.org {type_name} type",
                "subClassOf": [f"schema:{parent}" for parent in hierarchy.get(type_name, [])]
            }
        
        # Устанавливаем отношения между типами
        for type_name, parents in hierarchy.items():
            type_id = f"schema:{type_name}"
            for parent in parents:
                parent_id = f"schema:{parent}"
                if parent_id not in self.relationships:
                    self.relationships[parent_id] = []
                if type_id not in self.relationships[parent_id]:
                    self.relationships[parent_id].append(type_id)
        
        # Основные свойства для разных типов
        base_properties = {
            "Thing": ["name", "description", "url", "image", "identifier"],
            "CreativeWork": ["author", "datePublished", "dateModified", "publisher", "headline", "keywords", "text"],
            "Article": ["articleBody", "wordCount", "articleSection"],
            "Person": ["givenName", "familyName", "email", "telephone", "birthDate", "jobTitle"],
            "Organization": ["logo", "address", "telephone", "email", "sameAs"],
            "LocalBusiness": ["openingHours", "priceRange", "telephone", "address"],
            "Product": ["brand", "sku", "offers", "review", "aggregateRating"],
            "Offer": ["price", "priceCurrency", "availability", "itemCondition"],
            "Event": ["startDate", "endDate", "location", "organizer", "performer"],
            "Recipe": ["recipeIngredient", "recipeInstructions", "cookTime", "prepTime", "totalTime"]
        }
        
        # Добавляем свойства для типов
        for type_name, props in base_properties.items():
            for prop in props:
                self.schema_properties[f"schema:{prop}"] = {
                    "label": prop,
                    "comment": f"Schema.org {prop} property",
                    "domain": [f"schema:{type_name}"],
                    "range": []
                }
        
        self.loaded = True
        logger.info(f"Set base Schema.org types and properties: {len(self.schema_types)} types, {len(self.schema_properties)} properties")
    
    def validate_type(self, schema_type: str) -> bool:
        """
        Проверяет, является ли тип допустимым типом Schema.org.
        
        Args:
            schema_type: Тип Schema.org
            
        Returns:
            bool: True, если тип допустимый, иначе False
        """
        if not self.loaded:
            return True  # Если схема не загружена, пропускаем валидацию
        
        # Удаляем префикс http://schema.org/ если он есть
        if schema_type.startswith("http://schema.org/") or schema_type.startswith("https://schema.org/"):
            schema_type = "schema:" + schema_type.split("/")[-1]
        
        # Нормализуем тип
        if not schema_type.startswith("schema:") and not schema_type.startswith("http"):
            schema_type = f"schema:{schema_type}"
        
        return schema_type in self.schema_types
    
    def validate_property(self, schema_property: str, schema_type: Optional[str] = None) -> bool:
        """
        Проверяет, является ли свойство допустимым для указанного типа Schema.org.
        
        Args:
            schema_property: Свойство Schema.org
            schema_type: Тип Schema.org (опционально)
            
        Returns:
            bool: True, если свойство допустимое, иначе False
        """
        if not self.loaded:
            return True  # Если схема не загружена, пропускаем валидацию
        
        # Удаляем префикс http://schema.org/ если он есть
        if schema_property.startswith("http://schema.org/") or schema_property.startswith("https://schema.org/"):
            schema_property = "schema:" + schema_property.split("/")[-1]
        
        # Нормализуем свойство
        if not schema_property.startswith("schema:") and not schema_property.startswith("http"):
            schema_property = f"schema:{schema_property}"
        
        # Проверяем, существует ли свойство
        if schema_property not in self.schema_properties:
            return False
        
        # Если тип не указан, считаем свойство валидным
        if not schema_type:
            return True
        
        # Удаляем префикс http://schema.org/ если он есть
        if schema_type.startswith("http://schema.org/") or schema_type.startswith("https://schema.org/"):
            schema_type = "schema:" + schema_type.split("/")[-1]
        
        # Нормализуем тип
        if not schema_type.startswith("schema:") and not schema_type.startswith("http"):
            schema_type = f"schema:{schema_type}"
        
        # Проверяем, является ли свойство допустимым для указанного типа
        property_info = self.schema_properties[schema_property]
        domains = property_info.get("domain", [])
        
        # Если список доменов пуст, считаем свойство валидным для любого типа
        if not domains:
            return True
        
        # Проверяем прямое соответствие
        if isinstance(domains, list):
            if schema_type in domains:
                return True
        elif isinstance(domains, dict):
            if schema_type == domains.get("@id"):
                return True
        else:
            if schema_type == domains:
                return True
        
        # Проверяем наследование
        for domain in domains:
            domain_id = domain.get("@id") if isinstance(domain, dict) else domain
            if self._is_subclass_of(schema_type, domain_id):
                return True
        
        return False
    
    def _is_subclass_of(self, type_id: str, parent_id: str) -> bool:
        """
        Проверяет, является ли тип подклассом родительского типа.
        
        Args:
            type_id: ID типа
            parent_id: ID родительского типа
            
        Returns:
            bool: True, если тип является подклассом родительского типа, иначе False
        """
        if type_id == parent_id:
            return True
        
        if type_id not in self.schema_types:
            return False
        
        type_info = self.schema_types[type_id]
        subclass_of = type_info.get("subClassOf", [])
        
        if isinstance(subclass_of, list):
            for parent in subclass_of:
                parent_type_id = parent.get("@id") if isinstance(parent, dict) else parent
                if parent_type_id == parent_id or self._is_subclass_of(parent_type_id, parent_id):
                    return True
        elif isinstance(subclass_of, dict):
            parent_type_id = subclass_of.get("@id")
            if parent_type_id == parent_id or self._is_subclass_of(parent_type_id, parent_id):
                return True
        else:
            if subclass_of == parent_id or self._is_subclass_of(subclass_of, parent_id):
                return True
        
        return False
    
    def get_all_subtypes(self, type_id: str) -> List[str]:
        """
        Получает список всех подтипов данного типа.
        
        Args:
            type_id: ID типа
            
        Returns:
            List[str]: Список ID подтипов
        """
        subtypes = []
        
        if type_id in self.relationships:
            subtypes.extend(self.relationships[type_id])
            
            for subtype in self.relationships[type_id]:
                subtypes.extend(self.get_all_subtypes(subtype))
        
        return subtypes


class SchemaOptimizer:
    """
    Класс для оптимизации схем разметки данных с учетом требований LLM-систем.
    
    Предоставляет функциональность для:
    1. Извлечения структурированных данных со страницы
    2. Валидации схем разметки
    3. Оптимизации схем для улучшения совместимости с LLM
    4. Генерации рекомендаций по улучшению разметки
    """
    
    def __init__(self, validator: Optional[SchemaValidator] = None):
        """
        Инициализирует оптимизатор схем.
        
        Args:
            validator: Валидатор схем (если None, создается новый)
        """
        self.validator = validator or SchemaValidator()
        
        # Список типов Schema.org, наиболее важных для LLM
        self.important_types = [
            "Article", "NewsArticle", "BlogPosting", "WebPage", "Product",
            "Review", "FAQPage", "Question", "Answer", "HowTo", "Recipe",
            "Event", "Organization", "Person", "LocalBusiness", "Course",
            "BreadcrumbList", "VideoObject", "AudioObject", "Movie", "Book"
        ]
        
        # Список свойств Schema.org, наиболее важных для LLM
        self.important_properties = {
            "Article": ["headline", "description", "articleBody", "author", "datePublished", "dateModified", "publisher", "image", "keywords", "wordCount"],
            "NewsArticle": ["headline", "description", "articleBody", "author", "datePublished", "dateModified", "publisher", "image", "keywords", "wordCount", "dateline"],
            "BlogPosting": ["headline", "description", "articleBody", "author", "datePublished", "dateModified", "publisher", "image", "keywords", "wordCount"],
            "WebPage": ["name", "description", "breadcrumb", "mainEntity", "datePublished", "dateModified", "author", "image", "keywords", "publisher"],
            "Product": ["name", "description", "brand", "offers", "aggregateRating", "review", "image", "sku", "mpn", "gtin", "category"],
            "Review": ["name", "reviewBody", "author", "datePublished", "itemReviewed", "reviewRating", "publisher"],
            "FAQPage": ["mainEntity", "name", "description"],
            "Question": ["name", "text", "acceptedAnswer", "suggestedAnswer", "author", "dateCreated"],
            "Answer": ["text", "author", "dateCreated", "upvoteCount"],
            "HowTo": ["name", "description", "step", "totalTime", "tool", "supply", "image", "prepTime", "performTime"],
            "Recipe": ["name", "description", "recipeIngredient", "recipeInstructions", "cookTime", "prepTime", "totalTime", "image", "recipeYield", "nutrition", "suitableForDiet"],
            "Event": ["name", "description", "startDate", "endDate", "location", "organizer", "performer", "image", "offers", "eventStatus", "eventAttendanceMode"],
            "Organization": ["name", "description", "logo", "address", "contactPoint", "url", "sameAs", "foundingDate", "founder", "memberOf", "numberOfEmployees"],
            "Person": ["name", "description", "image", "jobTitle", "worksFor", "alumniOf", "birthDate", "nationality", "address", "affiliation", "email", "telephone"],
            "LocalBusiness": ["name", "description", "address", "telephone", "openingHours", "priceRange", "image", "geo", "servesCuisine", "paymentAccepted", "currenciesAccepted"],
            "Course": ["name", "description", "provider", "hasCourseInstance", "courseCode", "image", "about", "timeRequired", "educationalCredentialAwarded"],
            "BreadcrumbList": ["itemListElement"],
            "VideoObject": ["name", "description", "thumbnailUrl", "uploadDate", "duration", "contentUrl", "embedUrl", "publisher", "creator", "transcript"],
            "AudioObject": ["name", "description", "contentUrl", "duration", "encodingFormat", "uploadDate", "transcript"],
            "Movie": ["name", "description", "director", "actor", "datePublished", "duration", "genre", "productionCompany", "image"],
            "Book": ["name", "description", "author", "publisher", "isbn", "datePublished", "numberOfPages", "genre", "bookEdition", "image"]
        }
        
        # Соответствие между типами Schema.org и LLM-тегами
        self.llm_tag_mapping = {
            "Article": ["ARTICLE", "CONTENT"],
            "NewsArticle": ["NEWS", "ARTICLE", "CONTENT"],
            "BlogPosting": ["BLOG", "ARTICLE", "CONTENT"],
            "WebPage": ["WEBPAGE", "CONTENT"],
            "Product": ["PRODUCT", "ITEM"],
            "Review": ["REVIEW", "OPINION"],
            "FAQPage": ["FAQ", "QUESTION", "ANSWER"],
            "Question": ["QUESTION"],
            "Answer": ["ANSWER"],
            "HowTo": ["HOWTO", "INSTRUCTIONS", "TUTORIAL"],
            "Recipe": ["RECIPE", "INSTRUCTIONS"],
            "Event": ["EVENT", "OCCASION"],
            "Organization": ["ORGANIZATION", "ENTITY"],
            "Person": ["PERSON", "ENTITY"],
            "LocalBusiness": ["BUSINESS", "ORGANIZATION", "ENTITY"],
            "Course": ["COURSE", "EDUCATIONAL"],
            "BreadcrumbList": ["NAVIGATION", "STRUCTURE"],
            "VideoObject": ["VIDEO", "MEDIA"],
            "AudioObject": ["AUDIO", "MEDIA"],
            "Movie": ["MOVIE", "CREATIVE_WORK"],
            "Book": ["BOOK", "CREATIVE_WORK"]
        }
        
        # Правила оптимизации для разных типов контента
        self.optimization_rules = {
            "Article": {
                "min_length": {
                    "headline": 10,
                    "description": 50,
                    "articleBody": 300
                },
                "required": ["headline", "description", "articleBody", "author", "datePublished"],
                "recommended": ["image", "keywords", "publisher", "dateModified"]
            },
            "Product": {
                "required": ["name", "description", "brand", "offers"],
                "recommended": ["image", "sku", "aggregateRating", "review"]
            },
            "Recipe": {
                "required": ["name", "description", "recipeIngredient", "recipeInstructions"],
                "recommended": ["image", "cookTime", "prepTime", "totalTime", "recipeYield"]
            },
            "FAQPage": {
                "required": ["mainEntity"],
                "recommended": ["name", "description"]
            }
        }
    
    def extract_jsonld(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Извлекает данные JSON-LD со страницы.
        
        Args:
            html_content: HTML-содержимое страницы
            
        Returns:
            List[Dict[str, Any]]: Список объектов JSON-LD
        """
        results = []
        
        # Используем регулярное выражение для поиска скриптов JSON-LD
        jsonld_pattern = re.compile(r'<script[^>]*type=["|']application/ld\+json["|'][^>]*>(.*?)</script>', re.DOTALL)
        matches = jsonld_pattern.findall(html_content)
        
        for match in matches:
            try:
                # Очищаем содержимое от комментариев и лишних пробелов
                cleaned = re.sub(r'/\*.*?\*/', '', match, flags=re.DOTALL)
                cleaned = re.sub(r'\s+', ' ', cleaned)
                cleaned = cleaned.strip()
                
                # Парсим JSON
                data = json.loads(cleaned)
                
                # Обрабатываем случай с @graph
                if isinstance(data, dict) and '@graph' in data:
                    for item in data['@graph']:
                        # Добавляем контекст к каждому элементу графа
                        if '@context' in data and '@context' not in item:
                            item_with_context = item.copy()
                            item_with_context['@context'] = data['@context']
                            results.append(item_with_context)
                        else:
                            results.append(item)
                else:
                    results.append(data)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON-LD: {str(e)}")
                logger.debug(f"Invalid JSON-LD content: {match[:100]}...")
        
        # Удаляем дубликаты (по содержимому)
        unique_results = []
        seen_jsons = set()
        
        for item in results:
            item_json = json.dumps(item, sort_keys=True)
            if item_json not in seen_jsons:
                seen_jsons.add(item_json)
                unique_results.append(item)
        
        return unique_results
    
    def extract_microdata(self, html_content: str, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Извлекает данные Microdata со страницы.
        
        Args:
            html_content: HTML-содержимое страницы
            base_url: Базовый URL для резолвинга относительных ссылок
            
        Returns:
            List[Dict[str, Any]]: Список объектов Microdata
        """
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Находим все элементы с itemscope
            items = soup.find_all(itemscope=True)
            
            for item in items:
                # Пропускаем вложенные элементы, которые уже будут обработаны при рекурсивном обходе
                if item.parent and item.parent.has_attr('itemscope'):
                    continue
                
                # Обрабатываем элемент
                item_data = self._extract_microdata_item(item, base_url)
                if item_data:
                    results.append(item_data)
        except Exception as e:
            logger.error(f"Error extracting Microdata: {str(e)}")
        
        return results
    
    def _extract_microdata_item(self, item, base_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Рекурсивно извлекает данные из элемента Microdata.
        
        Args:
            item: Элемент BeautifulSoup с атрибутом itemscope
            base_url: Базовый URL для резолвинга относительных ссылок
            
        Returns:
            Optional[Dict[str, Any]]: Данные элемента Microdata или None
        """
        if not item.has_attr('itemscope'):
            return None
        
        result = {}
        
        # Получаем тип элемента
        if item.has_attr('itemtype'):
            itemtype = item['itemtype']
            # Если это URL типа Schema.org, извлекаем только название типа
            if isinstance(itemtype, str) and ('schema.org' in itemtype or 'schema.org' in itemtype):
                type_name = itemtype.split('/')[-1]
                result['@type'] = type_name
            else:
                result['@type'] = itemtype
        
        # Получаем id элемента
        if item.has_attr('itemid'):
            result['@id'] = item['itemid']
        
        # Получаем свойства элемента
        props = {}
        
        # Обрабатываем свойства самого элемента
        if item.has_attr('itemprop'):
            for prop_name in item['itemprop'].split():
                props[prop_name] = self._get_item_value(item, base_url)
        
        # Обрабатываем все дочерние элементы с itemprop
        for prop_elem in item.find_all(attrs={'itemprop': True}, recursive=True):
            # Пропускаем элементы, которые находятся внутри других элементов с itemscope
            parent_with_scope = prop_elem.find_parent(itemscope=True)
            if parent_with_scope and parent_with_scope != item:
                continue
            
            for prop_name in prop_elem['itemprop'].split():
                # Если элемент имеет itemscope, обрабатываем его рекурсивно
                if prop_elem.has_attr('itemscope'):
                    prop_value = self._extract_microdata_item(prop_elem, base_url)
                else:
                    prop_value = self._get_item_value(prop_elem, base_url)
                
                # Добавляем свойство
                if prop_name in props:
                    if isinstance(props[prop_name], list):
                        props[prop_name].append(prop_value)
                    else:
                        props[prop_name] = [props[prop_name], prop_value]
                else:
                    props[prop_name] = prop_value
        
        # Объединяем свойства с результатом
        result.update(props)
        
        return result
    
    def _get_item_value(self, elem, base_url: Optional[str] = None) -> Any:
        """
        Получает значение элемента Microdata.
        
        Args:
            elem: Элемент BeautifulSoup
            base_url: Базовый URL для резолвинга относительных ссылок
            
        Returns:
            Any: Значение элемента
        """
        # Проверяем различные атрибуты в порядке приоритета
        if elem.has_attr('content'):
            return elem['content']
        elif elem.has_attr('datetime'):
            return elem['datetime']
        elif elem.name == 'meta':
            return elem.get('content', '')
        elif elem.name == 'a' or elem.name == 'link':
            href = elem.get('href', '')
            if base_url and href and not (href.startswith('http://') or href.startswith('https://') or href.startswith('//')):
                return urljoin(base_url, href)
            return href
        elif elem.name == 'img':
            src = elem.get('src', '')
            if base_url and src and not (src.startswith('http://') or src.startswith('https://') or src.startswith('//')):
                return urljoin(base_url, src)
            return src
        elif elem.name == 'time':
            return elem.get('datetime', elem.text.strip())
        elif elem.name == 'data':
            return elem.get('value', elem.text.strip())
        elif elem.name == 'meter' or elem.name == 'output':
            return elem.get('value', elem.text.strip())
        else:
            return elem.text.strip()
    
    def extract_rdfa(self, html_content: str, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Извлекает данные RDFa со страницы.
        
        Args:
            html_content: HTML-содержимое страницы
            base_url: Базовый URL для резолвинга относительных ссылок
            
        Returns:
            List[Dict[str, Any]]: Список объектов RDFa
        """
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Находим все элементы с typeof (это аналог itemscope в RDFa)
            items = soup.find_all(attrs={'typeof': True})
            
            for item in items:
                # Пропускаем вложенные элементы, которые уже будут обработаны при рекурсивном обходе
                parent_with_typeof = item.find_parent(attrs={'typeof': True})
                if parent_with_typeof and parent_with_typeof != item:
                    continue
                
                # Обрабатываем элемент
                item_data = self._extract_rdfa_item(item, base_url)
                if item_data:
                    results.append(item_data)
        except Exception as e:
            logger.error(f"Error extracting RDFa: {str(e)}")
        
        return results
    
    def _extract_rdfa_item(self, item, base_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Рекурсивно извлекает данные из элемента RDFa.
        
        Args:
            item: Элемент BeautifulSoup с атрибутом typeof
            base_url: Базовый URL для резолвинга относительных ссылок
            
        Returns:
            Optional[Dict[str, Any]]: Данные элемента RDFa или None
        """
        if not item.has_attr('typeof'):
            return None
        
        result = {}
        
        # Получаем тип элемента
        typeof = item['typeof']
        # Если это URL типа Schema.org, извлекаем только название типа
        if isinstance(typeof, str) and ('schema.org' in typeof or 'Schema.org' in typeof):
            type_name = typeof.split('/')[-1]
            result['@type'] = type_name
        else:
            result['@type'] = typeof
        
        # Получаем id элемента
        if item.has_attr('resource'):
            result['@id'] = item['resource']
        elif item.has_attr('about'):
            result['@id'] = item['about']
        
        # Получаем свойства элемента
        props = {}
        
        # Обрабатываем свойства самого элемента
        if item.has_attr('property'):
            for prop_name in item['property'].split():
                # Если это URL свойства Schema.org, извлекаем только название свойства
                if 'schema.org' in prop_name or 'Schema.org' in prop_name:
                    prop_name = prop_name.split('/')[-1]
                props[prop_name] = self._get_rdfa_value(item, base_url)
        
        # Обрабатываем все дочерние элементы с property
        for prop_elem in item.find_all(attrs={'property': True}, recursive=True):
            # Пропускаем элементы, которые находятся внутри других элементов с typeof
            parent_with_typeof = prop_elem.find_parent(attrs={'typeof': True})
            if parent_with_typeof and parent_with_typeof != item:
                continue
            
            for prop_name in prop_elem['property'].split():
                # Если это URL свойства Schema.org, извлекаем только название свойства
                if 'schema.org' in prop_name or 'Schema.org' in prop_name:
                    prop_name = prop_name.split('/')[-1]
                
                # Если элемент имеет typeof, обрабатываем его рекурсивно
                if prop_elem.has_attr('typeof'):
                    prop_value = self._extract_rdfa_item(prop_elem, base_url)
                else:
                    prop_value = self._get_rdfa_value(prop_elem, base_url)
                
                # Добавляем свойство
                if prop_name in props:
                    if isinstance(props[prop_name], list):
                        props[prop_name].append(prop_value)
                    else:
                        props[prop_name] = [props[prop_name], prop_value]
                else:
                    props[prop_name] = prop_value
        
        # Объединяем свойства с результатом
        result.update(props)
        
        return result
    
    def _get_rdfa_value(self, elem, base_url: Optional[str] = None) -> Any:
        """
        Получает значение элемента RDFa.
        
        Args:
            elem: Элемент BeautifulSoup
            base_url: Базовый URL для резолвинга относительных ссылок
            
        Returns:
            Any: Значение элемента
        """
        # Проверяем различные атрибуты в порядке приоритета
        if elem.has_attr('content'):
            return elem['content']
        elif elem.has_attr('resource'):
            return elem['resource']
        elif elem.has_attr('href'):
            href = elem['href']
            if base_url and href and not (href.startswith('http://') or href.startswith('https://') or href.startswith('//')):
                return urljoin(base_url, href)
            return href
        elif elem.has_attr('src'):
            src = elem['src']
            if base_url and src and not (src.startswith('http://') or src.startswith('https://') or src.startswith('//')):
                return urljoin(base_url, src)
            return src
        elif elem.has_attr('datetime'):
            return elem['datetime']
        else:
            return elem.text.strip()
    
    def extract_all_structured_data(self, html_content: str, base_url: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Извлекает все структурированные данные со страницы.
        
        Args:
            html_content: HTML-содержимое страницы
            base_url: Базовый URL для резолвинга относительных ссылок
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Словарь с разными типами структурированных данных
        """
        try:
            # Если base_url не указан, пытаемся извлечь его из HTML
            if not base_url:
                soup = BeautifulSoup(html_content, 'html.parser')
                base_tag = soup.find('base', href=True)
                if base_tag:
                    base_url = base_tag['href']
                    
                    # Проверяем, является ли URL абсолютным
                    parsed_url = urlparse(base_url)
                    if not parsed_url.scheme:
                        logger.warning("Base URL does not have a scheme, ignoring it")
                        base_url = None
            
            result = {
                'jsonld': self.extract_jsonld(html_content),
                'microdata': self.extract_microdata(html_content, base_url),
                'rdfa': self.extract_rdfa(html_content, base_url)
            }
            
            # Добавляем агрегированные данные
            result['aggregated'] = self._aggregate_structured_data(result)
            
            return result
        except Exception as e:
            logger.error(f"Error extracting all structured data: {str(e)}")
            return {
                'jsonld': [],
                'microdata': [],
                'rdfa': [],
                'aggregated': []
            }
    
    def _aggregate_structured_data(self, structured_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Агрегирует структурированные данные из разных источников.
        
        Args:
            structured_data: Словарь с разными типами структурированных данных
            
        Returns:
            List[Dict[str, Any]]: Агрегированные структурированные данные
        """
        # Объединяем все данные
        all_data = structured_data['jsonld'] + structured_data['microdata'] + structured_data['rdfa']
        
        # Создаем словарь для хранения уникальных данных по типу
        aggregated_by_type = defaultdict(list)
        
        for item in all_data:
            # Пропускаем элементы без типа
            if '@type' not in item:
                continue
            
            item_type = item['@type']
            
            # Нормализуем тип, если это список
            if isinstance(item_type, list):
                if not item_type:
                    continue
                item_type = item_type[0]
            
            # Добавляем элемент в соответствующий список
            aggregated_by_type[item_type].append(item)
        
        # Объединяем данные одинаковых типов
        result = []
        
        for item_type, items in aggregated_by_type.items():
            # Если есть только один элемент данного типа, добавляем его как есть
            if len(items) == 1:
                result.append(items[0])
                continue
            
            # Если есть несколько элементов, пытаемся объединить их
            merged_item = {'@type': item_type}
            
            # Для каждого свойства выбираем наиболее полное значение
            all_props = set()
            for item in items:
                all_props.update(item.keys())
            
            for prop in all_props:
                if prop == '@type':
                    continue
                
                prop_values = []
                for item in items:
                    if prop in item:
                        prop_value = item[prop]
                        if prop_value not in prop_values:
                            prop_values.append(prop_value)
                
                if len(prop_values) == 1:
                    merged_item[prop] = prop_values[0]
                elif len(prop_values) > 1:
                    # Для простых свойств берем самое длинное значение
                    if all(isinstance(v, (str, int, float, bool)) for v in prop_values):
                        if all(isinstance(v, str) for v in prop_values):
                            merged_item[prop] = max(prop_values, key=len)
                        else:
                            merged_item[prop] = prop_values[0]
                    else:
                        # Для сложных свойств берем список всех значений
                        merged_item[prop] = prop_values
            
            result.append(merged_item)
        
        return result
    
    def validate_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет корректность схемы разметки.
        
        Args:
            schema_data: Данные схемы разметки
            
        Returns:
            Dict[str, Any]: Результат валидации
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Проверяем наличие типа
        if '@type' not in schema_data:
            result['valid'] = False
            result['errors'].append("Missing @type property")
            result['recommendations'].append("Add @type property with a valid Schema.org type")
            return result
        
        schema_type = schema_data['@type']
        
        # Если тип является списком, проверяем первый элемент
        if isinstance(schema_type, list):
            if not schema_type:
                result['valid'] = False
                result['errors'].append("Empty @type list")
                result['recommendations'].append("Add at least one valid Schema.org type to @type list")
                return result
            schema_type = schema_type[0]
        
        # Удаляем префикс http://schema.org/ если он есть
        if schema_type.startswith("http://schema.org/") or schema_type.startswith("https://schema.org/"):
            schema_type = schema_type.split("/")[-1]
        
        # Проверяем тип
        if not self.validator.validate_type(schema_type):
            result['valid'] = False
            result['errors'].append(f"Invalid @type: {schema_type}")
            result['recommendations'].append(f"Replace {schema_type} with a valid Schema.org type")
        
        # Проверяем контекст
        if '@context' not in schema_data:
            result['warnings'].append("Missing @context property")
            result['recommendations'].append("Add @context property with value 'https://schema.org' or 'http://schema.org'")
        elif schema_data['@context'] != 'https://schema.org' and schema_data['@context'] != 'http://schema.org':
            result['warnings'].append(f"Unexpected @context value: {schema_data['@context']}")
            result['recommendations'].append("Set @context property to 'https://schema.org'")
        
        # Проверяем свойства
        for prop_name, prop_value in schema_data.items():
            # Пропускаем специальные свойства
            if prop_name.startswith('@'):
                continue
            
            # Проверяем свойство
            if not self.validator.validate_property(prop_name, schema_type):
                result['warnings'].append(f"Property '{prop_name}' may not be valid for type '{schema_type}'")
                result['recommendations'].append(f"Verify that '{prop_name}' is appropriate for type '{schema_type}' or remove it")
            
            # Проверяем вложенные схемы
            if isinstance(prop_value, dict) and '@type' in prop_value:
                nested_validation = self.validate_schema(prop_value)
                if not nested_validation['valid']:
                    result['valid'] = False
                    for error in nested_validation['errors']:
                        result['errors'].append(f"In property '{prop_name}': {error}")
                for warning in nested_validation['warnings']:
                    result['warnings'].append(f"In property '{prop_name}': {warning}")
                for recommendation in nested_validation['recommendations']:
                    result['recommendations'].append(f"In property '{prop_name}': {recommendation}")
            elif isinstance(prop_value, list):
                for i, item in enumerate(prop_value):
                    if isinstance(item, dict) and '@type' in item:
                        nested_validation = self.validate_schema(item)
                        if not nested_validation['valid']:
                            result['valid'] = False
                            for error in nested_validation['errors']:
                                result['errors'].append(f"In property '{prop_name}[{i}]': {error}")
                        for warning in nested_validation['warnings']:
                            result['warnings'].append(f"In property '{prop_name}[{i}]': {warning}")
                        for recommendation in nested_validation['recommendations']:
                            result['recommendations'].append(f"In property '{prop_name}[{i}]': {recommendation}")
        
        # Проверяем наличие важных свойств для данного типа
        if schema_type in self.important_properties:
            important_props = self.important_properties[schema_type]
            missing_props = [prop for prop in important_props if prop not in schema_data]
            if missing_props:
                for prop in missing_props:
                    result['warnings'].append(f"Missing important property '{prop}' for type '{schema_type}'")
                    result['recommendations'].append(f"Add '{prop}' property to improve schema completeness")
        
        # Проверяем правила оптимизации для данного типа
        if schema_type in self.optimization_rules:
            rules = self.optimization_rules[schema_type]
            
            # Проверяем обязательные свойства
            if 'required' in rules:
                for prop in rules['required']:
                    if prop not in schema_data:
                        result['valid'] = False
                        result['errors'].append(f"Missing required property '{prop}' for type '{schema_type}'")
                        result['recommendations'].append(f"Add '{prop}' property to ensure schema validity")
            
            # Проверяем рекомендуемые свойства
            if 'recommended' in rules:
                for prop in rules['recommended']:
                    if prop not in schema_data:
                        result['warnings'].append(f"Missing recommended property '{prop}' for type '{schema_type}'")
                        result['recommendations'].append(f"Add '{prop}' property to improve schema completeness")
            
            # Проверяем минимальную длину строковых свойств
            if 'min_length' in rules:
                for prop, min_length in rules['min_length'].items():
                    if prop in schema_data and isinstance(schema_data[prop], str) and len(schema_data[prop]) < min_length:
                        result['warnings'].append(f"Property '{prop}' is too short (min length: {min_length})")
                        result['recommendations'].append(f"Expand '{prop}' to at least {min_length} characters")
        
        return result
    
    def optimize_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизирует схему разметки для улучшения совместимости с LLM.
        
        Args:
            schema_data: Данные схемы разметки
            
        Returns:
            Dict[str, Any]: Оптимизированная схема
        """
        result = schema_data.copy()
        
        # Проверяем наличие типа
        if '@type' not in result:
            result['@type'] = 'Thing'
        
        schema_type = result['@type']
        
        # Если тип является списком, используем первый элемент
        if isinstance(schema_type, list):
            if not schema_type:
                schema_type = 'Thing'
                result['@type'] = schema_type
            else:
                schema_type = schema_type[0]
        
        # Удаляем префикс http://schema.org/ если он есть
        if schema_type.startswith("http://schema.org/") or schema_type.startswith("https://schema.org/"):
            schema_type = schema_type.split("/")[-1]
        
        # Добавляем контекст, если его нет
        if '@context' not in result:
            result['@context'] = 'https://schema.org'
        
        # Нормализуем URL контекста
        if result['@context'] == 'http://schema.org':
            result['@context'] = 'https://schema.org'
        
        # Добавляем важные свойства с заглушками для данного типа
        if schema_type in self.important_properties:
            important_props = self.important_properties[schema_type]
            for prop in important_props:
                if prop not in result:
                    # Добавляем заглушку для свойства
                    if prop in ['name', 'headline', 'title']:
                        result[prop] = "[Untitled]"
                    elif prop in ['description', 'text', 'reviewBody', 'articleBody']:
                        result[prop] = "[No description provided]"
                    elif prop in ['datePublished', 'dateCreated', 'dateModified', 'startDate', 'endDate']:
                        result[prop] = datetime.datetime.now().strftime('%Y-%m-%d')
                    elif prop in ['image']:
                        result[prop] = {
                            '@type': 'ImageObject',
                            'url': '[Image URL]',
                            'width': 1200,
                            'height': 630
                        }
                    elif prop in ['author', 'publisher', 'provider', 'creator', 'founder', 'performer', 'organizer']:
                        if prop not in result:
                            result[prop] = {
                                '@type': 'Organization',
                                'name': '[Organization Name]'
                            }
                    elif prop in ['brand']:
                        result[prop] = {
                            '@type': 'Brand',
                            'name': '[Brand Name]'
                        }
                    elif prop in ['offers']:
                        result[prop] = {
                            '@type': 'Offer',
                            'price': '0',
                            'priceCurrency': 'USD',
                            'availability': 'https://schema.org/InStock'
                        }
                    elif prop in ['aggregateRating']:
                        result[prop] = {
                            '@type': 'AggregateRating',
                            'ratingValue': '5',
                            'reviewCount': '1'
                        }
                    elif prop in ['mainEntity']:
                        if schema_type == 'FAQPage':
                            result[prop] = [{
                                '@type': 'Question',
                                'name': '[Question]',
                                'acceptedAnswer': {
                                    '@type': 'Answer',
                                    'text': '[Answer]'
                                }
                            }]
                    elif prop in ['reviewRating']:
                        result[prop] = {
                            '@type': 'Rating',
                            'ratingValue': '5',
                            'bestRating': '5',
                            'worstRating': '1'
                        }
                    elif prop in ['itemReviewed']:
                        result[prop] = {
                            '@type': 'Thing',
                            'name': '[Item Name]'
                        }
                    elif prop in ['recipeIngredient']:
                        result[prop] = ['[Ingredient 1]', '[Ingredient 2]']
                    elif prop in ['recipeInstructions']:
                        result[prop] = [{
                            '@type': 'HowToStep',
                            'text': '[Instruction Step]'
                        }]
                    elif prop in ['step']:
                        result[prop] = [{
                            '@type': 'HowToStep',
                            'text': '[Step 1]'
                        }]
        
        # Нормализуем даты в формате ISO 8601
        for prop, value in result.items():
            if prop in ['datePublished', 'dateCreated', 'dateModified', 'startDate', 'endDate'] and isinstance(value, str):
                # Попытка преобразовать в формат ISO 8601
                try:
                    date_formats = [
                        '%Y-%m-%d',             # 2023-01-01
                        '%Y-%m-%dT%H:%M:%S',    # 2023-01-01T12:00:00
                        '%Y-%m-%dT%H:%M:%SZ',   # 2023-01-01T12:00:00Z
                        '%Y/%m/%d',             # 2023/01/01
                        '%d.%m.%Y',             # 01.01.2023
                        '%d/%m/%Y',             # 01/01/2023
                        '%m/%d/%Y',             # 01/01/2023
                        '%B %d, %Y',            # January 01, 2023
                        '%d %B %Y',             # 01 January 2023
                        '%Y'                    # 2023
                    ]
                    
                    parsed_date = None
                    for date_format in date_formats:
                        try:
                            parsed_date = datetime.datetime.strptime(value, date_format)
                            break
                        except ValueError:
                            continue
                    
                    if parsed_date:
                        result[prop] = parsed_date.strftime('%Y-%m-%d')
                except Exception:
                    # Оставляем как есть, если не удалось преобразовать
                # Оставляем как есть, если не удалось преобразовать
                self.logger.warning(f"Не удалось преобразовать дату '{value}' для свойства {prop}")
                
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
        
        # Рекурсивно оптимизируем вложенные объекты
        for prop, value in result.items():
            if prop.startswith('@'):
                continue
                
            if isinstance(value, dict) and '@type' in value:
                result[prop] = self.optimize_schema(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict) and '@type' in item:
                        value[i] = self.optimize_schema(item)
        
        return result
    
    def generate_recommendations(self, schema_data: Dict[str, Any]) -> List[str]:
        """
        Генерирует рекомендации по улучшению схемы разметки для LLM.
        
        Args:
            schema_data: Данные схемы разметки
            
        Returns:
            List[str]: Список рекомендаций
        """
        # Валидируем схему
        validation_result = self.validate_schema(schema_data)
        
        # Возвращаем рекомендации из результата валидации
        return validation_result['recommendations']
    
    def add_llm_tags(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Добавляет теги для LLM в схему разметки.
        
        Args:
            schema_data: Данные схемы разметки
            
        Returns:
            Dict[str, Any]: Схема с добавленными тегами для LLM
        """
        result = schema_data.copy()
        
        # Проверяем наличие типа
        if '@type' not in result:
            return result
        
        schema_type = result['@type']
        
        # Если тип является списком, используем первый элемент
        if isinstance(schema_type, list):
            if not schema_type:
                return result
            schema_type = schema_type[0]
        
        # Удаляем префикс http://schema.org/ если он есть
        if schema_type.startswith("http://schema.org/") or schema_type.startswith("https://schema.org/"):
            schema_type = schema_type.split("/")[-1]
        
        # Добавляем теги для LLM
        if schema_type in self.llm_tag_mapping:
            result['llm_tags'] = self.llm_tag_mapping[schema_type]
        
        # Рекурсивно добавляем теги для вложенных объектов
        for prop, value in result.items():
            if prop.startswith('@') or prop == 'llm_tags':
                continue
                
            if isinstance(value, dict) and '@type' in value:
                result[prop] = self.add_llm_tags(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict) and '@type' in item:
                        value[i] = self.add_llm_tags(item)
        
        return result
    
    def analyze_page_schema(self, html_content: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализирует разметку Schema.org на странице и возвращает подробный отчет.
        
        Args:
            html_content: HTML-содержимое страницы
            url: URL страницы (для резолвинга относительных ссылок)
            
        Returns:
            Dict[str, Any]: Подробный отчет
        """
        # Извлекаем все структурированные данные
        structured_data = self.extract_all_structured_data(html_content, url)
        
        # Создаем отчет
        report = {
            'url': url,
            'timestamp': datetime.datetime.now().isoformat(),
            'data_sources': {
                'jsonld': {
                    'count': len(structured_data['jsonld']),
                    'items': structured_data['jsonld']
                },
                'microdata': {
                    'count': len(structured_data['microdata']),
                    'items': structured_data['microdata']
                },
                'rdfa': {
                    'count': len(structured_data['rdfa']),
                    'items': structured_data['rdfa']
                },
                'aggregated': {
                    'count': len(structured_data['aggregated']),
                    'items': structured_data['aggregated']
                }
            },
            'types': {},
            'validation': {},
            'llm_compatibility': {},
            'recommendations': []
        }
        
        # Анализируем типы
        all_items = structured_data['jsonld'] + structured_data['microdata'] + structured_data['rdfa']
        types_count = defaultdict(int)
        
        for item in all_items:
            if '@type' in item:
                item_type = item['@type']
                
                # Нормализуем тип, если это список
                if isinstance(item_type, list):
                    if not item_type:
                        continue
                    item_type = item_type[0]
                
                # Удаляем префикс http://schema.org/ если он есть
                if item_type.startswith("http://schema.org/") or item_type.startswith("https://schema.org/"):
                    item_type = item_type.split("/")[-1]
                
                types_count[item_type] += 1
        
        report['types'] = dict(types_count)
        
        # Проверяем важные типы
        important_types_found = []
        for type_name in self.important_types:
            if type_name in types_count:
                important_types_found.append(type_name)
        
        report['important_types_found'] = important_types_found
        
        # Валидируем каждый элемент
        for i, item in enumerate(all_items):
            if '@type' not in item:
                continue
                
            item_type = item['@type']
            
            # Нормализуем тип, если это список
            if isinstance(item_type, list):
                if not item_type:
                    continue
                item_type = item_type[0]
            
            # Удаляем префикс http://schema.org/ если он есть
            if item_type.startswith("http://schema.org/") or item_type.startswith("https://schema.org/"):
                item_type = item_type.split("/")[-1]
            
            validation_result = self.validate_schema(item)
            report['validation'][f'item_{i}_{item_type}'] = validation_result
            
            # Добавляем рекомендации
            for recommendation in validation_result['recommendations']:
                if recommendation not in report['recommendations']:
                    report['recommendations'].append(recommendation)
            
            # Проверяем совместимость с LLM
            if item_type in self.important_types:
                required_props = set(self.important_properties.get(item_type, []))
                existing_props = set(prop for prop in item.keys() if not prop.startswith('@'))
                missing_props = required_props - existing_props
                
                report['llm_compatibility'][f'item_{i}_{item_type}'] = {
                    'compatible': len(missing_props) == 0,
                    'missing_properties': list(missing_props),
                    'llm_tags': self.llm_tag_mapping.get(item_type, [])
                }
        
        # Общие рекомендации
        if not all_items:
            report['recommendations'].append("Add structured data to the page using JSON-LD format")
        
        if not any('@context' in item and item['@context'] in ['http://schema.org', 'https://schema.org'] for item in all_items if isinstance(item, dict)):
            report['recommendations'].append("Add @context property with value 'https://schema.org' to all JSON-LD data")
        
        # Рекомендации по важным типам
        missing_important_types = set(self.important_types) - set(important_types_found)
        for type_name in missing_important_types:
            # Рекомендуем только релевантные типы
            if type_name == 'Article' or type_name == 'NewsArticle' or type_name == 'BlogPosting':
                # Проверяем, есть ли на странице большие текстовые блоки
                soup = BeautifulSoup(html_content, 'html.parser')
                article_tags = soup.find_all(['article', 'main', 'div'], class_=['post', 'article', 'content', 'entry'])
                
                if article_tags:
                    report['recommendations'].append(f"Consider adding {type_name} structured data for main content")
            
            elif type_name == 'Product':
                # Проверяем, есть ли на странице элементы, характерные для продуктов
                soup = BeautifulSoup(html_content, 'html.parser')
                product_indicators = soup.find_all(['div', 'section'], class_=['product', 'item', 'price'])
                
                if product_indicators:
                    report['recommendations'].append(f"Consider adding {type_name} structured data for product information")
            
            elif type_name == 'FAQPage':
                # Проверяем, есть ли на странице структура вопрос-ответ
                soup = BeautifulSoup(html_content, 'html.parser')
                faq_indicators = soup.find_all(['div', 'section', 'ul'], class_=['faq', 'faqs', 'questions', 'qa'])
                
                if faq_indicators:
                    report['recommendations'].append(f"Consider adding {type_name} structured data for FAQ sections")
        
        return report

def improved_date_processing(self, prop, value, result):
    try:
        # Попытка разбора различных форматов даты
        parsed_date = dateutil.parser.parse(value)
        result[prop] = parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        self.logger.warning(f"Не удалось преобразовать дату '{value}' для свойства {prop}: {str(e)}")
        
        # Попытка извлечь дату с использованием регулярных выражений
        # Форматы: YYYY-MM-DD, DD.MM.YYYY, MM/DD/YYYY, и т.д.
        date_patterns = [
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD или YYYY/MM/DD
            r'(\d{1,2})[/.](\d{1,2})[/.](\d{4})',  # DD.MM.YYYY или DD/MM/YYYY
            r'(\d{4})(\d{2})(\d{2})'               # YYYYMMDD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, value)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if len(groups[0]) == 4:  # Первая группа - год (YYYY-MM-DD)
                        year, month, day = groups
                    elif len(groups[2]) == 4:  # Последняя группа - год (DD.MM.YYYY)
                        day, month, year = groups
                    
                    try:
                        # Проверяем, что значения в допустимом диапазоне
                        year_int = int(year)
                        month_int = int(month)
                        day_int = int(day)
                        
                        if 1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100:
                            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            result[prop] = formatted_date
                            self.logger.info(f"Дата успешно преобразована альтернативным способом: {value} -> {formatted_date}")
                            return
                    except ValueError:
                        continue
        
        # Если не удалось разобрать дату, оставляем оригинальное значение
        self.logger.info(f"Оставляем оригинальное значение даты: {value}")
        result[prop] = value



# Функция для создания экземпляра SchemaOptimizer с параметрами по умолчанию
def create_schema_optimizer(**kwargs) -> SchemaOptimizer:
    """
    Создает экземпляр SchemaOptimizer с настройками.
    
    Args:
        **kwargs: Параметры для создания SchemaValidator
        
    Returns:
        SchemaOptimizer: Экземпляр оптимизатора схем
    """
    validator = SchemaValidator(**kwargs)
    return SchemaOptimizer(validator=validator)
