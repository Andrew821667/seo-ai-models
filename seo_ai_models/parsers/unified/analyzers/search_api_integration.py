"""
Интеграция с API поисковых систем для точного анализа поисковой выдачи.
Поддерживает Google Search API, SerpAPI и другие поисковые API.
"""

import logging
import time
import json
import os
import re
from typing import Dict, List, Optional, Any, Union
import random
import urllib.parse
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchAPIIntegration:
    """
    Интеграция с API поисковых систем для получения точных данных поисковой выдачи.
    Поддерживает различные API и имеет встроенную ротацию ключей и прокси.
    """
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        api_provider: str = "serpapi",
        proxies: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        country: str = "us",
        language: str = "en",
        results_count: int = 10,
        include_features: bool = True
    ):
        """
        Инициализация интеграции с API поисковых систем.
        
        Args:
            api_keys: Список API-ключей для ротации
            api_provider: Провайдер API ('serpapi', 'google', 'custom')
            proxies: Список прокси для ротации (формат: 'http://user:pass@host:port')
            max_retries: Максимальное количество повторных попыток
            retry_delay: Задержка между попытками в секундах
            country: Код страны для поиска
            language: Код языка для поиска
            results_count: Количество результатов для получения
            include_features: Включать ли расширенные функции поисковой выдачи
        """
        self.api_keys = api_keys or []
        self.api_provider = api_provider
        self.proxies = proxies or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.country = country
        self.language = language
        self.results_count = results_count
        self.include_features = include_features
        
        # Проверяем наличие API-ключей из переменных окружения
        self._load_api_keys_from_env()
        
        # Подсчет запросов для балансировки нагрузки
        self.request_count = 0
        
        logger.info(f"SearchAPIIntegration initialized with provider: {api_provider}")
        logger.info(f"Available API keys: {len(self.api_keys)}")
        logger.info(f"Available proxies: {len(self.proxies)}")
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Выполняет поисковый запрос через API.
        
        Args:
            query: Поисковый запрос
            **kwargs: Дополнительные параметры поиска
            
        Returns:
            Dict[str, Any]: Результаты поиска
        """
        # Инкрементируем счетчик запросов
        self.request_count += 1
        
        # Получаем API ключ с ротацией
        api_key = self._get_next_api_key()
        
        # Получаем прокси с ротацией
        proxy = self._get_next_proxy()
        
        # Выполняем запрос с повторными попытками
        for attempt in range(self.max_retries):
            try:
                if self.api_provider == "serpapi":
                    result = self._search_with_serpapi(query, api_key, proxy, **kwargs)
                elif self.api_provider == "google":
                    result = self._search_with_google_api(query, api_key, proxy, **kwargs)
                else:
                    result = self._search_with_custom_api(query, api_key, proxy, **kwargs)
                    
                return result
                
            except Exception as e:
                logger.error(f"Error searching for '{query}': {str(e)}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    
                    # Меняем ключ и прокси для следующей попытки
                    api_key = self._get_next_api_key()
                    proxy = self._get_next_proxy()
                else:
                    logger.error(f"Max retries reached. Failed to search for '{query}'")
                    
                    # В случае ошибки возвращаем заглушку
                    return self._generate_fallback_results(query)
    
    def _search_with_serpapi(self, query: str, api_key: str, proxy: Optional[str], **kwargs) -> Dict[str, Any]:
        """
        Выполняет поиск через SerpAPI.
        
        Args:
            query: Поисковый запрос
            api_key: API-ключ
            proxy: Прокси для запроса
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результаты поиска
        """
        # Если доступен SerpAPI клиент и ключ
        if not api_key:
            raise ValueError("SerpAPI requires an API key")
            
        # Базовые параметры
        params = {
            "q": query,
            "api_key": api_key,
            "num": kwargs.get("results_count", self.results_count),
            "gl": kwargs.get("country", self.country),
            "hl": kwargs.get("language", self.language)
        }
        
        # Добавляем дополнительные параметры
        if "device" in kwargs:
            params["device"] = kwargs["device"]
        if "location" in kwargs:
            params["location"] = kwargs["location"]
        
        # Выполняем запрос
        url = "https://serpapi.com/search"
        proxies = {"http": proxy, "https": proxy} if proxy else None
        
        response = requests.get(url, params=params, proxies=proxies)
        response.raise_for_status()
        
        data = response.json()
        
        # Обрабатываем результаты
        results = []
        for i, result in enumerate(data.get("organic_results", [])):
            results.append({
                "position": i + 1,
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "displayed_url": result.get("displayed_link", ""),
                "sitelinks": result.get("sitelinks", []),
                "rich_snippet": result.get("rich_snippet", None),
                "timestamp": time.time()
            })
            
        # Извлекаем связанные запросы
        related_searches = []
        for item in data.get("related_searches", []):
            related_searches.append(item.get("query", ""))
            
        # Извлекаем "Люди также спрашивают"
        people_also_ask = []
        for item in data.get("related_questions", []):
            people_also_ask.append({
                "question": item.get("question", ""),
                "snippet": item.get("snippet", "")
            })
            
        # Формируем итоговый результат
        return {
            "query": query,
            "results": results,
            "related_queries": related_searches,
            "people_also_ask": people_also_ask,
            "search_features": {
                "knowledge_panel": "knowledge_graph" in data,
                "top_stories": "top_stories" in data,
                "local_pack": "local_results" in data,
                "ads": "ads" in data
            },
            "search_metadata": {
                "id": data.get("search_metadata", {}).get("id", ""),
                "status": "success",
                "json_endpoint": data.get("search_metadata", {}).get("json_endpoint", ""),
                "created_at": data.get("search_metadata", {}).get("created_at", ""),
                "processed_at": data.get("search_metadata", {}).get("processed_at", ""),
                "google_url": data.get("search_metadata", {}).get("google_url", ""),
                "raw_html_file": data.get("search_metadata", {}).get("raw_html_file", ""),
                "total_time_taken": data.get("search_metadata", {}).get("total_time_taken", 0)
            },
            "search_parameters": {
                "engine": "google",
                "q": query,
                "gl": params.get("gl", self.country),
                "hl": params.get("hl", self.language)
            },
            "search_information": {
                "total_results": data.get("search_information", {}).get("total_results", 0),
                "time_taken_displayed": data.get("search_information", {}).get("time_taken_displayed", 0)
            }
        }
    
    def _search_with_google_api(self, query: str, api_key: str, proxy: Optional[str], **kwargs) -> Dict[str, Any]:
        """
        Выполняет поиск через Google Custom Search API.
        
        Args:
            query: Поисковый запрос
            api_key: API-ключ
            proxy: Прокси для запроса
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результаты поиска
        """
        if not api_key:
            raise ValueError("Google API requires an API key")
            
        # Здесь реализация запроса к Google Custom Search API
        # В текущей версии - упрощенная заглушка
        
        return self._generate_fallback_results(query)
    
    def _search_with_custom_api(self, query: str, api_key: str, proxy: Optional[str], **kwargs) -> Dict[str, Any]:
        """
        Выполняет поиск через пользовательский API.
        
        Args:
            query: Поисковый запрос
            api_key: API-ключ
            proxy: Прокси для запроса
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результаты поиска
        """
        # Пользовательская реализация поискового API
        # В текущей версии - упрощенная заглушка
        
        return self._generate_fallback_results(query)
    
    def _load_api_keys_from_env(self):
        """
        Загружает API-ключи из переменных окружения.
        """
        # Проверяем различные форматы переменных окружения
        for env_var in ["SERPAPI_KEY", "SERPAPI_KEYS", "SEARCH_API_KEY", "SEARCH_API_KEYS"]:
            if env_var in os.environ:
                keys = os.environ[env_var].split(",")
                keys = [k.strip() for k in keys if k.strip()]
                self.api_keys.extend(keys)
                
        # Удаляем дубликаты
        self.api_keys = list(set(self.api_keys))
    
    def _get_next_api_key(self) -> str:
        """
        Возвращает следующий API-ключ из доступных с ротацией.
        
        Returns:
            str: API-ключ или пустая строка
        """
        if not self.api_keys:
            return ""
            
        # Выбираем ключ по остатку от деления счетчика запросов на количество ключей
        key_index = self.request_count % len(self.api_keys)
        return self.api_keys[key_index]
    
    def _get_next_proxy(self) -> Optional[str]:
        """
        Возвращает следующий прокси из доступных с ротацией.
        
        Returns:
            Optional[str]: Прокси или None
        """
        if not self.proxies:
            return None
            
        # Выбираем прокси по остатку от деления счетчика запросов на количество прокси
        proxy_index = self.request_count % len(self.proxies)
        return self.proxies[proxy_index]
    
    def _generate_fallback_results(self, query: str) -> Dict[str, Any]:
        """
        Генерирует заглушку для результатов поиска в случае ошибки.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Dict[str, Any]: Заглушка результатов поиска
        """
        results = []
        
        # Создаем заглушку результатов
        for i in range(min(10, self.results_count)):
            results.append({
                "position": i + 1,
                "title": f"Sample Result {i+1} for {query}",
                "url": f"http://example.com/result{i+1}",
                "snippet": f"Sample snippet containing the query '{query}' and other information for result {i+1}.",
                "displayed_url": f"example.com/result{i+1}",
                "timestamp": time.time()
            })
            
        # Связанные запросы
        related_terms = ["guide", "tutorial", "example", "best", "review", "vs", "alternative"]
        related_queries = [f"{query} {term}" for term in related_terms[:min(7, len(related_terms))]]
        
        return {
            "query": query,
            "results": results,
            "related_queries": related_queries,
            "people_also_ask": [
                {"question": f"What is {query}?", "snippet": f"Sample answer about {query}..."},
                {"question": f"How to use {query}?", "snippet": f"Sample guide about using {query}..."},
                {"question": f"Why is {query} important?", "snippet": f"Sample explanation about importance of {query}..."}
            ],
            "search_features": {
                "knowledge_panel": False,
                "top_stories": False,
                "local_pack": False,
                "ads": False
            },
            "search_metadata": {
                "id": f"fallback_{int(time.time())}",
                "status": "fallback",
                "engine": "fallback",
                "created_at": time.time()
            },
            "note": "This is a fallback result generated due to API error"
        }
