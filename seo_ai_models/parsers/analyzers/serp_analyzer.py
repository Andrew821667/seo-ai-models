"""
SERP Analyzer модуль для проекта SEO AI Models.
Предоставляет функциональность для анализа страниц результатов поисковых систем.
"""

import logging
import re
import time
import random
from typing import Dict, List, Optional, Any, Union, Tuple
import json
from urllib.parse import urlparse, quote_plus

import requests
from bs4 import BeautifulSoup

from seo_ai_models.parsers.extractors.content_extractor import ContentExtractor
from seo_ai_models.parsers.extractors.meta_extractor import MetaExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SERPAnalyzer:
   """
   Анализирует страницы результатов поисковых систем и извлекает полезную информацию
   для SEO-анализа и исследования конкурентов.
   """
   
   def __init__(
       self,
       user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
       search_engine: str = "google",
       results_count: int = 10,
       pages_to_analyze: int = 5,
       delay: float = 2.0,
       content_extractor: Optional[ContentExtractor] = None,
       meta_extractor: Optional[MetaExtractor] = None
   ):
       """
       Инициализация SERPAnalyzer.
       
       Args:
           user_agent: User-Agent для запросов
           search_engine: Поисковая система для использования ('google', 'bing')
           results_count: Количество результатов для анализа
           pages_to_analyze: Количество верхних страниц для детального анализа
           delay: Задержка между запросами в секундах
           content_extractor: Экземпляр ContentExtractor для анализа контента
           meta_extractor: Экземпляр MetaExtractor для мета-анализа
       """
       self.user_agent = user_agent
       self.search_engine = search_engine.lower()
       self.results_count = results_count
       self.pages_to_analyze = pages_to_analyze
       self.delay = delay
       
       self.content_extractor = content_extractor or ContentExtractor()
       self.meta_extractor = meta_extractor or MetaExtractor()
       
       self.headers = {
           "User-Agent": self.user_agent,
           "Accept": "text/html,application/xhtml+xml,application/xml",
           "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
       }
       
       self.search_urls = {
           "google": "https://www.google.com/search?q={}&num={}",
           "bing": "https://www.bing.com/search?q={}&count={}"
       }
       
   def _extract_google_results(self, html: str) -> List[Dict[str, Any]]:
       """
       Извлечение результатов поиска из HTML Google.
       
       Args:
           html: HTML результатов поиска Google
           
       Returns:
           List[Dict]: Список результатов поиска
       """
       soup = BeautifulSoup(html, 'html.parser')
       results = []
       
       # Результаты Google находятся в <div class="g">
       result_divs = soup.find_all("div", class_="g")
       
       for div in result_divs:
           try:
               # Извлечение ссылки и заголовка
               link_element = div.find("a")
               if not link_element or not link_element.get("href"):
                   continue
                   
               url = link_element.get("href")
               
               # Пропуск не-http URL
               if not url.startswith(("http://", "https://")):
                   continue
                   
               title_element = div.find("h3")
               title = title_element.get_text() if title_element else ""
               
               # Извлечение сниппета
               snippet_element = div.find("div", class_="IsZvec")
               snippet = ""
               if snippet_element:
                   snippet = snippet_element.get_text(strip=True)
                   
               # Получение домена
               parsed_url = urlparse(url)
               domain = parsed_url.netloc
               
               results.append({
                   "position": len(results) + 1,
                   "title": title,
                   "url": url,
                   "domain": domain,
                   "snippet": snippet
               })
               
               # Собираем только столько результатов, сколько запрошено
               if len(results) >= self.results_count:
                   break
                   
           except Exception as e:
               logger.error(f"Ошибка извлечения результата: {e}")
               
       return results
   
   def _extract_bing_results(self, html: str) -> List[Dict[str, Any]]:
       """
       Извлечение результатов поиска из HTML Bing.
       
       Args:
           html: HTML результатов поиска Bing
           
       Returns:
           List[Dict]: Список результатов поиска
       """
       soup = BeautifulSoup(html, 'html.parser')
       results = []
       
       # Результаты Bing находятся в <li class="b_algo">
       result_items = soup.find_all("li", class_="b_algo")
       
       for item in result_items:
           try:
               # Извлечение ссылки и заголовка
               link_element = item.find("a")
               if not link_element or not link_element.get("href"):
                   continue
                   
               url = link_element.get("href")
               
               # Пропуск не-http URL
               if not url.startswith(("http://", "https://")):
                   continue
                   
               title = link_element.get_text()
               
               # Извлечение сниппета
               snippet_element = item.find("p")
               snippet = ""
               if snippet_element:
                   snippet = snippet_element.get_text(strip=True)
                   
               # Получение домена
               parsed_url = urlparse(url)
               domain = parsed_url.netloc
               
               results.append({
                   "position": len(results) + 1,
                   "title": title,
                   "url": url,
                   "domain": domain,
                   "snippet": snippet
               })
               
               # Собираем только столько результатов, сколько запрошено
               if len(results) >= self.results_count:
                   break
                   
           except Exception as e:
               logger.error(f"Ошибка извлечения результата: {e}")
               
       return results
   
   def search(self, query: str) -> List[Dict[str, Any]]:
       """
       Выполнение поиска и извлечение результатов.
       
       Args:
           query: Поисковый запрос
           
       Returns:
           List[Dict]: Список результатов поиска
       """
       if self.search_engine not in self.search_urls:
           raise ValueError(f"Неподдерживаемая поисковая система: {self.search_engine}")
           
       search_url = self.search_urls[self.search_engine].format(
           quote_plus(query), self.results_count
       )
       
       logger.info(f"Поиск по запросу: {query}")
       
       try:
           response = requests.get(search_url, headers=self.headers, timeout=10)
           response.raise_for_status()
           
           if self.search_engine == "google":
               results = self._extract_google_results(response.text)
           elif self.search_engine == "bing":
               results = self._extract_bing_results(response.text)
           else:
               results = []
               
           logger.info(f"Найдено {len(results)} результатов для запроса: {query}")
           return results
           
       except requests.RequestException as e:
           logger.error(f"Ошибка при поиске {query}: {e}")
           return []
           
   def analyze_top_results(self, query: str) -> Dict[str, Any]:
       """
       Анализ верхних результатов поиска для запроса.
       
       Args:
           query: Поисковый запрос
           
       Returns:
           Dict: Анализ верхних результатов поиска
       """
       search_results = self.search(query)
       
       if not search_results:
           logger.warning(f"Не найдены результаты поиска для запроса: {query}")
           return {
               "query": query,
               "results_count": 0,
               "search_engine": self.search_engine,
               "results": [],
               "analysis": {}
           }
           
       # Анализ верхних страниц подробно
       detailed_results = []
       
       for result in search_results[:self.pages_to_analyze]:
           try:
               # Получение контента страницы
               response = requests.get(result["url"], headers=self.headers, timeout=10)
               
               if response.status_code == 200:
                   # Извлечение контента и мета-информации
                   content_data = self.content_extractor.extract_content(response.text, result["url"])
                   meta_data = self.meta_extractor.extract_meta_information(response.text, result["url"])
                   
                   detailed_result = {
                       **result,
                       "content_analysis": content_data,
                       "meta_analysis": meta_data
                   }
                   
                   detailed_results.append(detailed_result)
                   
               else:
                   logger.warning(f"Не удалось получить контент с {result['url']}: HTTP {response.status_code}")
                   detailed_results.append(result)
                   
           except Exception as e:
               logger.error(f"Ошибка при анализе {result['url']}: {e}")
               detailed_results.append(result)
               
           # Уважительное сканирование - добавление задержки
           time.sleep(self.delay + random.uniform(0, 0.5))
           
       # Выполнение статистического анализа результатов поиска
       domains = {}
       title_lengths = []
       content_lengths = []
       heading_counts = {"h1": [], "h2": [], "h3": [], "h4": [], "h5": [], "h6": []}
       word_counts = []
       
       # Сбор общих данных для статистического анализа
       for result in detailed_results:
           # Частота доменов
           domain = result.get("domain", "")
           domains[domain] = domains.get(domain, 0) + 1
           
           # Длина заголовка
           if "title" in result:
               title_lengths.append(len(result["title"]))
               
           # Статистика контента, если доступна
           if "content_analysis" in result and "metadata" in result["content_analysis"]:
               metadata = result["content_analysis"]["metadata"]
               
               if "text_length" in metadata:
                   content_lengths.append(metadata["text_length"])
                   
               if "heading_counts" in metadata:
                   for h_level, count in metadata["heading_counts"].items():
                       if h_level in heading_counts:
                           heading_counts[h_level].append(count)
                           
           # Количество слов, если доступно
           if "meta_analysis" in result and "statistics" in result["meta_analysis"]:
               stats = result["meta_analysis"]["statistics"]
               if "word_count" in stats:
                   word_counts.append(stats["word_count"])
                   
       # Расчет средних значений и создание сводки анализа
       analysis = {
           "domains_frequency": domains,
           "title_length": {
               "average": sum(title_lengths) / len(title_lengths) if title_lengths else 0,
               "min": min(title_lengths) if title_lengths else 0,
               "max": max(title_lengths) if title_lengths else 0
           },
           "content_length": {
               "average": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
               "min": min(content_lengths) if content_lengths else 0,
               "max": max(content_lengths) if content_lengths else 0
           },
           "word_count": {
               "average": sum(word_counts) / len(word_counts) if word_counts else 0,
               "min": min(word_counts) if word_counts else 0,
               "max": max(word_counts) if word_counts else 0
           },
           "heading_counts": {
               h_level: {
                   "average": sum(counts) / len(counts) if counts else 0,
                   "min": min(counts) if counts else 0,
                   "max": max(counts) if counts else 0
               }
               for h_level, counts in heading_counts.items() if counts
           }
       }
       
       # Извлечение общих слов в заголовках и сниппетах
       all_titles = " ".join([r["title"] for r in search_results if "title" in r])
       all_snippets = " ".join([r["snippet"] for r in search_results if "snippet" in r])
       
       # Простое извлечение слов с подсчетом частоты
       title_words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
       snippet_words = re.findall(r'\b[a-zA-Z]{3,}\b', all_snippets.lower())
       
       common_title_words = self._get_word_frequency(title_words, top=20)
       common_snippet_words = self._get_word_frequency(snippet_words, top=20)
       
       analysis["common_words"] = {
           "titles": common_title_words,
           "snippets": common_snippet_words
       }
       
       return {
           "query": query,
           "results_count": len(search_results),
           "detailed_results_count": len(detailed_results),
           "search_engine": self.search_engine,
           "results": search_results,
           "detailed_results": detailed_results,
           "analysis": analysis
       }
       
   def _get_word_frequency(self, words: List[str], top: int = 10) -> Dict[str, int]:
       """
       Расчет частоты слов из списка слов.
       
       Args:
           words: Список слов
           top: Количество наиболее частых слов для возврата
           
       Returns:
           Dict: Частота слов
       """
       # Распространенные стоп-слова на английском для фильтрации
       stop_words = {
           'the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 
           'that', 'by', 'this', 'be', 'are', 'from', 'or', 'an', 'as', 'can', 
           'your', 'have', 'more', 'was', 'not', 'been', 'their', 'they', 'what'
       }
       
       # Подсчет частоты слов, исключая стоп-слова
       word_count = {}
       for word in words:
           if word not in stop_words:
               word_count[word] = word_count.get(word, 0) + 1
               
       # Получение N наиболее частых слов
       sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
       return dict(sorted_words[:top])
       
   def get_related_queries(self, query: str) -> List[str]:
       """
       Извлечение связанных запросов/предложений для поискового термина.
       
       Args:
           query: Поисковый запрос
           
       Returns:
           List[str]: Связанные запросы
       """
       related_queries = []
       
       if self.search_engine == "google":
           # Связанные поиски Google находятся внизу результатов поиска
           search_url = self.search_urls["google"].format(quote_plus(query), 10)
           
           try:
               response = requests.get(search_url, headers=self.headers, timeout=10)
               soup = BeautifulSoup(response.text, 'html.parser')
               
               # Поиск связанных запросов
               related_divs = soup.find_all("div", class_="BNeawe")
               for div in related_divs:
                   text = div.get_text().strip()
                   if text and text != query and len(text) < 100:  # Простая эвристика для связанных запросов
                       related_queries.append(text)
                       
           except Exception as e:
               logger.error(f"Ошибка при получении связанных запросов для {query}: {e}")
               
       # Удаление дубликатов с сохранением порядка
       seen = set()
       return [q for q in related_queries if not (q in seen or seen.add(q))]
