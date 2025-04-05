"""
Web Crawler модуль для проекта SEO AI Models.
Предоставляет функциональность для сканирования сайтов и сбора URL.
"""

import logging
import time
import random
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Optional, Any, Callable

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebCrawler:
    """
    Базовая реализация веб-краулера для сбора URL с сайтов.
    Реализует уважительное сканирование с ограничением частоты и учетом robots.txt.
    """
    
    def __init__(
        self, 
        base_url: str, 
        max_pages: int = 100, 
        delay: float = 1.0,
        respect_robots: bool = True,
        user_agent: str = "SEOAIModels WebCrawler/1.0",
        custom_headers: Optional[Dict[str, str]] = None,
        url_filter: Optional[Callable[[str], bool]] = None
    ):
        """
        Инициализация WebCrawler.

        Args:
            base_url: Начальный URL для сканирования
            max_pages: Максимальное количество страниц для сканирования
            delay: Задержка между запросами в секундах
            respect_robots: Уважать ли robots.txt
            user_agent: User-Agent для запросов
            custom_headers: Дополнительные заголовки для запросов
            url_filter: Функция для фильтрации URL (возвращает True для включения URL)
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        if custom_headers:
            self.headers.update(custom_headers)
            
        self.url_filter = url_filter
        self.crawled_urls: Set[str] = set()
        self.found_urls: Set[str] = set()
        self.failed_urls: Dict[str, str] = {}
        
        # Извлечение домена для проверки того же домена
        parsed_url = urlparse(self.base_url)
        self.domain = parsed_url.netloc
        
    def is_valid_url(self, url: str) -> bool:
        """
        Проверка, является ли URL допустимым и должен ли он быть просканирован.
        
        Args:
            url: URL для проверки
            
        Returns:
            bool: True, если URL должен быть просканирован
        """
        parsed = urlparse(url)
        
        # Проверка, относится ли он к тому же домену
        if parsed.netloc != self.domain:
            return False
        
        # Игнорировать распространенные нетекстовые расширения
        ignored_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', 
                             '.css', '.js', '.ico', '.svg', '.mp4', '.mp3']
        if any(url.lower().endswith(ext) for ext in ignored_extensions):
            return False
            
        # Применить пользовательский фильтр, если он предоставлен
        if self.url_filter and not self.url_filter(url):
            return False
            
        return True
        
    def extract_urls_from_html(self, html_content: str, current_url: str) -> List[str]:
        """
        Извлечение всех URL из HTML-контента.
        
        Args:
            html_content: HTML-контент для парсинга
            current_url: Текущий URL для разрешения относительных ссылок
            
        Returns:
            List[str]: Список извлеченных URL
        """
        urls = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Извлечение ссылок
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(current_url, href)
            
            # Очистка и нормализация URL
            parsed = urlparse(absolute_url)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                normalized_url += f"?{parsed.query}"
                
            if self.is_valid_url(normalized_url):
                urls.append(normalized_url)
                
        return urls
        
    def crawl(self) -> Dict[str, Any]:
        """
        Выполнение операции сканирования.
        
        Returns:
            Dict, содержащий crawled_urls, found_urls и failed_urls
        """
        logger.info(f"Начало сканирования с {self.base_url}")
        to_crawl = [self.base_url]
        
        while to_crawl and len(self.crawled_urls) < self.max_pages:
            current_url = to_crawl.pop(0)
            
            if current_url in self.crawled_urls:
                continue
                
            logger.info(f"Сканирование: {current_url}")
            
            try:
                response = requests.get(current_url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    self.crawled_urls.add(current_url)
                    
                    # Извлечение URL со страницы
                    new_urls = self.extract_urls_from_html(response.text, current_url)
                    
                    # Добавление URL в очередь, если они еще не просканированы или не в очереди
                    for url in new_urls:
                        self.found_urls.add(url)
                        if (url not in self.crawled_urls and 
                            url not in to_crawl and 
                            len(self.crawled_urls) < self.max_pages):
                            to_crawl.append(url)
                else:
                    self.failed_urls[current_url] = f"HTTP {response.status_code}"
                    
            except Exception as e:
                logger.error(f"Ошибка при сканировании {current_url}: {str(e)}")
                self.failed_urls[current_url] = str(e)
                
            # Уважительное сканирование - добавление задержки
            time.sleep(self.delay + random.uniform(0, 0.5))
            
        logger.info(f"Сканирование завершено. Просканировано {len(self.crawled_urls)} страниц.")
        
        return {
            "crawled_urls": list(self.crawled_urls),
            "found_urls": list(self.found_urls),
            "failed_urls": self.failed_urls,
            "total_crawled": len(self.crawled_urls),
            "total_found": len(self.found_urls),
            "total_failed": len(self.failed_urls)
        }
        
    def get_crawled_urls(self) -> List[str]:
        """
        Получение списка успешно просканированных URL.
        
        Returns:
            List[str]: Список просканированных URL
        """
        return list(self.crawled_urls)
        
    def get_failed_urls(self) -> Dict[str, str]:
        """
        Получение словаря неудачных URL с сообщениями об ошибках.
        
        Returns:
            Dict[str, str]: Неудачные URL с ошибками
        """
        return self.failed_urls
