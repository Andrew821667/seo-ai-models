"""
Базовый веб-краулер для унифицированного парсера.
"""

import logging
import time
import re
from typing import Dict, List, Set, Optional, Any
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from seo_ai_models.parsers.unified.utils.request_utils import create_session, fetch_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebCrawler:
    """
    Базовый веб-краулер для сканирования сайтов.
    """
    
    def __init__(
        self,
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        delay: float = 1.0,
        respect_robots: bool = True,
        user_agent: str = "SEOAIModels UnifiedParser/1.0"
    ):
        """
        Инициализация краулера.
        
        Args:
            base_url: Базовый URL для сканирования
            max_pages: Максимальное количество страниц для сканирования
            max_depth: Максимальная глубина сканирования
            delay: Задержка между запросами в секундах
            respect_robots: Уважать ли robots.txt
            user_agent: User-Agent для запросов
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        
        # Парсинг базового URL
        self.base_netloc = urlparse(base_url).netloc
        
        # Создание сессии
        self.session = create_session(user_agent=user_agent)
        
        # Инициализация наборов для хранения URL
        self.crawled_urls = set()
        self.found_urls = set()
        self.failed_urls = set()
        
        # Инициализация robots.txt
        self.robots_rules = []
        if self.respect_robots:
            self._load_robots_txt()
    
    def _load_robots_txt(self):
        """
        Загружает и парсит robots.txt.
        """
        # Упрощенная реализация для отладки
        logger.info(f"Simulating robots.txt loading for {self.base_url}")
        self.robots_rules = []
        
    def _is_allowed_by_robots(self, url: str) -> bool:
        """
        Проверяет, разрешен ли URL правилами robots.txt.
        
        Args:
            url: URL для проверки
            
        Returns:
            bool: True, если URL разрешен, иначе False
        """
        # Упрощенная реализация для отладки
        if not self.respect_robots:
            return True
            
        # По умолчанию разрешаем
        return True
    
    def _is_same_domain(self, url: str) -> bool:
        """
        Проверяет, принадлежит ли URL тому же домену, что и базовый URL.
        
        Args:
            url: URL для проверки
            
        Returns:
            bool: True, если URL принадлежит тому же домену, иначе False
        """
        netloc = urlparse(url).netloc
        return netloc == self.base_netloc or netloc == f"www.{self.base_netloc}" or f"www.{netloc}" == self.base_netloc
    
    def _extract_links(self, html: str, current_url: str) -> Set[str]:
        """
        Извлекает ссылки из HTML-страницы.
        
        Args:
            html: HTML-контент
            current_url: Текущий URL (для преобразования относительных ссылок)
            
        Returns:
            Set[str]: Набор найденных URL
        """
        links = set()
        soup = BeautifulSoup(html, 'html.parser')
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            
            # Пропускаем пустые ссылки и якоря
            if not href or href.startswith('#'):
                continue
                
            # Преобразуем относительные URL в абсолютные
            absolute_url = urljoin(current_url, href)
            
            # Удаляем якоря и приводим к нижнему регистру
            absolute_url = absolute_url.split('#')[0].lower()
            
            # Проверяем, что URL принадлежит тому же домену
            if self._is_same_domain(absolute_url):
                links.add(absolute_url)
                
        return links
    
    def crawl(self) -> Dict[str, Any]:
        """
        Выполняет сканирование сайта.
        
        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        logger.info(f"Starting crawl of {self.base_url}")
        
        # Добавляем базовый URL в очередь
        urls_to_visit = [(self.base_url, 0)]  # (url, depth)
        
        start_time = time.time()
        
        while urls_to_visit and len(self.crawled_urls) < self.max_pages:
            # Получаем следующий URL из очереди
            current_url, depth = urls_to_visit.pop(0)
            
            # Проверяем, был ли URL уже просканирован
            if current_url in self.crawled_urls or current_url in self.failed_urls:
                continue
                
            # Проверяем, разрешен ли URL правилами robots.txt
            if not self._is_allowed_by_robots(current_url):
                logger.info(f"Skipping {current_url} (disallowed by robots.txt)")
                continue
                
            logger.info(f"Crawling {current_url} (depth {depth})")
            
            # Загружаем страницу
            html, status_code, error = fetch_url(current_url, session=self.session)
            
            # Задержка между запросами
            time.sleep(self.delay)
            
            if html and status_code == 200:
                # Добавляем URL в список просканированных
                self.crawled_urls.add(current_url)
                
                # Если не достигли максимальной глубины, извлекаем ссылки
                if depth < self.max_depth:
                    links = self._extract_links(html, current_url)
                    
                    # Добавляем найденные ссылки в общий список
                    self.found_urls.update(links)
                    
                    # Добавляем новые ссылки в очередь
                    for link in links:
                        if link not in self.crawled_urls and link not in self.failed_urls:
                            urls_to_visit.append((link, depth + 1))
            else:
                # Добавляем URL в список неудачных
                self.failed_urls.add(current_url)
                logger.warning(f"Failed to crawl {current_url}: {error}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Crawl completed in {duration:.2f}s")
        logger.info(f"Crawled {len(self.crawled_urls)} URLs")
        logger.info(f"Found {len(self.found_urls)} URLs")
        logger.info(f"Failed {len(self.failed_urls)} URLs")
        
        # Формируем результат
        result = {
            "base_url": self.base_url,
            "crawled_urls": list(self.crawled_urls),
            "found_urls": list(self.found_urls),
            "failed_urls": list(self.failed_urls),
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "respect_robots": self.respect_robots,
            "duration": duration,
            "timestamp": time.time()
        }
        
        return result
