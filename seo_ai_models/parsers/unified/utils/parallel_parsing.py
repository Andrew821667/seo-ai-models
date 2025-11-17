"""
Многопоточный парсинг для ускорения обработки больших сайтов.
Использует ThreadPoolExecutor для параллельного выполнения задач парсинга.
"""

import logging
import time
import concurrent.futures
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from urllib.parse import urlparse
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ParallelParser:
    """
    Обеспечивает параллельное выполнение задач парсинга с контролем нагрузки.
    """

    def __init__(
        self,
        max_workers: int = 10,
        rate_limit: float = 0.5,
        per_domain_rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        respect_robots: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Инициализация многопоточного парсера.

        Args:
            max_workers: Максимальное количество потоков
            rate_limit: Минимальный интервал между запросами в секундах
            per_domain_rate_limit: Минимальный интервал между запросами к одному домену в секундах
            max_retries: Максимальное количество повторных попыток
            timeout: Таймаут для выполнения задачи в секундах
            respect_robots: Уважать ли robots.txt
            progress_callback: Функция обратного вызова для отображения прогресса
        """
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.per_domain_rate_limit = per_domain_rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.progress_callback = progress_callback

        # Блокировки для управления доступом к общим ресурсам
        self.lock = threading.Lock()

        # Время последнего запроса (общее и по доменам)
        self.last_request_time = 0
        self.domain_last_request = {}

        # Статистика выполнения
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0,
            "timeouts": 0,
            "start_time": 0,
            "end_time": 0,
        }

        logger.info(f"ParallelParser initialized with {max_workers} workers")

    def parse_urls(
        self,
        urls: List[str],
        parse_function: Callable[[str, Dict[str, Any]], Any],
        function_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Параллельный парсинг списка URL с использованием заданной функции.

        Args:
            urls: Список URL для парсинга
            parse_function: Функция для парсинга одного URL
            function_kwargs: Дополнительные аргументы для функции парсинга

        Returns:
            Dict[str, Any]: Результаты парсинга
        """
        if not urls:
            return {"error": "Empty URL list", "results": {}}

        # Инициализация статистики
        self.stats["total_tasks"] = len(urls)
        self.stats["completed_tasks"] = 0
        self.stats["failed_tasks"] = 0
        self.stats["retried_tasks"] = 0
        self.stats["timeouts"] = 0
        self.stats["start_time"] = time.time()

        # Подготовка аргументов
        function_kwargs = function_kwargs or {}
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Отправка задач на выполнение
            future_to_url = {
                executor.submit(
                    self._parse_url_with_rate_limit, url, parse_function, function_kwargs
                ): url
                for url in urls
            }

            # Обработка результатов по мере завершения
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]

                try:
                    result = future.result()
                    results[url] = result

                    with self.lock:
                        self.stats["completed_tasks"] += 1

                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    results[url] = {"error": str(e)}

                    with self.lock:
                        self.stats["failed_tasks"] += 1

                # Обновляем прогресс
                if self.progress_callback:
                    with self.lock:
                        completed = self.stats["completed_tasks"]
                        total = self.stats["total_tasks"]
                    self.progress_callback(completed, total)

        # Завершающая статистика
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]

        logger.info(f"Parallel parsing completed in {duration:.2f}s")
        logger.info(f"Tasks: {self.stats['completed_tasks']}/{self.stats['total_tasks']} completed")
        logger.info(
            f"Failed: {self.stats['failed_tasks']}, Retries: {self.stats['retried_tasks']}, Timeouts: {self.stats['timeouts']}"
        )

        return {"results": results, "stats": self.stats, "duration": duration}

    def _parse_url_with_rate_limit(
        self,
        url: str,
        parse_function: Callable[[str, Dict[str, Any]], Any],
        function_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Парсинг URL с учетом ограничений скорости и повторных попыток.

        Args:
            url: URL для парсинга
            parse_function: Функция для парсинга
            function_kwargs: Дополнительные аргументы

        Returns:
            Any: Результат парсинга
        """
        domain = urlparse(url).netloc

        for attempt in range(self.max_retries):
            try:
                # Применяем ограничение скорости
                self._apply_rate_limit(domain)

                # Выполняем парсинг с таймаутом
                result = self._run_with_timeout(parse_function, url, function_kwargs)

                return result

            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout parsing {url} (attempt {attempt+1}/{self.max_retries})")

                with self.lock:
                    self.stats["timeouts"] += 1

            except Exception as e:
                logger.warning(
                    f"Error parsing {url}: {str(e)} (attempt {attempt+1}/{self.max_retries})"
                )

            # Отмечаем повторную попытку
            with self.lock:
                self.stats["retried_tasks"] += 1

        # Если все попытки исчерпаны, выбрасываем исключение
        raise Exception(f"Failed to parse {url} after {self.max_retries} attempts")

    def _apply_rate_limit(self, domain: str):
        """
        Применяет ограничение скорости запросов.

        Args:
            domain: Домен для ограничения скорости
        """
        with self.lock:
            # Получаем текущее время
            current_time = time.time()

            # Глобальное ограничение скорости
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last_request)

            # Ограничение скорости для конкретного домена
            if domain in self.domain_last_request:
                time_since_domain_request = current_time - self.domain_last_request[domain]
                if time_since_domain_request < self.per_domain_rate_limit:
                    time.sleep(self.per_domain_rate_limit - time_since_domain_request)

            # Обновляем время последнего запроса
            self.last_request_time = time.time()
            self.domain_last_request[domain] = time.time()

    def _run_with_timeout(
        self,
        parse_function: Callable[[str, Dict[str, Any]], Any],
        url: str,
        function_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Выполняет функцию парсинга с таймаутом.

        Args:
            parse_function: Функция для парсинга
            url: URL для парсинга
            function_kwargs: Дополнительные аргументы

        Returns:
            Any: Результат парсинга

        Raises:
            concurrent.futures.TimeoutError: Если выполнение превысило таймаут
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(parse_function, url, **function_kwargs)
            return future.result(timeout=self.timeout)
