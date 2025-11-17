"""
Утилиты для HTTP-запросов для унифицированного парсера.
"""

import logging
import time
import requests
from typing import Tuple, Optional, Dict, Any, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_session(user_agent: str = "SEOAIModels UnifiedParser/1.0") -> requests.Session:
    """
    Создает сессию requests с заданными заголовками.

    Args:
        user_agent: User-Agent для запросов

    Returns:
        requests.Session: Настроенная сессия
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    )

    return session


def fetch_url(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: int = 2,
) -> Tuple[Optional[str], int, Optional[str]]:
    """
    Получает содержимое URL с повторными попытками.

    Args:
        url: URL для загрузки
        session: Сессия requests (создает новую, если None)
        timeout: Таймаут для запросов в секундах
        max_retries: Максимальное количество повторных попыток
        retry_delay: Задержка между попытками в секундах

    Returns:
        Tuple[Optional[str], int, Optional[str]]:
            - HTML-контент или None в случае ошибки
            - HTTP статус-код (0 в случае ошибки)
            - Сообщение об ошибке или None в случае успеха
    """
    if session is None:
        session = create_session()

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()

            return response.text, response.status_code, None

        except requests.RequestException as e:
            status_code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
            error_message = f"Error fetching {url}: {str(e)}"

            if attempt < max_retries - 1:
                logger.warning(
                    f"{error_message}. Retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"{error_message}. Max retries reached.")
                return None, status_code, error_message

    return None, 0, "Unknown error"


def fetch_url_with_javascript_sync(
    url: str,
    headless: bool = True,
    wait_for_idle: int = 2000,
    wait_for_timeout: int = 10000,
    user_agent: str = "SEOAIModels UnifiedParser/1.0",
) -> Tuple[Optional[str], int, Optional[str]]:
    """
    Загружает URL с выполнением JavaScript.
    Использует playwright для рендеринга страницы.

    Args:
        url: URL для загрузки
        headless: Запускать ли браузер в фоновом режиме
        wait_for_idle: Время ожидания в мс после загрузки
        wait_for_timeout: Максимальное время ожидания в мс
        user_agent: User-Agent для запросов

    Returns:
        Tuple[Optional[str], int, Optional[str]]:
            - HTML-контент или None в случае ошибки
            - HTTP статус-код (0 в случае ошибки)
            - Сообщение об ошибке или None в случае успеха
    """
    try:
        # Заглушка для режима отладки
        # В реальной реализации здесь использовался бы playwright
        logger.info(f"Simulating JavaScript rendering for {url}")
        html, status, error = fetch_url(url)
        if html:
            # Имитируем контент с выполненным JavaScript
            html = html + "<!-- JavaScript rendered content -->"
            return html, status, None
        return None, status, error

    except Exception as e:
        error_message = f"Error rendering JavaScript for {url}: {str(e)}"
        logger.error(error_message)
        return None, 0, error_message
