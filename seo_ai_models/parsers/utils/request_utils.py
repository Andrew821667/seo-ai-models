"""
Утилиты для запросов в SEO AI Models парсерах.
"""

import logging
import time
import random
from typing import Dict, Optional, Tuple, Union
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Tuple[int, ...] = (500, 502, 504),
    user_agent: Optional[str] = None,
    additional_headers: Optional[Dict[str, str]] = None
) -> requests.Session:
    """
    Создание сессии requests с функциональностью повторных попыток.
    
    Args:
        retries: Количество повторных попыток для неудачных запросов
        backoff_factor: Коэффициент задержки для повторных попыток
        status_forcelist: HTTP-коды состояния для повторных попыток
        user_agent: Строка User-Agent для использования
        additional_headers: Дополнительные заголовки для добавления в сессию
        
    Returns:
        requests.Session: Настроенная сессия
    """
    session = requests.Session()
    
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Установка заголовков по умолчанию
    default_headers = {
        "User-Agent": user_agent or "SEOAIModels Bot/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }
    
    if additional_headers:
        default_headers.update(additional_headers)
        
    session.headers.update(default_headers)
    
    return session

def fetch_url(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: Union[float, Tuple[float, float]] = (5, 15),
    delay: Optional[float] = None,
    verify_ssl: bool = True
) -> Tuple[Optional[str], int, Dict[str, str]]:
    """
    Получение содержимого URL с опциональной задержкой.
    
    Args:
        url: URL для получения
        session: Сессия requests для использования
        timeout: Таймаут запроса
        delay: Задержка перед запросом (в секундах)
        verify_ssl: Проверять ли SSL-сертификаты
        
    Returns:
        Tuple из (content, status_code, headers)
    """
    if delay:
        # Добавление случайности в задержку
        time.sleep(delay + random.uniform(0, delay * 0.25))
        
    # Использование предоставленной сессии или создание новой
    use_session = session or requests.Session()
    
    try:
        logger.info(f"Получение URL: {url}")
        
        response = use_session.get(
            url,
            timeout=timeout,
            verify=verify_ssl
        )
        
        return response.text, response.status_code, response.headers
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении {url}: {e}")
        return None, 0, {}
        
    finally:
        # Закрытие сессии, если мы создали ее
        if session is None:
            use_session.close()
