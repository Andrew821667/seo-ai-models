"""
Request utilities для проекта SEO AI Models.
Предоставляет функции для HTTP-запросов и управления сессиями.
"""

import logging
import time
import asyncio
from typing import Dict, Optional, Any, Tuple, Union
import requests
from requests import Session
from requests.exceptions import RequestException

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "SEOAIModels/1.0"
DEFAULT_TIMEOUT = 30  # секунд

def create_session(
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
    custom_headers: Optional[Dict[str, str]] = None
) -> Session:
    """
    Создание и настройка сессии requests для HTTP-запросов.
    
    Args:
        user_agent: User-Agent для запросов
        timeout: Таймаут в секундах
        custom_headers: Дополнительные заголовки
        
    Returns:
        Session: Настроенная сессия requests
    """
    session = requests.Session()
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    
    if custom_headers:
        headers.update(custom_headers)
        
    session.headers.update(headers)
    session.timeout = timeout
    
    return session

def fetch_url(
    url: str,
    session: Optional[Session] = None,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    timeout: int = DEFAULT_TIMEOUT,
    allow_redirects: bool = True
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Получение содержимого URL с поддержкой повторных попыток.
    
    Args:
        url: URL для запроса
        session: Сессия requests для повторного использования
        retry_count: Количество повторных попыток при ошибке
        retry_delay: Задержка в секундах между повторными попытками
        timeout: Таймаут в секундах
        allow_redirects: Разрешать ли перенаправления
        
    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
            - HTML-контент (None при ошибке)
            - Заголовки ответа (None при ошибке)
            - Сообщение об ошибке или None при успехе
    """
    if session is None:
        session = create_session(timeout=timeout)
        
    error_msg = None
    
    for attempt in range(retry_count):
        try:
            response = session.get(
                url, 
                timeout=timeout,
                allow_redirects=allow_redirects
            )
            
            response.raise_for_status()
            return response.text, dict(response.headers), None
            
        except RequestException as e:
            error_msg = f"Attempt {attempt+1}/{retry_count} failed: {str(e)}"
            logger.warning(error_msg)
            
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
                
    return None, None, error_msg

async def fetch_url_with_javascript(
    url: str,
    headless: bool = True,
    wait_for_idle: int = 2000,  # мс
    wait_for_timeout: int = 30000,  # мс
    browser_type: str = "chromium",
    user_agent: str = DEFAULT_USER_AGENT
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Асинхронное получение контента URL с выполнением JavaScript через Playwright.
    
    Args:
        url: URL для запроса
        headless: Запускать ли браузер без интерфейса
        wait_for_idle: Время ожидания в мс после событий 'networkidle'
        wait_for_timeout: Максимальное время ожидания в мс
        browser_type: Тип браузера ('chromium', 'firefox', 'webkit')
        user_agent: User-Agent для запросов
        
    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
            - HTML-контент после выполнения JavaScript (None при ошибке)
            - Заголовки ответа (None при ошибке)
            - Сообщение об ошибке или None при успехе
    """
    try:
        async with async_playwright() as p:
            # Выбор браузера
            if browser_type == "firefox":
                browser_instance = p.firefox
            elif browser_type == "webkit":
                browser_instance = p.webkit
            else:
                browser_instance = p.chromium
                
            browser = await browser_instance.launch(headless=headless)
            
            try:
                context = await browser.new_context(
                    user_agent=user_agent,
                    viewport={'width': 1366, 'height': 768},
                )
                
                page = await context.new_page()
                
                response = await page.goto(url, wait_until="networkidle", timeout=wait_for_timeout)
                
                if not response:
                    return None, None, "No response received"
                    
                await page.wait_for_timeout(wait_for_idle)
                
                # Извлечение заголовков
                headers = {}
                for header in await response.all_headers():
                    headers[header[0]] = header[1]
                
                # Получение HTML после полной загрузки
                content = await page.content()
                
                return content, headers, None
                
            finally:
                await browser.close()
                
    except Exception as e:
        error_msg = f"JavaScript rendering error: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg

def fetch_url_with_javascript_sync(
    url: str,
    headless: bool = True,
    wait_for_idle: int = 2000,
    wait_for_timeout: int = 30000,
    browser_type: str = "chromium",
    user_agent: str = DEFAULT_USER_AGENT
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Синхронная обертка для получения контента URL с выполнением JavaScript.
    
    Args:
        url: URL для запроса
        headless: Запускать ли браузер без интерфейса
        wait_for_idle: Время ожидания в мс после событий 'networkidle'
        wait_for_timeout: Максимальное время ожидания в мс
        browser_type: Тип браузера ('chromium', 'firefox', 'webkit')
        user_agent: User-Agent для запросов
        
    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
            - HTML-контент после выполнения JavaScript (None при ошибке)
            - Заголовки ответа (None при ошибке)
            - Сообщение об ошибке или None при успехе
    """
    return asyncio.run(fetch_url_with_javascript(
        url, headless, wait_for_idle, wait_for_timeout, browser_type, user_agent
    ))
