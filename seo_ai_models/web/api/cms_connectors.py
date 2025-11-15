
"""
Модуль для интеграции с различными CMS (Content Management Systems).
Предоставляет коннекторы для WordPress, Drupal, Joomla и других CMS.
"""

import logging
import requests
from typing import Dict, List, Optional, Any, Union
import json
import os
import time
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class CMSConnector(ABC):
    """Базовый абстрактный класс для коннекторов CMS."""
    
    def __init__(self, 
                 site_url: str, 
                 api_key: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None,
                 verify_ssl: bool = True,
                 timeout: int = 30):
        """
        Инициализирует базовый коннектор CMS.
        
        Args:
            site_url: URL сайта
            api_key: Ключ API (если поддерживается)
            username: Имя пользователя (если требуется авторизация)
            password: Пароль (если требуется авторизация)
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
        """
        self.site_url = site_url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.password = password
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit_delay = 1  # Минимальная задержка между запросами (секунды)
        
    def _make_request(self, method: str, endpoint: str, 
                     params: Optional[Dict[str, Any]] = None, 
                     data: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     files: Optional[Dict[str, Any]] = None,
                     auth: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Выполняет HTTP-запрос с учетом ограничения частоты запросов.
        
        Args:
            method: HTTP-метод (GET, POST, PUT, DELETE)
            endpoint: Конечная точка API (относительный путь)
            params: Параметры запроса
            data: Данные запроса
            headers: Заголовки запроса
            files: Файлы для загрузки
            auth: Аутентификация (кортеж из имени пользователя и пароля)
            
        Returns:
            Dict[str, Any]: Ответ API
            
        Raises:
            requests.RequestException: В случае ошибки запроса
        """
        # Добавляем базовые заголовки
        headers = headers or {}
        headers.update({
            'User-Agent': 'SEO AI Models CMS Connector/1.0',
            'Accept': 'application/json'
        })
        
        # Используем аутентификацию, если указана
        if not auth and self.username and self.password:
            auth = (self.username, self.password)
        
        # Проверяем, не нарушаем ли ограничения частоты запросов
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        # Формируем URL
        url = f"{self.site_url}/{endpoint.lstrip('/')}"
        
        # Выполняем запрос
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data if method.upper() in ('POST', 'PUT', 'PATCH') and not files else None,
                data=data if method.upper() in ('POST', 'PUT', 'PATCH') and files else None,
                headers=headers,
                files=files,
                auth=auth,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            
            # Обновляем время последнего запроса
            self.last_request_time = time.time()
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            # Если ответ пустой или не в формате JSON, возвращаем словарь с кодом состояния
            if not response.text or not response.headers.get('Content-Type', '').startswith('application/json'):
                return {'status_code': response.status_code}
            
            # Возвращаем результат
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise
    
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в CMS.
        
        Returns:
            bool: True, если аутентификация успешна, иначе False
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def get_pages(self) -> List[Dict[str, Any]]:
        """
        Получает список страниц из CMS.
        
        Returns:
            List[Dict[str, Any]]: Список страниц
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Получает содержимое страницы из CMS.
        
        Args:
            page_id: ID страницы
            
        Returns:
            Dict[str, Any]: Содержимое страницы
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def update_page_content(self, page_id: str, content: str) -> bool:
        """
        Обновляет содержимое страницы в CMS.
        
        Args:
            page_id: ID страницы
            content: Новое содержимое
            
        Returns:
            bool: True, если обновление успешно, иначе False
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def get_cms_info(self) -> Dict[str, Any]:
        """
        Получает информацию о CMS.
        
        Returns:
            Dict[str, Any]: Информация о CMS
        """
        raise NotImplementedError("Subclasses must implement this method")


class WordPressConnector(CMSConnector):
    """Коннектор для WordPress с использованием WordPress REST API."""
    
    def __init__(self, 
                 site_url: str, 
                 api_key: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None,
                 verify_ssl: bool = True,
                 timeout: int = 30,
                 api_namespace: str = "wp/v2"):
        """
        Инициализирует коннектор WordPress.
        
        Args:
            site_url: URL сайта WordPress
            api_key: Ключ API (для методов аутентификации, отличных от Basic Auth)
            username: Имя пользователя WordPress
            password: Пароль пользователя WordPress
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
            api_namespace: Пространство имен API (по умолчанию "wp/v2")
        """
        super().__init__(site_url, api_key, username, password, verify_ssl, timeout)
        self.api_namespace = api_namespace
        self.api_base = f"wp-json/{api_namespace}"
        self.token = None
        self.token_expiry = None
        
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в WordPress с использованием JWT.
        
        Returns:
            bool: True, если аутентификация успешна, иначе False
        """
        # Проверяем наличие имени пользователя и пароля
        if not self.username or not self.password:
            # Если используется API-ключ, проверяем его
            if self.api_key:
                # Пробуем получить информацию о сайте
                try:
                    response = self._make_request(
                        method="GET",
                        endpoint="wp-json",
                        headers={
                            'Authorization': f'Bearer {self.api_key}'
                        }
                    )
                    if response.get('name'):
                        return True
                except requests.RequestException:
                    return False
            return False
        
        # Если токен уже получен и не истек, используем его
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return True
        
        # Пробуем использовать JWT-аутентификацию
        try:
            response = self._make_request(
                method="POST",
                endpoint="wp-json/jwt-auth/v1/token",
                data={
                    'username': self.username,
                    'password': self.password
                }
            )
            
            if 'token' in response:
                self.token = response['token']
                # Устанавливаем срок действия токена (обычно 24 часа)
                self.token_expiry = datetime.now() + timedelta(hours=24)
                return True
        except requests.RequestException:
            pass
        
        # Если JWT не работает, пробуем Basic Auth
        try:
            # Просто пробуем получить информацию о сайте
            self._make_request(
                method="GET",
                endpoint=self.api_base,
                auth=(self.username, self.password)
            )
            return True
        except requests.RequestException:
            return False
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Возвращает заголовки аутентификации для запросов.
        
        Returns:
            Dict[str, str]: Заголовки аутентификации
        """
        headers = {}
        
        # Если есть токен, используем его
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        # Если есть API-ключ, используем его
        elif self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def get_pages(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Получает список страниц из WordPress.
        
        Args:
            params: Дополнительные параметры запроса
            
        Returns:
            List[Dict[str, Any]]: Список страниц
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with WordPress")
        
        # Объединяем параметры по умолчанию с дополнительными
        params = params or {}
        default_params = {
            'per_page': 100,
            'status': 'publish'
        }
        params = {**default_params, **params}
        
        # Получаем страницы
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/pages",
                params=params,
                headers=self._get_auth_headers()
            )
            
            # Если ответ - список, возвращаем его
            if isinstance(response, list):
                return response
            
            # Иначе возвращаем пустой список
            return []
        except requests.RequestException as e:
            logger.error(f"Error getting WordPress pages: {str(e)}")
            return []
    
    def get_posts(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Получает список записей из WordPress.
        
        Args:
            params: Дополнительные параметры запроса
            
        Returns:
            List[Dict[str, Any]]: Список записей
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with WordPress")
        
        # Объединяем параметры по умолчанию с дополнительными
        params = params or {}
        default_params = {
            'per_page': 100,
            'status': 'publish'
        }
        params = {**default_params, **params}
        
        # Получаем записи
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/posts",
                params=params,
                headers=self._get_auth_headers()
            )
            
            # Если ответ - список, возвращаем его
            if isinstance(response, list):
                return response
            
            # Иначе возвращаем пустой список
            return []
        except requests.RequestException as e:
            logger.error(f"Error getting WordPress posts: {str(e)}")
            return []
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Получает содержимое страницы из WordPress.
        
        Args:
            page_id: ID страницы
            
        Returns:
            Dict[str, Any]: Содержимое страницы
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with WordPress")
        
        # Получаем содержимое страницы
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/pages/{page_id}",
                headers=self._get_auth_headers()
            )
            
            return response
        except requests.RequestException as e:
            logger.error(f"Error getting WordPress page content: {str(e)}")
            return {}
    
    def update_page_content(self, page_id: str, content: str, title: Optional[str] = None) -> bool:
        """
        Обновляет содержимое страницы в WordPress.
        
        Args:
            page_id: ID страницы
            content: Новое содержимое
            title: Новый заголовок (если требуется обновить)
            
        Returns:
            bool: True, если обновление успешно, иначе False
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with WordPress")
        
        # Формируем данные для обновления
        data = {'content': content}
        if title:
            data['title'] = title
        
        # Обновляем содержимое страницы
        try:
            response = self._make_request(
                method="POST",
                endpoint=f"{self.api_base}/pages/{page_id}",
                data=data,
                headers=self._get_auth_headers()
            )
            
            # Проверяем успешность обновления
            return 'id' in response and str(response['id']) == str(page_id)
        except requests.RequestException as e:
            logger.error(f"Error updating WordPress page content: {str(e)}")
            return False
    
    def get_cms_info(self) -> Dict[str, Any]:
        """
        Получает информацию о WordPress.
        
        Returns:
            Dict[str, Any]: Информация о WordPress
        """
        try:
            response = self._make_request(
                method="GET",
                endpoint="wp-json",
                headers=self._get_auth_headers()
            )
            
            return {
                'name': response.get('name', ''),
                'description': response.get('description', ''),
                'url': response.get('url', ''),
                'version': response.get('namespaces', []),
                'cms_type': 'wordpress'
            }
        except requests.RequestException as e:
            logger.error(f"Error getting WordPress info: {str(e)}")
            return {
                'cms_type': 'wordpress',
                'error': str(e)
            }


class DrupalConnector(CMSConnector):
    """Коннектор для Drupal с использованием Drupal JSON:API."""
    
    def __init__(self, 
                 site_url: str, 
                 api_key: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None,
                 verify_ssl: bool = True,
                 timeout: int = 30):
        """
        Инициализирует коннектор Drupal.
        
        Args:
            site_url: URL сайта Drupal
            api_key: Ключ API (для OAuth)
            username: Имя пользователя Drupal
            password: Пароль пользователя Drupal
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
        """
        super().__init__(site_url, api_key, username, password, verify_ssl, timeout)
        self.api_base = "jsonapi"
        self.token = None
        self.token_expiry = None
        self.csrf_token = None
        
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в Drupal.
        
        Returns:
            bool: True, если аутентификация успешна, иначе False
        """
        # Проверяем наличие имени пользователя и пароля
        if not self.username or not self.password:
            # Если используется API-ключ, проверяем его
            if self.api_key:
                # Пробуем получить информацию о сайте
                try:
                    response = self._make_request(
                        method="GET",
                        endpoint=self.api_base,
                        headers={
                            'Authorization': f'Bearer {self.api_key}'
                        }
                    )
                    if response.get('links'):
                        return True
                except requests.RequestException:
                    return False
            return False
        
        # Если токен уже получен и не истек, используем его
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return True
        
        # Получаем CSRF-токен
        try:
            response = self.session.get(
                f"{self.site_url}/session/token",
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            self.csrf_token = response.text
        except requests.RequestException as e:
            logger.error(f"Error getting CSRF token: {str(e)}")
            return False
        
        # Аутентифицируемся
        try:
            response = self._make_request(
                method="POST",
                endpoint="user/login",
                data={
                    'name': self.username,
                    'pass': self.password
                },
                headers={
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': self.csrf_token
                }
            )
            
            if 'current_user' in response:
                self.token = response.get('csrf_token')
                # Устанавливаем срок действия токена (обычно 24 часа)
                self.token_expiry = datetime.now() + timedelta(hours=24)
                return True
        except requests.RequestException:
            pass
        
        # Если стандартная аутентификация не работает, пробуем Basic Auth
        try:
            # Просто пробуем получить информацию о сайте
            self._make_request(
                method="GET",
                endpoint=self.api_base,
                auth=(self.username, self.password)
            )
            return True
        except requests.RequestException:
            return False
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Возвращает заголовки аутентификации для запросов.
        
        Returns:
            Dict[str, str]: Заголовки аутентификации
        """
        headers = {}
        
        # Если есть CSRF-токен, используем его
        if self.csrf_token:
            headers['X-CSRF-Token'] = self.csrf_token
        
        # Если есть токен, используем его
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        # Если есть API-ключ, используем его
        elif self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def get_pages(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Получает список страниц из Drupal.
        
        Args:
            params: Дополнительные параметры запроса
            
        Returns:
            List[Dict[str, Any]]: Список страниц
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Drupal")
        
        # Объединяем параметры по умолчанию с дополнительными
        params = params or {}
        default_params = {
            'page[limit]': 50
        }
        params = {**default_params, **params}
        
        # Получаем страницы
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/node/page",
                params=params,
                headers=self._get_auth_headers()
            )
            
            # Если ответ содержит данные, возвращаем их
            if 'data' in response and isinstance(response['data'], list):
                return response['data']
            
            # Иначе возвращаем пустой список
            return []
        except requests.RequestException as e:
            logger.error(f"Error getting Drupal pages: {str(e)}")
            return []
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Получает содержимое страницы из Drupal.
        
        Args:
            page_id: ID страницы (UUID)
            
        Returns:
            Dict[str, Any]: Содержимое страницы
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Drupal")
        
        # Получаем содержимое страницы
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/node/page/{page_id}",
                headers=self._get_auth_headers()
            )
            
            # Если ответ содержит данные, возвращаем их
            if 'data' in response:
                return response['data']
            
            # Иначе возвращаем пустой словарь
            return {}
        except requests.RequestException as e:
            logger.error(f"Error getting Drupal page content: {str(e)}")
            return {}
    
    def update_page_content(self, page_id: str, content: str, title: Optional[str] = None) -> bool:
        """
        Обновляет содержимое страницы в Drupal.
        
        Args:
            page_id: ID страницы (UUID)
            content: Новое содержимое
            title: Новый заголовок (если требуется обновить)
            
        Returns:
            bool: True, если обновление успешно, иначе False
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Drupal")
        
        # Получаем текущее состояние страницы
        page = self.get_page_content(page_id)
        if not page:
            logger.error(f"Page with ID {page_id} not found")
            return False
        
        # Формируем данные для обновления
        data = {
            'data': {
                'type': 'node--page',
                'id': page_id,
                'attributes': {}
            }
        }
        
        # Если указан контент, обновляем его
        if content:
            data['data']['attributes']['body'] = {
                'value': content,
                'format': 'full_html'
            }
        
        # Если указан заголовок, обновляем его
        if title:
            data['data']['attributes']['title'] = title
        
        # Обновляем содержимое страницы
        try:
            response = self._make_request(
                method="PATCH",
                endpoint=f"{self.api_base}/node/page/{page_id}",
                data=data,
                headers={
                    **self._get_auth_headers(),
                    'Content-Type': 'application/vnd.api+json'
                }
            )
            
            # Проверяем успешность обновления
            return 'data' in response and response['data']['id'] == page_id
        except requests.RequestException as e:
            logger.error(f"Error updating Drupal page content: {str(e)}")
            return False
    
    def get_cms_info(self) -> Dict[str, Any]:
        """
        Получает информацию о Drupal.
        
        Returns:
            Dict[str, Any]: Информация о Drupal
        """
        try:
            response = self._make_request(
                method="GET",
                endpoint=self.api_base,
                headers=self._get_auth_headers()
            )
            
            return {
                'version': response.get('meta', {}).get('links', {}).get('meta', {}).get('drupal', {}).get('version', ''),
                'cms_type': 'drupal',
                'links': response.get('links', {})
            }
        except requests.RequestException as e:
            logger.error(f"Error getting Drupal info: {str(e)}")
            return {
                'cms_type': 'drupal',
                'error': str(e)
            }


class JoomlaConnector(CMSConnector):
    """Коннектор для Joomla с использованием Joomla Web Services API."""
    
    def __init__(self, 
                 site_url: str, 
                 api_key: str,
                 verify_ssl: bool = True,
                 timeout: int = 30):
        """
        Инициализирует коннектор Joomla.
        
        Args:
            site_url: URL сайта Joomla
            api_key: Ключ API (обязательный для Joomla Web Services)
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
        """
        super().__init__(site_url, api_key, None, None, verify_ssl, timeout)
        self.api_base = "api/index.php/v1"
        
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в Joomla.
        
        Returns:
            bool: True, если аутентификация успешна, иначе False
        """
        # Проверяем наличие API-ключа
        if not self.api_key:
            return False
        
        # Пробуем получить информацию о сайте
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/config",
                headers={
                    'X-Joomla-Token': self.api_key
                }
            )
            
            return response.get('data') is not None
        except requests.RequestException:
            return False
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Возвращает заголовки аутентификации для запросов.
        
        Returns:
            Dict[str, str]: Заголовки аутентификации
        """
        headers = {}
        
        # Если есть API-ключ, используем его
        if self.api_key:
            headers['X-Joomla-Token'] = self.api_key
        
        return headers
    
    def get_pages(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Получает список страниц (статей) из Joomla.
        
        Args:
            params: Дополнительные параметры запроса
            
        Returns:
            List[Dict[str, Any]]: Список страниц
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Joomla")
        
        # Объединяем параметры по умолчанию с дополнительными
        params = params or {}
        default_params = {
            'limit': 100,
            'state': 1  # Опубликованные статьи
        }
        params = {**default_params, **params}
        
        # Получаем статьи
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/content/articles",
                params=params,
                headers=self._get_auth_headers()
            )
            
            # Если ответ содержит данные, возвращаем их
            if 'data' in response and isinstance(response['data'], list):
                return response['data']
            
            # Иначе возвращаем пустой список
            return []
        except requests.RequestException as e:
            logger.error(f"Error getting Joomla articles: {str(e)}")
            return []
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Получает содержимое страницы (статьи) из Joomla.
        
        Args:
            page_id: ID страницы
            
        Returns:
            Dict[str, Any]: Содержимое страницы
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Joomla")
        
        # Получаем содержимое статьи
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/content/articles/{page_id}",
                headers=self._get_auth_headers()
            )
            
            # Если ответ содержит данные, возвращаем их
            if 'data' in response:
                return response['data']
            
            # Иначе возвращаем пустой словарь
            return {}
        except requests.RequestException as e:
            logger.error(f"Error getting Joomla article content: {str(e)}")
            return {}
    
    def update_page_content(self, page_id: str, content: str, title: Optional[str] = None) -> bool:
        """
        Обновляет содержимое страницы (статьи) в Joomla.
        
        Args:
            page_id: ID страницы
            content: Новое содержимое
            title: Новый заголовок (если требуется обновить)
            
        Returns:
            bool: True, если обновление успешно, иначе False
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Joomla")
        
        # Формируем данные для обновления
        data = {}
        
        # Если указан контент, обновляем его
        if content:
            data['articletext'] = content
        
        # Если указан заголовок, обновляем его
        if title:
            data['title'] = title
        
        # Обновляем статью
        try:
            response = self._make_request(
                method="PATCH",
                endpoint=f"{self.api_base}/content/articles/{page_id}",
                data=data,
                headers=self._get_auth_headers()
            )
            
            # Проверяем успешность обновления
            return 'data' in response and str(response['data']['id']) == str(page_id)
        except requests.RequestException as e:
            logger.error(f"Error updating Joomla article: {str(e)}")
            return False
    
    def get_cms_info(self) -> Dict[str, Any]:
        """
        Получает информацию о Joomla.
        
        Returns:
            Dict[str, Any]: Информация о Joomla
        """
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"{self.api_base}/config",
                headers=self._get_auth_headers()
            )
            
            config = response.get('data', {})
            
            return {
                'name': config.get('sitename', ''),
                'description': config.get('MetaDesc', ''),
                'version': config.get('version', ''),
                'cms_type': 'joomla'
            }
        except requests.RequestException as e:
            logger.error(f"Error getting Joomla info: {str(e)}")
            return {
                'cms_type': 'joomla',
                'error': str(e)
            }


class BitrixConnector(CMSConnector):
    """Коннектор для 1С-Битрикс с использованием REST API."""
    
    def __init__(self, 
                 site_url: str, 
                 api_key: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None,
                 verify_ssl: bool = True,
                 timeout: int = 30):
        """
        Инициализирует коннектор 1С-Битрикс.
        
        Args:
            site_url: URL сайта 1С-Битрикс
            api_key: Ключ API (для REST API)
            username: Имя пользователя (для авторизации)
            password: Пароль (для авторизации)
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
        """
        super().__init__(site_url, api_key, username, password, verify_ssl, timeout)
        self.api_base = "rest"
        self.sessid = None
        
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в 1С-Битрикс.
        
        Returns:
            bool: True, если аутентификация успешна, иначе False
        """
        # Если используется API-ключ, просто проверяем его
        if self.api_key:
            try:
                response = self._make_request(
                    method="POST",
                    endpoint=f"{self.api_base}/user.current",
                    data={
                        'auth': self.api_key
                    }
                )
                
                return response.get('result') is not None
            except requests.RequestException:
                pass
        
        # Если используются логин и пароль, пробуем аутентифицироваться
        if self.username and self.password:
            try:
                # Получаем SESSID для авторизации
                login_page = self.session.get(
                    f"{self.site_url}/auth/",
                    verify=self.verify_ssl,
                    timeout=self.timeout
                )
                
                # Извлекаем SESSID из HTML-страницы (упрощенный вариант)
                import re
                sessid_match = re.search(r'name="sessid" value="([^"]+)"', login_page.text)
                if sessid_match:
                    self.sessid = sessid_match.group(1)
                
                # Выполняем авторизацию
                login_response = self.session.post(
                    f"{self.site_url}/auth/",
                    data={
                        'AUTH_FORM': 'Y',
                        'TYPE': 'AUTH',
                        'USER_LOGIN': self.username,
                        'USER_PASSWORD': self.password,
                        'sessid': self.sessid
                    },
                    verify=self.verify_ssl,
                    timeout=self.timeout
                )
                
                # Проверяем успешность авторизации
                return 'Вы зарегистрированы и успешно авторизовались' in login_response.text
            except requests.RequestException:
                pass
        
        return False
        
    def _get_auth_params(self) -> Dict[str, str]:
        """
        Возвращает параметры аутентификации для запросов.
        
        Returns:
            Dict[str, str]: Параметры аутентификации
        """
        params = {}
        
        # Если есть API-ключ, используем его
        if self.api_key:
            params['auth'] = self.api_key
        
        # Если есть SESSID, используем его
        if self.sessid:
            params['sessid'] = self.sessid
        
        return params
    
    def get_pages(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Получает список страниц из 1С-Битрикс.
        
        Args:
            params: Дополнительные параметры запроса
            
        Returns:
            List[Dict[str, Any]]: Список страниц
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with 1C-Bitrix")
        
        # Объединяем параметры по умолчанию с дополнительными
        params = params or {}
        default_params = {
            'iblock_type': 'content',
            'iblock_id': 1,  # ID инфоблока со страницами (может отличаться)
            'active': 'Y'
        }
        api_params = {**default_params, **params, **self._get_auth_params()}
        
        # Получаем страницы
        try:
            response = self._make_request(
                method="POST",
                endpoint=f"{self.api_base}/iblock.element.get",
                data=api_params
            )
            
            # Если ответ содержит результаты, возвращаем их
            if 'result' in response and isinstance(response['result'], list):
                return response['result']
            
            # Иначе возвращаем пустой список
            return []
        except requests.RequestException as e:
            logger.error(f"Error getting 1C-Bitrix pages: {str(e)}")
            return []
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Получает содержимое страницы из 1С-Битрикс.
        
        Args:
            page_id: ID страницы
            
        Returns:
            Dict[str, Any]: Содержимое страницы
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with 1C-Bitrix")
        
        # Получаем содержимое страницы
        try:
            response = self._make_request(
                method="POST",
                endpoint=f"{self.api_base}/iblock.element.get",
                data={
                    'id': page_id,
                    **self._get_auth_params()
                }
            )
            
            # Если ответ содержит результаты, возвращаем первый элемент
            if 'result' in response and isinstance(response['result'], list) and response['result']:
                return response['result'][0]
            
            # Иначе возвращаем пустой словарь
            return {}
        except requests.RequestException as e:
            logger.error(f"Error getting 1C-Bitrix page content: {str(e)}")
            return {}
    
    def update_page_content(self, page_id: str, content: str, title: Optional[str] = None) -> bool:
        """
        Обновляет содержимое страницы в 1С-Битрикс.
        
        Args:
            page_id: ID страницы
            content: Новое содержимое
            title: Новый заголовок (если требуется обновить)
            
        Returns:
            bool: True, если обновление успешно, иначе False
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with 1C-Bitrix")
        
        # Формируем данные для обновления
        fields = {}
        
        # Если указан контент, обновляем его
        if content:
            fields['DETAIL_TEXT'] = content
            fields['DETAIL_TEXT_TYPE'] = 'html'
        
        # Если указан заголовок, обновляем его
        if title:
            fields['NAME'] = title
        
        # Обновляем страницу
        try:
            response = self._make_request(
                method="POST",
                endpoint=f"{self.api_base}/iblock.element.update",
                data={
                    'id': page_id,
                    'fields': fields,
                    **self._get_auth_params()
                }
            )
            
            # Проверяем успешность обновления
            return 'result' in response and response['result']
        except requests.RequestException as e:
            logger.error(f"Error updating 1C-Bitrix page: {str(e)}")
            return False
    
    def get_cms_info(self) -> Dict[str, Any]:
        """
        Получает информацию о 1С-Битрикс.
        
        Returns:
            Dict[str, Any]: Информация о 1С-Битрикс
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            return {
                'cms_type': '1c-bitrix',
                'error': 'Failed to authenticate'
            }
        
        try:
            response = self._make_request(
                method="POST",
                endpoint=f"{self.api_base}/main.version",
                data=self._get_auth_params()
            )
            
            return {
                'version': response.get('result', ''),
                'cms_type': '1c-bitrix'
            }
        except requests.RequestException as e:
            logger.error(f"Error getting 1C-Bitrix info: {str(e)}")
            return {
                'cms_type': '1c-bitrix',
                'error': str(e)
            }


# Фабрика для создания коннекторов CMS
class CMSConnectorFactory:
    """Фабрика для создания коннекторов CMS."""
    
    @staticmethod
    def create_connector(cms_type: str, site_url: str, **kwargs) -> CMSConnector:
        """
        Создает коннектор для указанной CMS.
        
        Args:
            cms_type: Тип CMS ('wordpress', 'drupal', 'joomla', 'bitrix')
            site_url: URL сайта
            **kwargs: Дополнительные параметры для коннектора
            
        Returns:
            CMSConnector: Коннектор для указанной CMS
            
        Raises:
            ValueError: Если указан неизвестный тип CMS
        """
        cms_type = cms_type.lower()
        
        if cms_type == 'wordpress':
            return WordPressConnector(site_url, **kwargs)
        elif cms_type == 'drupal':
            return DrupalConnector(site_url, **kwargs)
        elif cms_type == 'joomla':
            return JoomlaConnector(site_url, **kwargs)
        elif cms_type == 'bitrix' or cms_type == '1c-bitrix':
            return BitrixConnector(site_url, **kwargs)
        else:
            raise ValueError(f"Unknown CMS type: {cms_type}")
    
    @staticmethod
    def detect_cms(site_url: str) -> str:
        """
        Определяет тип CMS по URL сайта.
        
        Args:
            site_url: URL сайта
            
        Returns:
            str: Тип CMS ('wordpress', 'drupal', 'joomla', 'bitrix', 'unknown')
        """
        try:
            # Получаем главную страницу сайта
            response = requests.get(
                site_url,
                headers={
                    'User-Agent': 'SEO AI Models CMS Detector/1.0'
                },
                timeout=10
            )
            response.raise_for_status()
            
            html = response.text.lower()
            
            # Проверяем признаки WordPress
            if 'wp-content' in html or 'wp-includes' in html or 'wordpress' in html:
                return 'wordpress'
            
            # Проверяем признаки Drupal
            if 'drupal' in html or '/sites/default/' in html or 'drupal.settings' in html:
                return 'drupal'
            
            # Проверяем признаки Joomla
            if 'joomla' in html or '/templates/' in html or 'option=com_' in html:
                return 'joomla'
            
            # Проверяем признаки 1С-Битрикс
            if 'bitrix/js' in html or 'бизнес-процессы, устанавливаемая на Интранет-портал Битрикс24' in html:
                return 'bitrix'
            
            # Проверяем заголовки ответа
            for header, value in response.headers.items():
                header_lower = header.lower()
                value_lower = value.lower()
                
                if header_lower == 'x-powered-by':
                    if 'wordpress' in value_lower:
                        return 'wordpress'
                    elif 'drupal' in value_lower:
                        return 'drupal'
                    elif 'joomla' in value_lower:
                        return 'joomla'
                    elif 'bitrix' in value_lower:
                        return 'bitrix'
            
            # Пробуем определить CMS по наличию характерных URL
            for cms_type, paths in [
                ('wordpress', ['/wp-admin/', '/wp-login.php', '/wp-content/']),
                ('drupal', ['/user/login', '/admin/content', '/sites/default/']),
                ('joomla', ['/administrator/', '/components/', '/templates/']),
                ('bitrix', ['/bitrix/admin/', '/bitrix/js/', '/bitrix/templates/'])
            ]:
                for path in paths:
                    try:
                        check_url = site_url.rstrip('/') + path
                        check_response = requests.head(
                            check_url,
                            headers={
                                'User-Agent': 'SEO AI Models CMS Detector/1.0'
                            },
                            timeout=5
                        )
                        
                        if check_response.status_code < 400:
                            return cms_type
                    except requests.RequestException:
                        continue
            
            return 'unknown'
        except requests.RequestException:
            return 'unknown'
