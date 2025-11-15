
"""
Модуль для интеграции с аналитическими системами (Google Analytics, Яндекс.Метрика).
Предоставляет функциональность для получения и анализа данных о посещаемости сайта.
"""

import logging
import requests
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import os
from abc import ABC, abstractmethod
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

class AnalyticsConnector(ABC):
    """Базовый абстрактный класс для коннекторов аналитических систем."""
    
    def __init__(self, 
                 auth_config: Dict[str, Any],
                 verify_ssl: bool = True,
                 timeout: int = 60):
        """
        Инициализирует базовый коннектор аналитических систем.
        
        Args:
            auth_config: Конфигурация аутентификации
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
        """
        self.auth_config = auth_config
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.token = None
        self.token_expiry = None
        self.last_request_time = 0
        self.rate_limit_delay = 1  # Минимальная задержка между запросами (секунды)
        
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в аналитической системе.

        Returns:
            bool: True, если аутентификация успешна, иначе False

        Raises:
            NotImplementedError: Должен быть реализован в подклассе
        """
        raise NotImplementedError("Subclasses must implement authenticate() method")
    
    @abstractmethod
    def get_visits(self,
                  start_date: str,
                  end_date: str,
                  dimensions: Optional[List[str]] = None,
                  metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Получает данные о посещаемости сайта.

        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            dimensions: Измерения (группировки) для данных
            metrics: Метрики для данных

        Returns:
            Dict[str, Any]: Данные о посещаемости

        Raises:
            NotImplementedError: Должен быть реализован в подклассе
        """
        raise NotImplementedError("Subclasses must implement get_visits() method")
    
    @abstractmethod
    def get_sources(self,
                   start_date: str,
                   end_date: str) -> Dict[str, Any]:
        """
        Получает данные об источниках трафика.

        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'

        Returns:
            Dict[str, Any]: Данные об источниках трафика

        Raises:
            NotImplementedError: Должен быть реализован в подклассе
        """
        raise NotImplementedError("Subclasses must implement get_sources() method")
    
    @abstractmethod
    def get_page_stats(self,
                      page_path: str,
                      start_date: str,
                      end_date: str) -> Dict[str, Any]:
        """
        Получает статистику по конкретной странице.

        Args:
            page_path: Путь к странице
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'

        Returns:
            Dict[str, Any]: Статистика по странице

        Raises:
            NotImplementedError: Должен быть реализован в подклассе
        """
        raise NotImplementedError("Subclasses must implement get_page_stats() method")
    
    def generate_chart(self, 
                      data: pd.DataFrame, 
                      x: str, 
                      y: List[str],
                      title: str = '',
                      chart_type: str = 'line',
                      figsize: tuple = (10, 6)) -> str:
        """
        Генерирует диаграмму на основе данных.
        
        Args:
            data: Данные для диаграммы (pandas DataFrame)
            x: Название столбца для оси X
            y: Список названий столбцов для оси Y
            title: Заголовок диаграммы
            chart_type: Тип диаграммы ('line', 'bar', 'pie')
            figsize: Размер диаграммы (ширина, высота)
            
        Returns:
            str: Data URL с диаграммой в формате PNG (base64)
        """
        plt.figure(figsize=figsize)
        
        if chart_type == 'line':
            for col in y:
                plt.plot(data[x], data[col], label=col)
            plt.legend()
        elif chart_type == 'bar':
            if len(y) == 1:
                plt.bar(data[x], data[y[0]])
            else:
                data_subset = data[[x] + y].set_index(x)
                data_subset.plot(kind='bar', ax=plt.gca())
        elif chart_type == 'pie' and len(y) == 1:
            plt.pie(data[y[0]], labels=data[x], autopct='%1.1f%%')
        
        plt.title(title)
        plt.tight_layout()
        
        # Сохраняем диаграмму в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Кодируем диаграмму в base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"


class GoogleAnalyticsConnector(AnalyticsConnector):
    """Коннектор для Google Analytics 4 с использованием Google Analytics Data API."""
    
    def __init__(self, auth_config: Dict[str, Any], property_id: str,
                verify_ssl: bool = True, timeout: int = 60):
        """
        Инициализирует коннектор Google Analytics.
        
        Args:
            auth_config: Конфигурация аутентификации (ключ сервисного аккаунта или токен OAuth)
            property_id: ID свойства Google Analytics (GA4)
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
        """
        super().__init__(auth_config, verify_ssl, timeout)
        self.property_id = property_id
        self.api_base = "https://analyticsdata.googleapis.com/v1beta"
        
        # Проверяем наличие необходимых библиотек
        try:
            import google.oauth2.service_account
            import google.auth.transport.requests
            self.google_libs_available = True
        except ImportError:
            self.google_libs_available = False
            logger.warning("Google OAuth2 libraries not available. Some functionality may be limited.")
        
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в Google Analytics.
        
        Returns:
            bool: True, если аутентификация успешна, иначе False
        """
        # Если токен уже получен и не истек, используем его
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return True
        
        # Проверяем наличие необходимых библиотек
        if not self.google_libs_available:
            logger.error("Google OAuth2 libraries not available. Unable to authenticate.")
            return False
        
        try:
            # Импортируем необходимые библиотеки
            from google.oauth2.service_account import Credentials
            from google.auth.transport.requests import Request
            
            # Проверяем тип аутентификации
            if 'service_account_file' in self.auth_config:
                # Аутентификация с использованием ключа сервисного аккаунта
                credentials = Credentials.from_service_account_file(
                    self.auth_config['service_account_file'],
                    scopes=['https://www.googleapis.com/auth/analytics.readonly']
                )
            elif 'service_account_info' in self.auth_config:
                # Аутентификация с использованием JSON сервисного аккаунта
                credentials = Credentials.from_service_account_info(
                    self.auth_config['service_account_info'],
                    scopes=['https://www.googleapis.com/auth/analytics.readonly']
                )
            else:
                logger.error("Invalid authentication configuration for Google Analytics")
                return False
            
            # Обновляем токен, если необходимо
            if credentials.expired:
                credentials.refresh(Request())
            
            # Сохраняем токен и срок его действия
            self.token = credentials.token
            self.token_expiry = credentials.expiry
            
            return True
        except Exception as e:
            logger.error(f"Error authenticating with Google Analytics: {str(e)}")
            return False
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет запрос к Google Analytics API.
        
        Args:
            endpoint: Конечная точка API
            data: Данные запроса
            
        Returns:
            Dict[str, Any]: Ответ API
            
        Raises:
            Exception: В случае ошибки запроса
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Google Analytics")
        
        # Проверяем, не нарушаем ли ограничения частоты запросов
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        # Формируем URL
        url = f"{self.api_base}/{endpoint}"
        
        # Формируем заголовки
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Выполняем запрос
        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            
            # Обновляем время последнего запроса
            self.last_request_time = time.time()
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            # Возвращаем результат
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error making request to Google Analytics API: {str(e)}")
            raise
    
    def get_visits(self, 
                  start_date: str, 
                  end_date: str, 
                  dimensions: Optional[List[str]] = None,
                  metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Получает данные о посещаемости сайта из Google Analytics.
        
        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            dimensions: Измерения (группировки) для данных
            metrics: Метрики для данных
            
        Returns:
            Dict[str, Any]: Данные о посещаемости
        """
        # Устанавливаем значения по умолчанию
        dimensions = dimensions or ['date']
        metrics = metrics or ['activeUsers', 'sessions', 'screenPageViews']
        
        # Формируем запрос
        request_data = {
            'property': f'properties/{self.property_id}',
            'dateRanges': [
                {
                    'startDate': start_date,
                    'endDate': end_date
                }
            ],
            'dimensions': [{'name': dim} for dim in dimensions],
            'metrics': [{'name': metric} for metric in metrics]
        }
        
        try:
            # Выполняем запрос
            response = self._make_request('properties/{self.property_id}:runReport', request_data)
            
            # Обрабатываем ответ
            result = {
                'dimensions': dimensions,
                'metrics': metrics,
                'rows': []
            }
            
            # Проверяем наличие данных
            if 'rows' in response:
                for row in response['rows']:
                    dimension_values = [dim_value['value'] for dim_value in row['dimensionValues']]
                    metric_values = [float(metric_value['value']) for metric_value in row['metricValues']]
                    
                    result_row = {}
                    for i, dim in enumerate(dimensions):
                        result_row[dim] = dimension_values[i]
                    for i, metric in enumerate(metrics):
                        result_row[metric] = metric_values[i]
                    
                    result['rows'].append(result_row)
            
            return result
        except Exception as e:
            logger.error(f"Error getting visits from Google Analytics: {str(e)}")
            return {
                'dimensions': dimensions,
                'metrics': metrics,
                'rows': [],
                'error': str(e)
            }
    
    def get_sources(self,
                   start_date: str,
                   end_date: str) -> Dict[str, Any]:
        """
        Получает данные об источниках трафика из Google Analytics.
        
        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            Dict[str, Any]: Данные об источниках трафика
        """
        # Формируем запрос для получения источников трафика
        dimensions = ['sessionSource', 'sessionMedium']
        metrics = ['sessions', 'activeUsers', 'screenPageViews', 'conversions']
        
        # Получаем данные
        return self.get_visits(
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
            metrics=metrics
        )
    
    def get_page_stats(self,
                      page_path: str,
                      start_date: str,
                      end_date: str) -> Dict[str, Any]:
        """
        Получает статистику по конкретной странице из Google Analytics.
        
        Args:
            page_path: Путь к странице
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            Dict[str, Any]: Статистика по странице
        """
        # Формируем запрос
        request_data = {
            'property': f'properties/{self.property_id}',
            'dateRanges': [
                {
                    'startDate': start_date,
                    'endDate': end_date
                }
            ],
            'dimensions': [{'name': 'date'}],
            'metrics': [
                {'name': 'screenPageViews'},
                {'name': 'activeUsers'},
                {'name': 'userEngagementDuration'},
                {'name': 'conversions'}
            ],
            'dimensionFilter': {
                'filter': {
                    'fieldName': 'pagePath',
                    'stringFilter': {
                        'matchType': 'EXACT',
                        'value': page_path
                    }
                }
            }
        }
        
        try:
            # Выполняем запрос
            response = self._make_request('properties/{self.property_id}:runReport', request_data)
            
            # Обрабатываем ответ
            result = {
                'page_path': page_path,
                'start_date': start_date,
                'end_date': end_date,
                'rows': []
            }
            
            # Проверяем наличие данных
            if 'rows' in response:
                for row in response['rows']:
                    date = row['dimensionValues'][0]['value']
                    views = float(row['metricValues'][0]['value'])
                    users = float(row['metricValues'][1]['value'])
                    engagement_duration = float(row['metricValues'][2]['value'])
                    conversions = float(row['metricValues'][3]['value'])
                    
                    result['rows'].append({
                        'date': date,
                        'views': views,
                        'users': users,
                        'engagement_duration': engagement_duration,
                        'conversions': conversions
                    })
            
            return result
        except Exception as e:
            logger.error(f"Error getting page stats from Google Analytics: {str(e)}")
            return {
                'page_path': page_path,
                'start_date': start_date,
                'end_date': end_date,
                'rows': [],
                'error': str(e)
            }


class YandexMetrikaConnector(AnalyticsConnector):
    """Коннектор для Яндекс.Метрики с использованием Яндекс.Метрика API."""
    
    def __init__(self, auth_config: Dict[str, Any], counter_id: str,
                verify_ssl: bool = True, timeout: int = 60):
        """
        Инициализирует коннектор Яндекс.Метрики.
        
        Args:
            auth_config: Конфигурация аутентификации (OAuth-токен)
            counter_id: ID счетчика Яндекс.Метрики
            verify_ssl: Проверять SSL-сертификат
            timeout: Таймаут запросов (секунды)
        """
        super().__init__(auth_config, verify_ssl, timeout)
        self.counter_id = counter_id
        self.api_base = "https://api-metrika.yandex.net/stat/v1/data"
        
    def authenticate(self) -> bool:
        """
        Аутентифицирует коннектор в Яндекс.Метрике.
        
        Returns:
            bool: True, если аутентификация успешна, иначе False
        """
        # Если есть токен, считаем аутентификацию успешной
        if 'oauth_token' in self.auth_config:
            self.token = self.auth_config['oauth_token']
            return True
        
        # Если токена нет, аутентификация неуспешна
        return False
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет запрос к API Яндекс.Метрики.
        
        Args:
            params: Параметры запроса
            
        Returns:
            Dict[str, Any]: Ответ API
            
        Raises:
            Exception: В случае ошибки запроса
        """
        # Аутентифицируемся, если еще не сделали этого
        if not self.authenticate():
            raise Exception("Failed to authenticate with Yandex.Metrika")
        
        # Проверяем, не нарушаем ли ограничения частоты запросов
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        # Добавляем ID счетчика к параметрам
        params['id'] = self.counter_id
        
        # Формируем заголовки
        headers = {
            'Authorization': f'OAuth {self.token}'
        }
        
        # Выполняем запрос
        try:
            response = requests.get(
                self.api_base,
                params=params,
                headers=headers,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            
            # Обновляем время последнего запроса
            self.last_request_time = time.time()
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            # Возвращаем результат
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error making request to Yandex.Metrika API: {str(e)}")
            raise
    
    def get_visits(self, 
                  start_date: str, 
                  end_date: str, 
                  dimensions: Optional[List[str]] = None,
                  metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Получает данные о посещаемости сайта из Яндекс.Метрики.
        
        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            dimensions: Измерения (группировки) для данных
            metrics: Метрики для данных
            
        Returns:
            Dict[str, Any]: Данные о посещаемости
        """
        # Устанавливаем значения по умолчанию
        dimensions = dimensions or ['ym:s:date']
        metrics = metrics or ['ym:s:users', 'ym:s:visits', 'ym:s:pageviews']
        
        # Формируем параметры запроса
        params = {
            'date1': start_date,
            'date2': end_date,
            'dimensions': ','.join(dimensions),
            'metrics': ','.join(metrics),
            'limit': 10000
        }
        
        try:
            # Выполняем запрос
            response = self._make_request(params)
            
            # Обрабатываем ответ
            result = {
                'dimensions': [dim['name'] for dim in response['query']['dimensions']],
                'metrics': [metric['name'] for metric in response['query']['metrics']],
                'rows': []
            }
            
            # Проверяем наличие данных
            if 'data' in response:
                for item in response['data']:
                    row = {}
                    
                    # Добавляем значения измерений
                    for i, dim in enumerate(result['dimensions']):
                        row[dim] = item['dimensions'][i]['name']
                    
                    # Добавляем значения метрик
                    for i, metric in enumerate(result['metrics']):
                        row[metric] = item['metrics'][i]
                    
                    result['rows'].append(row)
            
            return result
        except Exception as e:
            logger.error(f"Error getting visits from Yandex.Metrika: {str(e)}")
            return {
                'dimensions': dimensions,
                'metrics': metrics,
                'rows': [],
                'error': str(e)
            }
    
    def get_sources(self,
                   start_date: str,
                   end_date: str) -> Dict[str, Any]:
        """
        Получает данные об источниках трафика из Яндекс.Метрики.
        
        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            Dict[str, Any]: Данные об источниках трафика
        """
        # Формируем запрос для получения источников трафика
        dimensions = ['ym:s:trafficSource', 'ym:s:sourceEngine']
        metrics = ['ym:s:visits', 'ym:s:users', 'ym:s:pageviews', 'ym:s:bounceRate', 'ym:s:pageDepth']
        
        # Получаем данные
        return self.get_visits(
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
            metrics=metrics
        )
    
    def get_page_stats(self,
                      page_path: str,
                      start_date: str,
                      end_date: str) -> Dict[str, Any]:
        """
        Получает статистику по конкретной странице из Яндекс.Метрики.
        
        Args:
            page_path: Путь к странице
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            Dict[str, Any]: Статистика по странице
        """
        # Формируем параметры запроса
        params = {
            'date1': start_date,
            'date2': end_date,
            'dimensions': 'ym:s:date',
            'metrics': 'ym:s:pageviews,ym:s:users,ym:s:avgVisitDurationSeconds,ym:s:bounceRate',
            'filters': f"ym:s:URL='{page_path}'",
            'limit': 10000
        }
        
        try:
            # Выполняем запрос
            response = self._make_request(params)
            
            # Обрабатываем ответ
            result = {
                'page_path': page_path,
                'start_date': start_date,
                'end_date': end_date,
                'rows': []
            }
            
            # Проверяем наличие данных
            if 'data' in response:
                for item in response['data']:
                    date = item['dimensions'][0]['name']
                    views = item['metrics'][0]
                    users = item['metrics'][1]
                    avg_duration = item['metrics'][2]
                    bounce_rate = item['metrics'][3]
                    
                    result['rows'].append({
                        'date': date,
                        'views': views,
                        'users': users,
                        'avg_duration': avg_duration,
                        'bounce_rate': bounce_rate
                    })
            
            return result
        except Exception as e:
            logger.error(f"Error getting page stats from Yandex.Metrika: {str(e)}")
            return {
                'page_path': page_path,
                'start_date': start_date,
                'end_date': end_date,
                'rows': [],
                'error': str(e)
            }


# Фабрика для создания коннекторов аналитических систем
class AnalyticsConnectorFactory:
    """Фабрика для создания коннекторов аналитических систем."""
    
    @staticmethod
    def create_connector(analytics_type: str, auth_config: Dict[str, Any], **kwargs) -> AnalyticsConnector:
        """
        Создает коннектор для указанной аналитической системы.
        
        Args:
            analytics_type: Тип аналитической системы ('google_analytics', 'yandex_metrika')
            auth_config: Конфигурация аутентификации
            **kwargs: Дополнительные параметры для коннектора
            
        Returns:
            AnalyticsConnector: Коннектор для указанной аналитической системы
            
        Raises:
            ValueError: Если указан неизвестный тип аналитической системы
        """
        analytics_type = analytics_type.lower()
        
        if analytics_type == 'google_analytics' or analytics_type == 'ga4':
            if 'property_id' not in kwargs:
                raise ValueError("property_id is required for Google Analytics connector")
            return GoogleAnalyticsConnector(auth_config, kwargs['property_id'], **kwargs)
        elif analytics_type == 'yandex_metrika' or analytics_type == 'yandex_metrica':
            if 'counter_id' not in kwargs:
                raise ValueError("counter_id is required for Yandex.Metrika connector")
            return YandexMetrikaConnector(auth_config, kwargs['counter_id'], **kwargs)
        else:
            raise ValueError(f"Unknown analytics type: {analytics_type}")
    
    @staticmethod
    def detect_analytics(site_url: str) -> List[str]:
        """
        Определяет аналитические системы, используемые на сайте.
        
        Args:
            site_url: URL сайта
            
        Returns:
            List[str]: Список используемых аналитических систем
        """
        result = []
        
        try:
            # Получаем главную страницу сайта
            response = requests.get(
                site_url,
                headers={
                    'User-Agent': 'SEO AI Models Analytics Detector/1.0'
                },
                timeout=10
            )
            response.raise_for_status()
            
            html = response.text
            
            # Проверяем наличие Google Analytics
            ga_patterns = [
                r'google-analytics\.com/analytics\.js',
                r'google-analytics\.com/ga\.js',
                r'googletagmanager\.com/gtag/js\?id=G-',
                r'googletagmanager\.com/gtag/js\?id=UA-',
                r'gtag\(.*G-[A-Z0-9]+\)',
                r'gtag\(.*UA-[0-9]+-[0-9]+\)'
            ]
            
            for pattern in ga_patterns:
                if re.search(pattern, html):
                    result.append('google_analytics')
                    break
            
            # Проверяем наличие Яндекс.Метрики
            ym_patterns = [
                r'mc\.yandex\.ru/metrika/watch\.js',
                r'mc\.yandex\.ru/watch/',
                r'yandex\.ru/metrika',
                r'ym\(.*[0-9]+\)'
            ]
            
            for pattern in ym_patterns:
                if re.search(pattern, html):
                    result.append('yandex_metrika')
                    break
            
            return result
        except requests.RequestException:
            return []


# Класс для интеграции аналитических данных
class AnalyticsIntegration:
    """Класс для интеграции аналитических данных."""
    
    def __init__(self, connectors: Optional[Dict[str, AnalyticsConnector]] = None):
        """
        Инициализирует интеграцию аналитических данных.
        
        Args:
            connectors: Словарь с коннекторами аналитических систем
        """
        self.connectors = connectors or {}
    
    def add_connector(self, name: str, connector: AnalyticsConnector):
        """
        Добавляет коннектор аналитической системы.
        
        Args:
            name: Имя коннектора
            connector: Коннектор аналитической системы
        """
        self.connectors[name] = connector
    
    def remove_connector(self, name: str) -> bool:
        """
        Удаляет коннектор аналитической системы.
        
        Args:
            name: Имя коннектора
            
        Returns:
            bool: True, если коннектор успешно удален, иначе False
        """
        if name in self.connectors:
            del self.connectors[name]
            return True
        return False
    
    def get_visits_combined(self, 
                           start_date: str, 
                           end_date: str, 
                           dimensions: Optional[List[str]] = None,
                           metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Получает комбинированные данные о посещаемости сайта из всех аналитических систем.
        
        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            dimensions: Измерения (группировки) для данных
            metrics: Метрики для данных
            
        Returns:
            Dict[str, Any]: Комбинированные данные о посещаемости
        """
        result = {
            'sources': {},
            'combined': {
                'dimensions': dimensions,
                'metrics': metrics,
                'rows': []
            }
        }
        
        # Получаем данные из всех аналитических систем
        for name, connector in self.connectors.items():
            try:
                data = connector.get_visits(
                    start_date=start_date,
                    end_date=end_date,
                    dimensions=dimensions,
                    metrics=metrics
                )
                
                result['sources'][name] = data
                
                # Добавляем данные в комбинированный результат
                # (здесь должна быть логика объединения данных)
            except Exception as e:
                logger.error(f"Error getting visits from {name}: {str(e)}")
                result['sources'][name] = {
                    'error': str(e)
                }
        
        return result
    
    def get_sources_combined(self,
                            start_date: str,
                            end_date: str) -> Dict[str, Any]:
        """
        Получает комбинированные данные об источниках трафика из всех аналитических систем.
        
        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            Dict[str, Any]: Комбинированные данные об источниках трафика
        """
        result = {
            'sources': {},
            'combined': {
                'rows': []
            }
        }
        
        # Получаем данные из всех аналитических систем
        for name, connector in self.connectors.items():
            try:
                data = connector.get_sources(
                    start_date=start_date,
                    end_date=end_date
                )
                
                result['sources'][name] = data
                
                # Добавляем данные в комбинированный результат
                # (здесь должна быть логика объединения данных)
            except Exception as e:
                logger.error(f"Error getting sources from {name}: {str(e)}")
                result['sources'][name] = {
                    'error': str(e)
                }
        
        return result
    
    def get_page_stats_combined(self,
                               page_path: str,
                               start_date: str,
                               end_date: str) -> Dict[str, Any]:
        """
        Получает комбинированную статистику по конкретной странице из всех аналитических систем.
        
        Args:
            page_path: Путь к странице
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            Dict[str, Any]: Комбинированная статистика по странице
        """
        result = {
            'page_path': page_path,
            'start_date': start_date,
            'end_date': end_date,
            'sources': {},
            'combined': {
                'rows': []
            }
        }
        
        # Получаем данные из всех аналитических систем
        for name, connector in self.connectors.items():
            try:
                data = connector.get_page_stats(
                    page_path=page_path,
                    start_date=start_date,
                    end_date=end_date
                )
                
                result['sources'][name] = data
                
                # Добавляем данные в комбинированный результат
                # (здесь должна быть логика объединения данных)
            except Exception as e:
                logger.error(f"Error getting page stats from {name}: {str(e)}")
                result['sources'][name] = {
                    'error': str(e)
                }
        
        return result
    
    def generate_combined_chart(self,
                               start_date: str,
                               end_date: str,
                               metric: str,
                               chart_type: str = 'line',
                               title: str = '') -> str:
        """
        Генерирует комбинированную диаграмму на основе данных из всех аналитических систем.
        
        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            metric: Метрика для диаграммы
            chart_type: Тип диаграммы ('line', 'bar', 'pie')
            title: Заголовок диаграммы
            
        Returns:
            str: Data URL с диаграммой в формате PNG (base64)
        """
        # Получаем данные из всех аналитических систем
        combined_data = pd.DataFrame()
        
        for name, connector in self.connectors.items():
            try:
                # Настраиваем метрики в зависимости от типа аналитической системы
                if isinstance(connector, GoogleAnalyticsConnector):
                    metrics = [metric]
                elif isinstance(connector, YandexMetrikaConnector):
                    metrics = [f'ym:s:{metric}']
                else:
                    metrics = [metric]
                
                data = connector.get_visits(
                    start_date=start_date,
                    end_date=end_date,
                    dimensions=['date'],
                    metrics=metrics
                )
                
                # Преобразуем данные в DataFrame
                if data['rows']:
                    df = pd.DataFrame(data['rows'])
                    df['source'] = name
                    
                    # Добавляем данные в комбинированный DataFrame
                    if combined_data.empty:
                        combined_data = df
                    else:
                        combined_data = pd.concat([combined_data, df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error getting data for chart from {name}: {str(e)}")
        
        # Если нет данных, возвращаем пустую строку
        if combined_data.empty:
            return ""
        
        # Подготавливаем данные для диаграммы
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        combined_data = combined_data.sort_values('date')
        
        # Получаем первый коннектор для генерации диаграммы
        first_connector = next(iter(self.connectors.values()), None)
        
        if not first_connector:
            return ""
        
        # Генерируем диаграмму
        return first_connector.generate_chart(
            data=combined_data,
            x='date',
            y=[metric] if metric in combined_data.columns else [],
            title=title or f"{metric.capitalize()} from {start_date} to {end_date}",
            chart_type=chart_type
        )
