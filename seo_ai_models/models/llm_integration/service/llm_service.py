"""
Универсальный сервис для взаимодействия с различными LLM API.

Модуль предоставляет единый интерфейс для работы с различными LLM-провайдерами,
абстрагируя детали взаимодействия с их API.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod

# Импортируем компоненты из common
from ..common.constants import (
    LLM_PROVIDERS, 
    LLM_MODELS, 
    DEFAULT_REQUEST_PARAMS
)
from ..common.exceptions import (
    ProviderNotSupportedError,
    ModelNotSupportedError,
    APIConnectionError,
    APIResponseError,
    TokenLimitExceededError,
    LocalModelError
)
from ..common.utils import (
    validate_provider_and_model,
    estimate_tokens,
    estimate_cost,
    get_default_params
)


class LLMProvider(ABC):
    """
    Абстрактный базовый класс для провайдеров LLM.
    
    Определяет общий интерфейс для всех провайдеров LLM, который должен
    быть реализован в конкретных классах-наследниках.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ на основе переданного промпта.
        
        Args:
            prompt: Текст промпта
            **kwargs: Дополнительные параметры запроса
            
        Returns:
            Dict[str, Any]: Ответ от LLM с метаданными
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def estimate_tokens(self, prompt: str) -> int:
        """
        Оценивает количество токенов в промпте для данного провайдера.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            int: Оценка количества токенов
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Оценивает стоимость запроса на основе количества токенов.
        
        Args:
            input_tokens: Количество токенов ввода
            output_tokens: Количество токенов вывода (прогнозируемое)
            
        Returns:
            float: Оценочная стоимость в рублях
        """
        raise NotImplementedError("Subclasses must implement this method")
    

class OpenAIProvider(LLMProvider):
    """
    Провайдер для работы с API OpenAI.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Инициализирует провайдер OpenAI.
        
        Args:
            api_key: API ключ OpenAI (если None, берется из переменной окружения OPENAI_API_KEY)
            model: Модель для использования (по умолчанию gpt-4o-mini)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ OpenAI не найден. Укажите его явно или через переменную окружения OPENAI_API_KEY")
        
        # Проверяем, поддерживается ли модель
        self.model = model
        if model not in LLM_MODELS.get("openai", []):
            raise ModelNotSupportedError(model, "openai", LLM_MODELS.get("openai", []))
        
        # Настройки по умолчанию
        self.default_params = get_default_params("openai")
        
        # Базовый URL для API
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ через API OpenAI.
        
        Args:
            prompt: Текст промпта
            **kwargs: Дополнительные параметры запроса
                - temperature: float = 0.2
                - max_tokens: int = 2048
                - top_p: float = 0.95
                - frequency_penalty: float = 0
                - presence_penalty: float = 0
                - stop: List[str] = None
                
        Returns:
            Dict[str, Any]: Словарь, содержащий ответ и метаданные
                - text: str - текст ответа
                - tokens: Dict[str, int] - информация о токенах
                - cost: float - оценочная стоимость в рублях
                - model: str - использованная модель
                - metadata: Dict[str, Any] - дополнительные метаданные из ответа API
        """
        # Объединяем параметры по умолчанию с переданными
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Формируем запрос
        messages = [{"role": "user", "content": prompt}]
        
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get("temperature", 0.2),
            "max_tokens": params.get("max_tokens", 2048),
            "top_p": params.get("top_p", 0.95),
            "frequency_penalty": params.get("frequency_penalty", 0),
            "presence_penalty": params.get("presence_penalty", 0),
        }
        
        if "stop" in params and params["stop"]:
            request_data["stop"] = params["stop"]
        
        # Логируем информацию о запросе
        self.logger.debug(f"Отправка запроса к OpenAI API. Модель: {self.model}")
        
        # Отправляем запрос
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data,
                timeout=60  # Таймаут 60 секунд
            )
            
            # Проверяем успешность запроса
            response.raise_for_status()
            result = response.json()
            
            # Извлекаем текст из ответа
            text = result["choices"][0]["message"]["content"]
            
            # Получаем информацию о токенах
            tokens = {
                "prompt": result["usage"]["prompt_tokens"],
                "completion": result["usage"]["completion_tokens"],
                "total": result["usage"]["total_tokens"]
            }
            
            # Оцениваем стоимость
            cost = self.estimate_cost(tokens["prompt"], tokens["completion"])
            
            return {
                "text": text,
                "tokens": tokens,
                "cost": cost,
                "model": self.model,
                "metadata": {
                    "finish_reason": result["choices"][0]["finish_reason"],
                    "id": result["id"],
                    "created": result["created"]
                }
            }
            
        except requests.exceptions.RequestException as e:
            # Обрабатываем ошибки соединения
            self.logger.error(f"Ошибка соединения с API OpenAI: {e}")
            raise APIConnectionError("openai", str(e))
        
        except (KeyError, IndexError) as e:
            # Обрабатываем ошибки в структуре ответа
            self.logger.error(f"Ошибка в структуре ответа OpenAI: {e}")
            raise APIResponseError("openai", response.status_code, response.text)
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Оценивает количество токенов в промпте для моделей OpenAI.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            int: Оценка количества токенов
        """
        return estimate_tokens(prompt, self.model)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Оценивает стоимость запроса к OpenAI на основе количества токенов.
        
        Args:
            input_tokens: Количество токенов ввода
            output_tokens: Количество токенов вывода
            
        Returns:
            float: Оценочная стоимость в рублях
        """
        return estimate_cost(input_tokens, output_tokens, "openai", self.model)


class AnthropicProvider(LLMProvider):
    """
    Провайдер для работы с API Anthropic (Claude).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet"):
        """
        Инициализирует провайдер Anthropic.
        
        Args:
            api_key: API ключ Anthropic (если None, берется из переменной окружения ANTHROPIC_API_KEY)
            model: Модель для использования (по умолчанию claude-3-sonnet)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ Anthropic не найден. Укажите его явно или через переменную окружения ANTHROPIC_API_KEY")
        
        # Проверяем, поддерживается ли модель
        self.model = model
        if model not in LLM_MODELS.get("anthropic", []):
            raise ModelNotSupportedError(model, "anthropic", LLM_MODELS.get("anthropic", []))
        
        # Настройки по умолчанию
        self.default_params = get_default_params("anthropic")
        
        # Базовый URL для API
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ через API Anthropic.
        
        Args:
            prompt: Текст промпта
            **kwargs: Дополнительные параметры запроса
                - temperature: float = 0.2
                - max_tokens: int = 2048
                - top_p: float = 0.95
                - stop_sequences: List[str] = None
                
        Returns:
            Dict[str, Any]: Словарь, содержащий ответ и метаданные
                - text: str - текст ответа
                - tokens: Dict[str, int] - информация о токенах
                - cost: float - оценочная стоимость в рублях
                - model: str - использованная модель
                - metadata: Dict[str, Any] - дополнительные метаданные из ответа API
        """
        # Объединяем параметры по умолчанию с переданными
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Формируем запрос
        request_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.2),
            "max_tokens": params.get("max_tokens", 2048),
            "top_p": params.get("top_p", 0.95),
        }
        
        if "stop_sequences" in params and params["stop_sequences"]:
            request_data["stop_sequences"] = params["stop_sequences"]
        
        # Логируем информацию о запросе
        self.logger.debug(f"Отправка запроса к Anthropic API. Модель: {self.model}")
        
        # Отправляем запрос
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json=request_data,
                timeout=60  # Таймаут 60 секунд
            )
            
            # Проверяем успешность запроса
            response.raise_for_status()
            result = response.json()
            
            # Извлекаем текст из ответа
            text = result["content"][0]["text"]
            
            # Получаем информацию о токенах (если доступно)
            tokens = {
                "prompt": result.get("usage", {}).get("input_tokens", 0),
                "completion": result.get("usage", {}).get("output_tokens", 0),
                "total": result.get("usage", {}).get("input_tokens", 0) + 
                        result.get("usage", {}).get("output_tokens", 0)
            }
            
            # Если в ответе нет информации о токенах, оцениваем их количество
            if tokens["prompt"] == 0:
                tokens["prompt"] = self.estimate_tokens(prompt)
                tokens["completion"] = self.estimate_tokens(text)
                tokens["total"] = tokens["prompt"] + tokens["completion"]
            
            # Оцениваем стоимость
            cost = self.estimate_cost(tokens["prompt"], tokens["completion"])
            
            return {
                "text": text,
                "tokens": tokens,
                "cost": cost,
                "model": self.model,
                "metadata": {
                    "stop_reason": result.get("stop_reason"),
                    "id": result.get("id"),
                    "type": result.get("type")
                }
            }
            
        except requests.exceptions.RequestException as e:
            # Обрабатываем ошибки соединения
            self.logger.error(f"Ошибка соединения с API Anthropic: {e}")
            raise APIConnectionError("anthropic", str(e))
        
        except (KeyError, IndexError) as e:
            # Обрабатываем ошибки в структуре ответа
            self.logger.error(f"Ошибка в структуре ответа Anthropic: {e}")
            raise APIResponseError("anthropic", response.status_code, response.text)
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Оценивает количество токенов в промпте для моделей Anthropic.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            int: Оценка количества токенов
        """
        # Anthropic использует свою систему токенизации, 
        # но для оценки можно использовать те же функции
        return estimate_tokens(prompt, "gpt-4")  # примерно соответствует Claude
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Оценивает стоимость запроса к Anthropic на основе количества токенов.
        
        Args:
            input_tokens: Количество токенов ввода
            output_tokens: Количество токенов вывода
            
        Returns:
            float: Оценочная стоимость в рублях
        """
        return estimate_cost(input_tokens, output_tokens, "anthropic", self.model)


class GigaChatProvider(LLMProvider):
    """
    Провайдер для работы с API GigaChat (Сбер).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gigachat-pro"):
        """
        Инициализирует провайдер GigaChat.
        
        Args:
            api_key: API ключ GigaChat (если None, берется из переменной окружения GIGACHAT_API_KEY)
            model: Модель для использования (по умолчанию gigachat-pro)
        """
        self.api_key = api_key or os.environ.get("GIGACHAT_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ GigaChat не найден. Укажите его явно или через переменную окружения GIGACHAT_API_KEY")
        
        # Проверяем, поддерживается ли модель
        self.model = model
        if model not in LLM_MODELS.get("gigachat", []):
            raise ModelNotSupportedError(model, "gigachat", LLM_MODELS.get("gigachat", []))
        
        # Настройки по умолчанию
        self.default_params = get_default_params("gigachat")
        
        # Базовый URL для API
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        
        # URL для получения токена авторизации
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        # Токен авторизации (получаем при первом запросе)
        self.auth_token = None
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def _get_auth_token(self) -> str:
        """
        Получает токен авторизации для API GigaChat.
        
        Returns:
            str: Токен авторизации
        
        Raises:
            APIConnectionError: При ошибке соединения с API авторизации
            APIResponseError: При ошибке в ответе API авторизации
        """
        try:
            response = requests.post(
                self.auth_url,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": "12345",
                    "Authorization": f"Basic {self.api_key}"
                },
                data="scope=GIGACHAT_API_PERS",
                timeout=30  # Таймаут 30 секунд
            )
            
            # Проверяем успешность запроса
            response.raise_for_status()
            result = response.json()
            
            # Извлекаем токен из ответа
            if "access_token" in result:
                return result["access_token"]
            else:
                raise APIResponseError("gigachat", response.status_code, 
                                     "В ответе отсутствует токен доступа")
                
        except requests.exceptions.RequestException as e:
            # Обрабатываем ошибки соединения
            self.logger.error(f"Ошибка соединения с API авторизации GigaChat: {e}")
            raise APIConnectionError("gigachat", str(e))
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ через API GigaChat.
        
        Args:
            prompt: Текст промпта
            **kwargs: Дополнительные параметры запроса
                - temperature: float = 0.2
                - max_tokens: int = 2048
                
        Returns:
            Dict[str, Any]: Словарь, содержащий ответ и метаданные
                - text: str - текст ответа
                - tokens: Dict[str, int] - информация о токенах
                - cost: float - оценочная стоимость в рублях
                - model: str - использованная модель
                - metadata: Dict[str, Any] - дополнительные метаданные из ответа API
        """
        # Объединяем параметры по умолчанию с переданными
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Если у нас нет токена авторизации, получаем его
        if not self.auth_token:
            self.auth_token = self._get_auth_token()
        
        # Формируем запрос
        request_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.2),
            "max_tokens": params.get("max_tokens", 2048),
        }
        
        # Логируем информацию о запросе
        self.logger.debug(f"Отправка запроса к GigaChat API. Модель: {self.model}")
        
        # Отправляем запрос
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.auth_token}"
                },
                json=request_data,
                timeout=60  # Таймаут 60 секунд
            )
            
            # Если токен истек, получаем новый и повторяем запрос
            if response.status_code == 401:
                self.logger.debug("Токен авторизации GigaChat истек. Получаем новый.")
                self.auth_token = self._get_auth_token()
                
                response = requests.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.auth_token}"
                    },
                    json=request_data,
                    timeout=60  # Таймаут 60 секунд
                )
            
            # Проверяем успешность запроса
            response.raise_for_status()
            result = response.json()
            
            # Извлекаем текст из ответа
            text = result["choices"][0]["message"]["content"]
            
            # Получаем информацию о токенах
            tokens = {
                "prompt": result["usage"]["prompt_tokens"],
                "completion": result["usage"]["completion_tokens"],
                "total": result["usage"]["total_tokens"]
            }
            
            # Оцениваем стоимость
            cost = self.estimate_cost(tokens["prompt"], tokens["completion"])
            
            return {
                "text": text,
                "tokens": tokens,
                "cost": cost,
                "model": self.model,
                "metadata": {
                    "finish_reason": result["choices"][0].get("finish_reason"),
                    "id": result.get("id"),
                    "created": result.get("created")
                }
            }
            
        except requests.exceptions.RequestException as e:
            # Обрабатываем ошибки соединения
            self.logger.error(f"Ошибка соединения с API GigaChat: {e}")
            raise APIConnectionError("gigachat", str(e))
        
        except (KeyError, IndexError) as e:
            # Обрабатываем ошибки в структуре ответа
            self.logger.error(f"Ошибка в структуре ответа GigaChat: {e}")
            raise APIResponseError("gigachat", response.status_code, response.text)
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Оценивает количество токенов в промпте для моделей GigaChat.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            int: Оценка количества токенов
        """
        # GigaChat использует свою систему токенизации, 
        # но для оценки можно использовать примерно те же функции
        return estimate_tokens(prompt, "gpt-3.5-turbo")  # примерно соответствует GigaChat
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Оценивает стоимость запроса к GigaChat на основе количества токенов.
        
        Args:
            input_tokens: Количество токенов ввода
            output_tokens: Количество токенов вывода
            
        Returns:
            float: Оценочная стоимость в рублях
        """
        return estimate_cost(input_tokens, output_tokens, "gigachat", self.model)


class LocalLLMProvider(LLMProvider):
    """
    Провайдер для работы с локально развернутыми LLM.
    
    Поддерживает работу с моделями, развернутыми локально через API,
    совместимый с OpenAI или через прямое взаимодействие с моделями.
    """
    
    def __init__(self, model: str = "llama-3-8b", api_url: Optional[str] = None):
        """
        Инициализирует провайдер для локальных LLM.
        
        Args:
            model: Название модели
            api_url: URL для API локальной модели (по умолчанию http://localhost:8000/v1/chat/completions)
        """
        # Проверяем, поддерживается ли модель
        self.model = model
        if model not in LLM_MODELS.get("local", []):
            raise ModelNotSupportedError(model, "local", LLM_MODELS.get("local", []))
        
        # URL для API
        self.api_url = api_url or "http://localhost:8000/v1/chat/completions"
        
        # Настройки по умолчанию
        self.default_params = get_default_params("local")
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ через API локальной модели.
        
        Args:
            prompt: Текст промпта
            **kwargs: Дополнительные параметры запроса
                - temperature: float = 0.1
                - max_tokens: int = 1024
                - top_p: float = 0.9
                
        Returns:
            Dict[str, Any]: Словарь, содержащий ответ и метаданные
                - text: str - текст ответа
                - tokens: Dict[str, int] - информация о токенах
                - cost: float - оценочная стоимость в рублях
                - model: str - использованная модель
                - metadata: Dict[str, Any] - дополнительные метаданные из ответа API
        """
        # Объединяем параметры по умолчанию с переданными
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Формируем запрос
        request_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.1),
            "max_tokens": params.get("max_tokens", 1024),
            "top_p": params.get("top_p", 0.9),
        }
        
        # Логируем информацию о запросе
        self.logger.debug(f"Отправка запроса к локальной модели. Модель: {self.model}")
        
        # Отправляем запрос
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=request_data,
                timeout=120  # Таймаут 120 секунд для локальных моделей
            )
            
            # Проверяем успешность запроса
            response.raise_for_status()
            result = response.json()
            
            # Извлекаем текст из ответа
            text = result["choices"][0]["message"]["content"]
            
            # Получаем информацию о токенах (если доступно)
            tokens = {}
            
            if "usage" in result:
                tokens = {
                    "prompt": result["usage"].get("prompt_tokens", 0),
                    "completion": result["usage"].get("completion_tokens", 0),
                    "total": result["usage"].get("total_tokens", 0)
                }
            else:
                # Если в ответе нет информации о токенах, оцениваем их количество
                tokens["prompt"] = self.estimate_tokens(prompt)
                tokens["completion"] = self.estimate_tokens(text)
                tokens["total"] = tokens["prompt"] + tokens["completion"]
            
            # Оцениваем стоимость
            cost = self.estimate_cost(tokens["prompt"], tokens["completion"])
            
            return {
                "text": text,
                "tokens": tokens,
                "cost": cost,
                "model": self.model,
                "metadata": {
                    "finish_reason": result["choices"][0].get("finish_reason"),
                    "id": result.get("id", "local_generation"),
                }
            }
            
        except requests.exceptions.RequestException as e:
            # Обрабатываем ошибки соединения
            self.logger.error(f"Ошибка соединения с API локальной модели: {e}")
            raise LocalModelError(self.model, str(e))
        
        except (KeyError, IndexError) as e:
            # Обрабатываем ошибки в структуре ответа
            self.logger.error(f"Ошибка в структуре ответа локальной модели: {e}")
            raise LocalModelError(self.model, f"Ошибка в структуре ответа: {e}")
    
    # Продолжение файла llm_service.py
    def estimate_tokens(self, prompt: str) -> int:
        """
        Оценивает количество токенов в промпте для локальных моделей.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            int: Оценка количества токенов
        """
        # Для локальных моделей используем базовую оценку
        return len(prompt) // 4  # Примерно 4 символа на токен
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Оценивает стоимость запроса к локальной модели на основе количества токенов.
        
        Args:
            input_tokens: Количество токенов ввода
            output_tokens: Количество токенов вывода
            
        Returns:
            float: Оценочная стоимость в рублях
        """
        return estimate_cost(input_tokens, output_tokens, "local", self.model)


class LLMService:
    """
    Универсальный сервис для работы с различными LLM провайдерами.
    
    Предоставляет единый интерфейс для взаимодействия с различными LLM,
    абстрагируя детали взаимодействия с конкретными API.
    """
    
    def __init__(self):
        """
        Инициализирует сервис LLM.
        """
        # Словарь с поддерживаемыми провайдерами и их классами
        self.provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gigachat": GigaChatProvider,
            "local": LocalLLMProvider,
        }
        
        # Словарь с экземплярами провайдеров
        self.providers = {}
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def add_provider(self, provider_type: str, **kwargs) -> None:
        """
        Добавляет провайдера в сервис.
        
        Args:
            provider_type: Тип провайдера (openai, anthropic, gigachat, local)
            **kwargs: Дополнительные параметры для инициализации провайдера
                - api_key: API ключ
                - model: Модель для использования
                
        Raises:
            ProviderNotSupportedError: Если провайдер не поддерживается
        """
        if provider_type not in self.provider_classes:
            raise ProviderNotSupportedError(provider_type, list(self.provider_classes.keys()))
        
        # Создаем экземпляр провайдера
        provider_class = self.provider_classes[provider_type]
        try:
            provider = provider_class(**kwargs)
            self.providers[provider_type] = provider
            self.logger.info(f"Добавлен провайдер {provider_type} с моделью {provider.model}")
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении провайдера {provider_type}: {e}")
            raise
    
    def remove_provider(self, provider_type: str) -> None:
        """
        Удаляет провайдера из сервиса.
        
        Args:
            provider_type: Тип провайдера
        """
        if provider_type in self.providers:
            del self.providers[provider_type]
            self.logger.info(f"Удален провайдер {provider_type}")
    
    def generate(self, prompt: str, provider: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ через указанного провайдера.
        
        Args:
            prompt: Текст промпта
            provider: Тип провайдера
            **kwargs: Дополнительные параметры запроса
                
        Returns:
            Dict[str, Any]: Ответ от LLM с метаданными
            
        Raises:
            ValueError: Если провайдер не был добавлен в сервис
        """
        if provider not in self.providers:
            raise ValueError(f"Провайдер {provider} не был добавлен в сервис")
        
        # Получаем экземпляр провайдера
        llm_provider = self.providers[provider]
        
        # Логируем информацию о запросе
        self.logger.debug(f"Генерация ответа с помощью провайдера {provider}")
        
        # Генерируем ответ
        return llm_provider.generate(prompt, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """
        Возвращает список добавленных провайдеров.
        
        Returns:
            List[str]: Список добавленных провайдеров
        """
        return list(self.providers.keys())
    
    def estimate_tokens(self, prompt: str, provider: str) -> int:
        """
        Оценивает количество токенов в промпте для указанного провайдера.
        
        Args:
            prompt: Текст промпта
            provider: Тип провайдера
            
        Returns:
            int: Оценка количества токенов
            
        Raises:
            ValueError: Если провайдер не был добавлен в сервис
        """
        if provider not in self.providers:
            raise ValueError(f"Провайдер {provider} не был добавлен в сервис")
        
        return self.providers[provider].estimate_tokens(prompt)
    
    def estimate_cost(self, prompt: str, provider: str, 
                     expected_output_length: int = None) -> float:
        """
        Оценивает стоимость запроса для указанного провайдера.
        
        Args:
            prompt: Текст промпта
            provider: Тип провайдера
            expected_output_length: Ожидаемая длина ответа в токенах 
                                   (если None, используется 25% от длины промпта)
            
        Returns:
            float: Оценочная стоимость в рублях
            
        Raises:
            ValueError: Если провайдер не был добавлен в сервис
        """
        if provider not in self.providers:
            raise ValueError(f"Провайдер {provider} не был добавлен в сервис")
        
        # Оцениваем количество токенов в промпте
        input_tokens = self.providers[provider].estimate_tokens(prompt)
        
        # Если ожидаемая длина ответа не указана, предполагаем 25% от длины промпта
        if expected_output_length is None:
            expected_output_length = input_tokens // 4
        
        return self.providers[provider].estimate_cost(input_tokens, expected_output_length)
    
    def get_provider_model(self, provider: str) -> str:
        """
        Возвращает модель, используемую указанным провайдером.
        
        Args:
            provider: Тип провайдера
            
        Returns:
            str: Название модели
            
        Raises:
            ValueError: Если провайдер не был добавлен в сервис
        """
        if provider not in self.providers:
            raise ValueError(f"Провайдер {provider} не был добавлен в сервис")
        
        return self.providers[provider].model
