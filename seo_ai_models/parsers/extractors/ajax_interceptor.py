"""
Модуль для перехвата и анализа AJAX-запросов в SPA.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple

from playwright.async_api import Page, Request, Response, Route

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AJAXInterceptor:
    """
    Перехватывает и анализирует AJAX-запросы в SPA-приложениях.
    """

    def __init__(
        self,
        record_api_calls: bool = True,
        analyze_responses: bool = True,
        ignore_extensions: List[str] = None,
        api_patterns: List[str] = None,
    ):
        """
        Инициализация AJAXInterceptor.

        Args:
            record_api_calls: Записывать ли API-вызовы
            analyze_responses: Анализировать ли ответы
            ignore_extensions: Расширения файлов для игнорирования
            api_patterns: Паттерны URL для идентификации API-вызовов
        """
        self.record_api_calls = record_api_calls
        self.analyze_responses = analyze_responses

        self.ignore_extensions = ignore_extensions or [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".svg",
            ".css",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".ico",
            ".mp4",
            ".webp",
            ".mp3",
            ".ogg",
            ".wav",
        ]

        self.api_patterns = api_patterns or [
            "/api/",
            "/graphql",
            "/v1/",
            "/v2/",
            "/v3/",
            "/rest/",
            "json",
            "data.",
            ".php",
            "ajax",
            "xhr",
            "action=",
        ]

        self.api_calls: List[Dict[str, Any]] = []
        self.json_responses: Dict[str, Any] = {}
        self.completed_requests: Set[str] = set()

    def _is_api_request(self, url: str) -> bool:
        """
        Определяет, является ли URL API-вызовом.

        Args:
            url: URL для проверки

        Returns:
            bool: True, если URL является API-вызовом
        """
        # Игнорируем запросы к статическим файлам
        if any(url.lower().endswith(ext) for ext in self.ignore_extensions):
            return False

        # Проверяем на паттерны API
        return any(pattern.lower() in url.lower() for pattern in self.api_patterns)

    async def setup_request_interception(self, page: Page):
        """
        Устанавливает перехват запросов на странице.

        Args:
            page: Объект Page Playwright
        """

        # Обработчик запросов
        async def on_request(request: Request):
            if not self.record_api_calls:
                return

            url = request.url

            if self._is_api_request(url):
                # Записываем информацию о запросе
                method = request.method
                headers = request.headers
                post_data = request.post_data

                self.api_calls.append(
                    {
                        "url": url,
                        "method": method,
                        "headers": headers,
                        "post_data": post_data,
                        "timestamp": asyncio.get_event_loop().time(),
                        "response": None,
                    }
                )

                logger.debug(f"Обнаружен API-вызов: {method} {url}")

        # Обработчик ответов
        async def on_response(response: Response):
            if not self.analyze_responses:
                return

            url = response.url

            if self._is_api_request(url) and url not in self.completed_requests:
                try:
                    content_type = response.headers.get("content-type", "")

                    if (
                        "application/json" in content_type
                        or "application/javascript" in content_type
                    ):
                        # Получаем текст ответа
                        text = await response.text()

                        # Пробуем распарсить JSON
                        try:
                            json_data = json.loads(text)

                            # Сохраняем ответ
                            self.json_responses[url] = json_data

                            # Обновляем записанный API-вызов
                            for call in self.api_calls:
                                if call["url"] == url and call["response"] is None:
                                    call["response"] = {
                                        "status": response.status,
                                        "headers": response.headers,
                                        "json": json_data,
                                    }
                                    break

                            logger.debug(f"Получен JSON-ответ от {url}")
                        except json.JSONDecodeError:
                            logger.debug(f"Не удалось распарсить JSON из ответа {url}")

                    self.completed_requests.add(url)

                except Exception as e:
                    logger.error(f"Ошибка при обработке ответа от {url}: {str(e)}")

        # Устанавливаем обработчики
        page.on("request", on_request)
        page.on("response", on_response)

    def get_api_calls(self) -> List[Dict[str, Any]]:
        """
        Получает список перехваченных API-вызовов.

        Returns:
            List[Dict[str, Any]]: Список API-вызовов с ответами
        """
        return self.api_calls

    def get_json_responses(self) -> Dict[str, Any]:
        """
        Получает словарь JSON-ответов.

        Returns:
            Dict[str, Any]: Словарь JSON-ответов по URL
        """
        return self.json_responses

    def extract_data_from_responses(self) -> Dict[str, Any]:
        """
        Извлекает структурированные данные из JSON-ответов.

        Returns:
            Dict[str, Any]: Структурированные данные из ответов
        """
        data = {"api_endpoints": [], "data_structure": {}, "entities": {}}

        # Анализ API-вызовов
        if self.api_calls:
            data["api_endpoints"] = [
                {
                    "url": call["url"],
                    "method": call["method"],
                    "status": call.get("response", {}).get("status"),
                }
                for call in self.api_calls
                if call.get("response")
            ]

        # Анализ структуры данных из ответов
        for url, json_data in self.json_responses.items():
            # Пытаемся определить тип данных и структуру
            if isinstance(json_data, dict):
                # Извлекаем верхнеуровневые ключи
                keys = list(json_data.keys())

                # Определяем, содержит ли ответ данные о сущностях
                if any(key in ["data", "items", "results", "content", "response"] for key in keys):
                    for key in ["data", "items", "results", "content", "response"]:
                        if key in json_data and json_data[key]:
                            # Извлекаем потенциальные сущности
                            entity_data = json_data[key]

                            if isinstance(entity_data, list) and entity_data:
                                # Если это список сущностей
                                entity_type = key
                                entity_sample = entity_data[0] if entity_data else {}

                                if isinstance(entity_sample, dict):
                                    data["entities"][entity_type] = {
                                        "fields": list(entity_sample.keys()),
                                        "count": len(entity_data),
                                        "sample": entity_sample,
                                    }

                # Сохраняем структуру ответа
                short_url = url.split("/")[-1].split("?")[0]
                data["data_structure"][short_url] = {"keys": keys, "url": url}

        return data
