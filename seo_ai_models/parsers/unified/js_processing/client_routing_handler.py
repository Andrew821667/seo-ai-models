"""
Обработчик клиентской маршрутизации для проекта SEO AI Models.
Отслеживает изменения URL и состояния в SPA-приложениях.
"""

import re
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from urllib.parse import urlparse


class ClientRoutingHandler:
    """
    Обработчик клиентской маршрутизации для SPA-приложений.
    Отслеживает изменения маршрутов без перезагрузки страницы.
    """

    def __init__(self, log_level: int = logging.INFO):
        """
        Инициализация обработчика маршрутизации.

        Args:
            log_level: Уровень логирования
        """
        self.logger = logging.getLogger("ClientRoutingHandler")
        self.logger.setLevel(log_level)
        self.route_changes = []
        self.visited_routes = set()
        self.detected_framework = None
        self.route_patterns = []

    def detect_router_framework(
        self, html_content: str, window_objects: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Определяет фреймворк маршрутизации, используемый на странице.

        Args:
            html_content: HTML-контент страницы
            window_objects: Список объектов JavaScript window (опционально)

        Returns:
            Optional[Dict[str, Any]]: Обнаруженный фреймворк или None
        """
        # Признаки различных фреймворков маршрутизации
        router_signatures = [
            # React Router
            {
                "name": "React Router",
                "html_patterns": [
                    "react-router",
                    "browserrouter",
                    "hashrouter",
                    "<Router",
                    "createBrowserRouter",
                ],
                "window_objects": ["ReactRouter", "createBrowserRouter"],
                "confidence": 0.8,
            },
            # Vue Router
            {
                "name": "Vue Router",
                "html_patterns": ["vue-router", "router-view", "router-link"],
                "window_objects": ["VueRouter", "$router"],
                "confidence": 0.9,
            },
            # Angular Router
            {
                "name": "Angular Router",
                "html_patterns": ["ng-view", "routerLink", "router-outlet"],
                "window_objects": ["NgRouter", "RouteReuseStrategy"],
                "confidence": 0.8,
            },
        ]

        # Проверяем сигнатуры HTML
        for signature in router_signatures:
            for pattern in signature["html_patterns"]:
                if re.search(pattern, html_content, re.IGNORECASE):
                    framework = {
                        "name": signature["name"],
                        "confidence": signature["confidence"],
                        "detected_by": f"html_pattern:{pattern}",
                    }
                    self.detected_framework = framework
                    self.logger.info(
                        f"Обнаружен фреймворк маршрутизации: {framework['name']} (уверенность: {framework['confidence']})"
                    )
                    return framework

        # Проверяем объекты window, если предоставлены
        if window_objects:
            for signature in router_signatures:
                for obj in signature["window_objects"]:
                    if obj in window_objects:
                        framework = {
                            "name": signature["name"],
                            "confidence": signature["confidence"],
                            "detected_by": f"window_object:{obj}",
                        }
                        self.detected_framework = framework
                        self.logger.info(
                            f"Обнаружен фреймворк маршрутизации: {framework['name']} (уверенность: {framework['confidence']})"
                        )
                        return framework

        self.logger.info("Фреймворк маршрутизации не обнаружен")
        return None

    def register_route_pattern(
        self, pattern: str, name: str, params: Optional[List[str]] = None
    ) -> None:
        """
        Регистрирует шаблон маршрута для распознавания.

        Args:
            pattern: Регулярное выражение для сопоставления с маршрутом
            name: Имя шаблона
            params: Список имен параметров, которые можно извлечь из маршрута
        """
        self.route_patterns.append(
            {"pattern": pattern, "name": name, "params": params or [], "regex": re.compile(pattern)}
        )

        self.logger.info(f"Зарегистрирован шаблон маршрута: {name} ({pattern})")

    def record_route_change(
        self, from_url: str, to_url: str, state_object: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Записывает изменение маршрута.

        Args:
            from_url: Исходный URL
            to_url: Новый URL
            state_object: Объект состояния (из History API)

        Returns:
            Dict[str, Any]: Информация об изменении маршрута
        """
        # Создаем запись об изменении маршрута
        route_change = {
            "from_url": from_url,
            "to_url": to_url,
            "state_object": state_object or {},
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
        }

        # Извлекаем путь из URL
        try:
            parsed_url = urlparse(to_url)
            route_change["path"] = parsed_url.path
        except:
            route_change["path"] = ""

        # Сопоставляем с шаблонами маршрутов
        if route_change["path"]:
            route_info = self._match_route_patterns(route_change["path"])
            if route_info:
                route_change["route_info"] = route_info

        # Добавляем URL в посещенные маршруты
        self.visited_routes.add(to_url)

        # Сохраняем изменение
        self.route_changes.append(route_change)

        self.logger.info(f"Записано изменение маршрута: {from_url} -> {to_url}")

        return route_change

    def _match_route_patterns(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Сопоставляет путь с зарегистрированными шаблонами маршрутов.

        Args:
            path: Путь для сопоставления

        Returns:
            Optional[Dict[str, Any]]: Информация о маршруте или None
        """
        for route in self.route_patterns:
            match = route["regex"].match(path)
            if match:
                # Извлекаем параметры из маршрута
                params = {}
                if route["params"] and match.groups():
                    for i, param_name in enumerate(route["params"]):
                        if i < len(match.groups()):
                            params[param_name] = match.groups()[i]

                return {"name": route["name"], "pattern": route["pattern"], "params": params}

        return None

    def get_route_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает историю изменений маршрутов.

        Returns:
            List[Dict[str, Any]]: История маршрутов
        """
        return self.route_changes

    def get_unique_routes(self) -> List[Dict[str, Any]]:
        """
        Возвращает список уникальных маршрутов, которые были посещены.

        Returns:
            List[Dict[str, Any]]: Список уникальных маршрутов
        """
        unique_routes = []

        for route_change in self.route_changes:
            route = {
                "url": route_change["to_url"],
                "path": route_change.get("path", ""),
                "route_info": route_change.get("route_info"),
            }
            unique_routes.append(route)

        return unique_routes

    def get_route_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по маршрутам.

        Returns:
            Dict[str, Any]: Статистика
        """
        return {
            "total_changes": len(self.route_changes),
            "unique_routes": len(self.visited_routes),
            "route_patterns": len(self.route_patterns),
            "framework": self.detected_framework,
        }
