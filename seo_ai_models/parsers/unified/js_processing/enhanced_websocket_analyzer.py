"""
Улучшенный WebSocket-анализатор для проекта SEO AI Models.
Обеспечивает продвинутое обнаружение и анализ WebSocket-коммуникации.
"""

import json
import time
import re
import logging
import base64
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple, Union, Callable


class WebSocketMessage:
    """Структура для хранения WebSocket-сообщений"""

    def __init__(
        self,
        content: Any,
        direction: str = "incoming",
        endpoint: Optional[str] = None,
        timestamp: Optional[float] = None,
        binary: bool = False,
        opcode: Optional[int] = None,
    ):
        self.content = content
        self.direction = direction  # "incoming" или "outgoing"
        self.endpoint = endpoint
        self.timestamp = timestamp or time.time()
        self.extracted_data = {}
        self.binary = binary
        self.opcode = opcode
        self.protocol = None
        self.message_type = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь"""
        return {
            "content": self.content,
            "direction": self.direction,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
            "extracted_data": self.extracted_data,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "binary": self.binary,
            "opcode": self.opcode,
            "protocol": self.protocol,
            "message_type": self.message_type,
        }


class EnhancedWebSocketAnalyzer:
    """
    Улучшенный анализатор WebSocket-коммуникации для извлечения данных с сайтов,
    использующих WebSocket для реального времени контента.
    """

    # Сигнатуры протоколов WebSocket
    PROTOCOLS = {
        "socketio": {
            "patterns": [
                r"^\d+\{.*\}$",  # Socket.IO v2/v3
                r"^\d+:.*$",  # Socket.IO v1
                r"^\d+-\d+:.*$",  # Socket.IO бинарный
            ],
            "identifiers": ["socket.io", "io.connect", "io.socket"],
        },
        "sockjs": {
            "patterns": [
                r"^[aohc]\[.*\]$",  # SockJS типы сообщений
                r'^[moe]:".*"$',  # SockJS с escape
            ],
            "identifiers": ["sockjs", "/sockjs/"],
        },
        "signalr": {
            "patterns": [
                r'^\{.*"target":.*"arguments":.*\}$',  # SignalR Core
                r'^\{.*"H":.*"M":.*"A":.*\}$',  # SignalR Legacy
            ],
            "identifiers": ["signalr", "hubConnection"],
        },
        "graphqlws": {
            "patterns": [
                r'^\{"type":"(connection_init|subscribe|complete|next|error)".*\}$',
                r'^\{"id":".*","type":".*".*\}$',
            ],
            "identifiers": ["graphql-ws", "subscriptions-transport"],
        },
        "mqtt": {
            "patterns": [r'^\{"topic":".*","payload":\{.*\}\}$'],
            "identifiers": ["mqtt", "mqttjs"],
        },
        "stomp": {
            "patterns": [r"^CONNECT\n", r"^SUBSCRIBE\n", r"^SEND\n"],
            "identifiers": ["stomp", "stompclient", "stompjs"],
        },
        "raw": {
            "patterns": [r"^.*$"],  # Всегда совпадает, используется как последнее средство
            "identifiers": [],
        },
    }

    # Типы сообщений по протоколам
    MESSAGE_TYPES = {
        "socketio": {
            "connect": [r"^0\{.*\}$", r"^0:.*$"],
            "disconnect": [r"^1\{.*\}$", r"^1:.*$"],
            "event": [r"^2\{.*\}$", r"^2:.*$"],
            "ack": [r"^3\{.*\}$", r"^3:.*$"],
            "error": [r"^4\{.*\}$", r"^4:.*$"],
            "binary": [r"^5\{.*\}$", r"^5:.*$"],
        },
        "sockjs": {
            "open": [r"^o$"],
            "heartbeat": [r"^h$"],
            "message": [r"^a\[.*\]$"],
            "close": [r"^c\[.*\]$"],
        },
        "signalr": {
            "invocation": [r'^\{.*"type":\s*1.*\}$', r'^\{.*"invocationId":.*"target":.*\}$'],
            "stream_item": [r'^\{.*"type":\s*2.*\}$'],
            "completion": [r'^\{.*"type":\s*3.*\}$'],
            "stream_invocation": [r'^\{.*"type":\s*4.*\}$'],
            "cancel_invocation": [r'^\{.*"type":\s*5.*\}$'],
            "ping": [r'^\{.*"type":\s*6.*\}$'],
            "close": [r'^\{.*"type":\s*7.*\}$'],
        },
        "graphqlws": {
            "connection_init": [r'^\{.*"type":\s*"connection_init".*\}$'],
            "connection_ack": [r'^\{.*"type":\s*"connection_ack".*\}$'],
            "subscribe": [r'^\{.*"type":\s*"subscribe".*\}$'],
            "next": [r'^\{.*"type":\s*"next".*\}$'],
            "error": [r'^\{.*"type":\s*"error".*\}$'],
            "complete": [r'^\{.*"type":\s*"complete".*\}$'],
        },
    }

    def __init__(self, log_level: int = logging.INFO):
        """
        Инициализация улучшенного анализатора WebSocket.

        Args:
            log_level: Уровень логирования
        """
        self.logger = logging.getLogger("EnhancedWebSocketAnalyzer")
        self.logger.setLevel(log_level)
        self.messages = []
        self.patterns = []
        self.endpoints = set()
        self.connection_info = {}
        self.protocol_stats = {}
        self.message_type_stats = {}
        self.data_extractors = {}  # Функции извлечения данных для протоколов

        # Регистрация стандартных экстракторов данных
        self._register_default_extractors()

        self.logger.info("Инициализирован улучшенный WebSocket-анализатор")

    def _register_default_extractors(self) -> None:
        """Регистрирует стандартные экстракторы данных для протоколов"""
        # Socket.IO
        self.register_data_extractor("socketio", self._extract_socketio_data)

        # SignalR
        self.register_data_extractor("signalr", self._extract_signalr_data)

        # GraphQL WS
        self.register_data_extractor("graphqlws", self._extract_graphqlws_data)

        # SockJS
        self.register_data_extractor("sockjs", self._extract_sockjs_data)

        # MQTT
        self.register_data_extractor("mqtt", self._extract_mqtt_data)

    def register_pattern(
        self,
        name: str,
        key_fields: List[str],
        value_fields: List[str],
        type_identifier: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> None:
        """
        Регистрирует шаблон для идентификации контента в WebSocket-сообщениях.

        Args:
            name: Имя для идентификации этого шаблона
            key_fields: Список полей, идентифицирующих тип контента
            value_fields: Список полей, содержащих ценный контент
            type_identifier: Опциональное строковое значение для идентификации типа сообщения
            protocol: Опциональный протокол, для которого применяется шаблон
        """
        self.patterns.append(
            {
                "name": name,
                "key_fields": key_fields,
                "value_fields": value_fields,
                "type_identifier": type_identifier,
                "protocol": protocol,
            }
        )

        self.logger.info(
            f"Зарегистрирован WebSocket-шаблон: {name}"
            + (f" для протокола {protocol}" if protocol else "")
        )

    def register_data_extractor(self, protocol: str, extractor_func: Callable) -> None:
        """
        Регистрирует функцию извлечения данных для определенного протокола.

        Args:
            protocol: Название протокола
            extractor_func: Функция-экстрактор, принимающая сообщение и возвращающая извлеченные данные
        """
        self.data_extractors[protocol] = extractor_func
        self.logger.info(f"Зарегистрирован экстрактор данных для протокола: {protocol}")

    def capture_message(
        self,
        message_data: Union[str, bytes, Dict, List],
        direction: str = "incoming",
        endpoint: Optional[str] = None,
        opcode: Optional[int] = None,
    ) -> Optional[WebSocketMessage]:
        """
        Захватывает WebSocket-сообщение для анализа.

        Args:
            message_data: Содержимое сообщения
            direction: 'incoming' или 'outgoing' для указания направления
            endpoint: URL-эндпоинт WebSocket, если доступен
            opcode: Опкод WebSocket (1 - текст, 2 - бинарные данные)

        Returns:
            WebSocketMessage: Объект сообщения или None в случае ошибки
        """
        try:
            # Определяем, бинарное ли это сообщение
            is_binary = isinstance(message_data, bytes) or opcode == 2

            # Обработка бинарных данных
            if is_binary and isinstance(message_data, bytes):
                # Пробуем интерпретировать как UTF-8 строку
                try:
                    parsed_data = message_data.decode("utf-8")
                except UnicodeDecodeError:
                    # Если не получается, сохраняем как Base64
                    parsed_data = {
                        "_binary": base64.b64encode(message_data).decode("ascii"),
                        "_size": len(message_data),
                    }
            # Обработка строковых данных
            elif isinstance(message_data, str):
                # Пробуем распарсить как JSON
                try:
                    parsed_data = json.loads(message_data)
                except json.JSONDecodeError:
                    # Не JSON, сохраняем как есть
                    parsed_data = message_data
            else:
                # Если уже объект, используем его
                parsed_data = message_data

            # Создаем объект сообщения
            message = WebSocketMessage(
                content=parsed_data,
                direction=direction,
                endpoint=endpoint,
                binary=is_binary,
                opcode=opcode,
            )

            # Определение протокола
            protocol = self._detect_protocol(parsed_data, endpoint)
            message.protocol = protocol

            # Определение типа сообщения на основе протокола
            message_type = None
            if protocol and protocol in self.MESSAGE_TYPES:
                message_type = self._detect_message_type(protocol, parsed_data)
                message.message_type = message_type

            # Отслеживание эндпоинтов
            if endpoint and endpoint not in self.endpoints:
                self.endpoints.add(endpoint)
                self.logger.info(f"Обнаружен новый WebSocket-эндпоинт: {endpoint}")

                # Инициализация информации о соединении
                if endpoint not in self.connection_info:
                    self.connection_info[endpoint] = {
                        "first_seen": time.time(),
                        "message_count": 0,
                        "incoming_count": 0,
                        "outgoing_count": 0,
                        "last_activity": time.time(),
                        "protocols_detected": set(),
                        "message_types": {},
                    }

            # Обновляем статистику
            if endpoint in self.connection_info:
                self.connection_info[endpoint]["message_count"] += 1
                self.connection_info[endpoint]["last_activity"] = time.time()

                if direction == "incoming":
                    self.connection_info[endpoint]["incoming_count"] += 1
                else:
                    self.connection_info[endpoint]["outgoing_count"] += 1

                if protocol:
                    self.connection_info[endpoint]["protocols_detected"].add(protocol)

                if message_type:
                    if message_type not in self.connection_info[endpoint]["message_types"]:
                        self.connection_info[endpoint]["message_types"][message_type] = 0
                    self.connection_info[endpoint]["message_types"][message_type] += 1

            # Обновляем статистику протоколов и типов сообщений
            if protocol:
                if protocol not in self.protocol_stats:
                    self.protocol_stats[protocol] = 0
                self.protocol_stats[protocol] += 1

                if message_type:
                    key = f"{protocol}.{message_type}"
                    if key not in self.message_type_stats:
                        self.message_type_stats[key] = 0
                    self.message_type_stats[key] += 1

            # Извлечение данных на основе протокола
            if protocol and protocol in self.data_extractors:
                extractor = self.data_extractors[protocol]
                extracted_data = extractor(message)
                if extracted_data:
                    for key, value in extracted_data.items():
                        if key not in message.extracted_data:
                            message.extracted_data[key] = []
                        if isinstance(value, list):
                            message.extracted_data[key].extend(value)
                        else:
                            message.extracted_data[key].append(value)

            # Обрабатываем сообщение с помощью зарегистрированных шаблонов
            self._process_message(message)

            # Сохраняем сообщение
            self.messages.append(message)

            msg_preview = (
                str(message_data)[:100] + "..."
                if len(str(message_data)) > 100
                else str(message_data)
            )
            self.logger.debug(f"Захвачено WebSocket-сообщение от {endpoint}: {msg_preview}")

            return message

        except Exception as e:
            self.logger.error(f"Ошибка обработки WebSocket-сообщения: {str(e)}")
            return None

    def _extract_socketio_data(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Извлекает данные из Socket.IO сообщения"""
        result = {}

        if not isinstance(message.content, str):
            return result

        # Для Socket.IO v2/v3 формат: <packet type>[<data>]
        match = re.match(r"^(\d+)(.*)$", message.content)
        if match:
            packet_type = match.group(1)
            data = match.group(2)

            # Если это JSON, распарсим его
            if data.startswith("{") and data.endswith("}"):
                try:
                    json_data = json.loads(data)
                    # Если это событие, извлечем его имя и аргументы
                    if packet_type == "2" and isinstance(json_data, dict) and "data" in json_data:
                        event_data = json_data["data"]
                        if isinstance(event_data, list) and len(event_data) > 0:
                            event_name = event_data[0] if isinstance(event_data[0], str) else None
                            event_args = event_data[1:] if len(event_data) > 1 else []

                            if event_name:
                                result["events"] = {"name": event_name, "args": event_args}
                    # Для других типов просто сохраняем данные
                    else:
                        result["data"] = json_data
                except:
                    pass

        return result

    def _extract_signalr_data(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Извлекает данные из SignalR сообщения"""
        result = {}

        if not isinstance(message.content, (dict, str)):
            return result

        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except:
                return result

        # SignalR Core формат
        if isinstance(content, dict):
            # Извлекаем данные вызова метода
            if "target" in content and "arguments" in content:
                result["invocations"] = {"method": content["target"], "args": content["arguments"]}
            # Извлекаем данные потокового элемента
            elif "type" in content and content["type"] == 2 and "item" in content:
                result["stream_items"] = content["item"]
            # Извлекаем данные завершения потока
            elif "type" in content and content["type"] == 3 and "result" in content:
                result["completions"] = content["result"]

        return result

    def _extract_graphqlws_data(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Извлекает данные из GraphQL WebSocket сообщения"""
        result = {}

        if not isinstance(message.content, (dict, str)):
            return result

        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except:
                return result

        # GraphQL WS формат
        if isinstance(content, dict) and "type" in content:
            msg_type = content["type"]

            # Извлекаем данные подписки
            if msg_type == "subscribe" and "payload" in content:
                result["subscriptions"] = content["payload"]
            # Извлекаем данные события
            elif msg_type == "next" and "payload" in content:
                result["events"] = content["payload"]
            # Извлекаем ошибки
            elif msg_type == "error" and "payload" in content:
                result["errors"] = content["payload"]

        return result

    def _extract_sockjs_data(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Извлекает данные из SockJS сообщения"""
        result = {}

        if not isinstance(message.content, str):
            return result

        # SockJS формат: a["message"]
        match = re.match(r"^a\[(.*?)\]$", message.content)
        if match:
            data_str = match.group(1)
            try:
                # Убираем кавычки и escape
                if data_str.startswith('"') and data_str.endswith('"'):
                    data_str = data_str[1:-1].replace('"', '"')

                # Пытаемся распарсить как JSON
                try:
                    json_data = json.loads(data_str)
                    result["messages"] = json_data
                except:
                    result["messages"] = data_str
            except:
                pass

        return result

    def _extract_mqtt_data(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Извлекает данные из MQTT сообщения"""
        result = {}

        if not isinstance(message.content, (dict, str)):
            return result

        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except:
                return result

        # MQTT формат: {"topic": "...", "payload": {...}}
        if isinstance(content, dict) and "topic" in content and "payload" in content:
            result["messages"] = {"topic": content["topic"], "payload": content["payload"]}

        return result

    def _detect_protocol(self, content: Any, endpoint: Optional[str] = None) -> Optional[str]:
        """
        Определяет протокол WebSocket по содержимому и эндпоинту.

        Args:
            content: Содержимое сообщения
            endpoint: URL-эндпоинт WebSocket

        Returns:
            Optional[str]: Название протокола или None
        """
        # Анализ по эндпоинту
        if endpoint:
            for protocol, info in self.PROTOCOLS.items():
                for ident in info["identifiers"]:
                    if ident.lower() in endpoint.lower():
                        return protocol

        # Анализ по содержимому сообщения
        content_str = None
        if isinstance(content, (dict, list)):
            try:
                content_str = json.dumps(content)
            except:
                content_str = str(content)
        else:
            content_str = str(content)

        for protocol, info in self.PROTOCOLS.items():
            for pattern in info["patterns"]:
                if re.match(pattern, content_str):
                    return protocol

        # Если нет совпадений, возвращаем raw
        return "raw"

    def _detect_message_type(self, protocol: str, content: Any) -> Optional[str]:
        """
        Определяет тип сообщения на основе протокола.

        Args:
            protocol: Название протокола
            content: Содержимое сообщения

        Returns:
            Optional[str]: Тип сообщения или None
        """
        if protocol not in self.MESSAGE_TYPES:
            return None

        content_str = None
        if isinstance(content, (dict, list)):
            try:
                content_str = json.dumps(content)
            except:
                content_str = str(content)
        else:
            content_str = str(content)

        for msg_type, patterns in self.MESSAGE_TYPES[protocol].items():
            for pattern in patterns:
                if re.match(pattern, content_str):
                    return msg_type

        return None

    def _process_message(self, message: WebSocketMessage) -> None:
        """
        Обрабатывает WebSocket-сообщение с зарегистрированными шаблонами.

        Args:
            message: Объект сообщения
        """
        content = message.content

        for pattern in self.patterns:
            # Если указан протокол и он не совпадает с протоколом сообщения, пропускаем
            if (
                pattern.get("protocol")
                and message.protocol
                and pattern["protocol"] != message.protocol
            ):
                continue

            extracted = self._extract_by_pattern(content, pattern)
            if extracted:
                pattern_name = pattern["name"]
                if pattern_name not in message.extracted_data:
                    message.extracted_data[pattern_name] = []
                message.extracted_data[pattern_name].append(extracted)

    def _extract_by_pattern(
        self, content: Any, pattern: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Извлекает контент из сообщения по заданному шаблону.

        Args:
            content: Содержимое сообщения
            pattern: Шаблон для применения

        Returns:
            Optional[Dict[str, Any]]: Извлеченный контент или None
        """
        # Обработка объектов
        if isinstance(content, dict):
            # Проверка идентификатора типа, если указан
            if pattern.get("type_identifier"):
                type_matches = False
                for key in content:
                    if (
                        key in pattern["key_fields"]
                        and str(content[key]) == pattern["type_identifier"]
                    ):
                        type_matches = True
                        break
                if not type_matches:
                    return None

            # Извлечение значений из полей
            result = {}
            for field in pattern.get("key_fields", []) + pattern.get("value_fields", []):
                if field in content:
                    result[field] = content[field]

            return result if result else None

        # Обработка массивов
        elif isinstance(content, list):
            results = []
            for item in content:
                extracted = self._extract_by_pattern(item, pattern)
                if extracted:
                    results.append(extracted)
            return {"items": results} if results else None

        return None

    def get_extracted_content(
        self, pattern_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Получает весь извлеченный контент из захваченных WebSocket-сообщений.

        Args:
            pattern_name: Опциональное имя для фильтрации по конкретному шаблону

        Returns:
            Dict[str, List[Dict[str, Any]]]: Словарь извлеченного контента
        """
        result = {}

        for message in self.messages:
            for name, content_list in message.extracted_data.items():
                if pattern_name and name != pattern_name:
                    continue

                if name not in result:
                    result[name] = []

                result[name].extend(content_list)

        return result

    def get_active_connections(self) -> List[Dict[str, Any]]:
        """
        Получает информацию об активных WebSocket-соединениях.

        Returns:
            List[Dict[str, Any]]: Информация о соединениях
        """
        connections = []

        for endpoint, info in self.connection_info.items():
            # Считаем соединение активным, если активность была в течение последних 60 секунд
            is_active = (time.time() - info.get("last_activity", 0)) < 60

            connections.append(
                {
                    "endpoint": endpoint,
                    "first_seen": datetime.fromtimestamp(info.get("first_seen", 0)).isoformat(),
                    "message_count": info.get("message_count", 0),
                    "incoming_count": info.get("incoming_count", 0),
                    "outgoing_count": info.get("outgoing_count", 0),
                    "last_activity": datetime.fromtimestamp(
                        info.get("last_activity", 0)
                    ).isoformat(),
                    "is_active": is_active,
                    "duration": time.time() - info.get("first_seen", time.time()),
                    "protocols_detected": list(info.get("protocols_detected", set())),
                }
            )

        return connections

    def get_message_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по сообщениям.

        Returns:
            Dict[str, Any]: Статистика сообщений
        """
        incoming = [m for m in self.messages if m.direction == "incoming"]
        outgoing = [m for m in self.messages if m.direction == "outgoing"]

        return {
            "total_messages": len(self.messages),
            "incoming_messages": len(incoming),
            "outgoing_messages": len(outgoing),
            "endpoint_count": len(self.endpoints),
            "patterns_count": len(self.patterns),
            "protocol_stats": self.protocol_stats,
            "message_type_stats": self.message_type_stats,
        }
