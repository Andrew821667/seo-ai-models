"""
WebSocket Analyzer для проекта SEO AI Models.
Отслеживает и анализирует WebSocket-коммуникацию для извлечения данных
с интерактивных сайтов, использующих постоянное соединение.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple, Union

class WebSocketMessage:
    """Структура для хранения WebSocket-сообщений"""
    
    def __init__(
        self, 
        content: Any, 
        direction: str = "incoming", 
        endpoint: Optional[str] = None,
        timestamp: Optional[float] = None
    ):
        self.content = content
        self.direction = direction  # "incoming" или "outgoing"
        self.endpoint = endpoint
        self.timestamp = timestamp or time.time()
        self.extracted_data = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь"""
        return {
            "content": self.content,
            "direction": self.direction,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
            "extracted_data": self.extracted_data,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }

class WebSocketPattern:
    """Шаблон для извлечения данных из WebSocket-сообщений"""
    
    def __init__(
        self, 
        name: str, 
        key_fields: List[str], 
        value_fields: List[str], 
        type_identifier: Optional[str] = None
    ):
        self.name = name
        self.key_fields = key_fields
        self.value_fields = value_fields
        self.type_identifier = type_identifier
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь"""
        return {
            "name": self.name,
            "key_fields": self.key_fields,
            "value_fields": self.value_fields,
            "type_identifier": self.type_identifier
        }

class WebSocketAnalyzer:
    """
    Анализатор WebSocket-коммуникации для извлечения данных с сайтов,
    использующих WebSocket для реального времени контента.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Инициализация анализатора WebSocket.
        
        Args:
            log_level: Уровень логирования
        """
        self.logger = logging.getLogger("WebSocketAnalyzer")
        self.logger.setLevel(log_level)
        self.messages = []
        self.patterns = []
        self.endpoints = set()
        self.connection_info = {}
        
    def register_pattern(self, name: str, key_fields: List[str], value_fields: List[str], 
                        type_identifier: Optional[str] = None) -> None:
        """
        Регистрирует шаблон для идентификации контента в WebSocket-сообщениях.
        
        Args:
            name: Имя для идентификации этого шаблона
            key_fields: Список полей, идентифицирующих тип контента
            value_fields: Список полей, содержащих ценный контент
            type_identifier: Опциональное строковое значение для идентификации типа сообщения
        """
        pattern = WebSocketPattern(name, key_fields, value_fields, type_identifier)
        self.patterns.append(pattern)
        self.logger.info(f"Зарегистрирован WebSocket-шаблон: {name}")
    
    def capture_message(self, message_data: Union[str, Dict, List], direction: str = "incoming", 
                       endpoint: Optional[str] = None) -> Optional[WebSocketMessage]:
        """
        Захватывает WebSocket-сообщение для анализа.
        
        Args:
            message_data: Содержимое сообщения (строка или объект)
            direction: 'incoming' или 'outgoing' для указания направления
            endpoint: URL-эндпоинт WebSocket, если доступен
            
        Returns:
            WebSocketMessage: Объект сообщения или None в случае ошибки
        """
        try:
            # Обработка строковых данных
            if isinstance(message_data, str):
                try:
                    parsed_data = json.loads(message_data)
                except json.JSONDecodeError:
                    # Не JSON, сохраняем как есть
                    parsed_data = message_data
            else:
                parsed_data = message_data
            
            # Создаем объект сообщения
            message = WebSocketMessage(
                content=parsed_data,
                direction=direction,
                endpoint=endpoint
            )
            
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
                        "last_activity": time.time()
                    }
            
            # Обновляем статистику
            if endpoint in self.connection_info:
                self.connection_info[endpoint]["message_count"] += 1
                self.connection_info[endpoint]["last_activity"] = time.time()
                
                if direction == "incoming":
                    self.connection_info[endpoint]["incoming_count"] += 1
                else:
                    self.connection_info[endpoint]["outgoing_count"] += 1
            
            # Обрабатываем сообщение с помощью зарегистрированных шаблонов
            self._process_message(message)
            
            # Сохраняем сообщение
            self.messages.append(message)
            
            msg_preview = str(message_data)[:100] + "..." if len(str(message_data)) > 100 else str(message_data)
            self.logger.debug(f"Захвачено WebSocket-сообщение от {endpoint}: {msg_preview}")
            
            return message
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки WebSocket-сообщения: {str(e)}")
            return None
    
    def _process_message(self, message: WebSocketMessage) -> None:
        """
        Обрабатывает WebSocket-сообщение с зарегистрированными шаблонами.
        
        Args:
            message: Объект сообщения
        """
        content = message.content
        
        for pattern in self.patterns:
            extracted = self._extract_by_pattern(content, pattern)
            if extracted:
                pattern_name = pattern.name
                if pattern_name not in message.extracted_data:
                    message.extracted_data[pattern_name] = []
                message.extracted_data[pattern_name].append(extracted)
    
    def _extract_by_pattern(self, content: Any, pattern: WebSocketPattern) -> Optional[Dict[str, Any]]:
        """
        Извлекает контент из сообщения по заданному шаблону.
        
        Args:
            content: Содержимое сообщения
            pattern: Шаблон для применения
            
        Returns:
            Dict[str, Any]: Извлеченный контент или None если шаблон не подходит
        """
        # Обработка объектов
        if isinstance(content, dict):
            # Проверка идентификатора типа, если указан
            if pattern.type_identifier:
                type_matches = False
                for key in content:
                    if key in pattern.key_fields and str(content[key]) == pattern.type_identifier:
                        type_matches = True
                        break
                if not type_matches:
                    return None
            
            # Извлечение значений из полей
            result = {}
            for field in pattern.key_fields + pattern.value_fields:
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
    
    def get_extracted_content(self, pattern_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
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
            is_active = (time.time() - info["last_activity"]) < 60
            
            connections.append({
                "endpoint": endpoint,
                "first_seen": datetime.fromtimestamp(info["first_seen"]).isoformat(),
                "message_count": info["message_count"],
                "incoming_count": info["incoming_count"],
                "outgoing_count": info["outgoing_count"],
                "last_activity": datetime.fromtimestamp(info["last_activity"]).isoformat(),
                "is_active": is_active,
                "duration": time.time() - info["first_seen"]
            })
            
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
            "extracted_data_count": sum(len(m.extracted_data) for m in self.messages)
        }
    
    def get_content_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """
        Извлекает контент определенного типа из всех сообщений.
        
        Args:
            content_type: Тип контента для извлечения
            
        Returns:
            List[Dict[str, Any]]: Список извлеченного контента
        """
        result = []
        
        for message in self.messages:
            if content_type in message.extracted_data:
                result.extend(message.extracted_data[content_type])
                
        return result
    
    def clear(self) -> None:
        """Очищает все захваченные сообщения и сбрасывает анализатор."""
        self.messages = []
        self.logger.info("Очищены все WebSocket-сообщения")
