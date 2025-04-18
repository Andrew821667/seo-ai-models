"""
Интегратор обработки JavaScript для проекта SEO AI Models.
Объединяет WebSocket, GraphQL и клиентскую маршрутизацию в единый интерфейс.
"""

import logging
from typing import Dict, List, Set, Optional, Any, Union

from seo_ai_models.parsers.unified.js_processing.websocket_analyzer import WebSocketAnalyzer
from seo_ai_models.parsers.unified.js_processing.graphql_interceptor import GraphQLInterceptor
from seo_ai_models.parsers.unified.js_processing.client_routing_handler import ClientRoutingHandler

logger = logging.getLogger(__name__)

class JSProcessingIntegrator:
    """
    Объединяет различные компоненты обработки JavaScript в единый интерфейс.
    Обеспечивает централизованный доступ к функциям анализа JavaScript.
    """
    
    def __init__(
        self,
        enable_websocket: bool = True,
        enable_graphql: bool = True,
        enable_client_routing: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Инициализация интегратора обработки JavaScript.
        
        Args:
            enable_websocket: Включить анализ WebSocket
            enable_graphql: Включить анализ GraphQL
            enable_client_routing: Включить обработку клиентской маршрутизации
            log_level: Уровень логирования
        """
        self.logger = logging.getLogger("JSProcessingIntegrator")
        self.logger.setLevel(log_level)
        
        self.enable_websocket = enable_websocket
        self.enable_graphql = enable_graphql
        self.enable_client_routing = enable_client_routing
        
        # Инициализация компонентов
        if self.enable_websocket:
            self.websocket_analyzer = WebSocketAnalyzer(log_level)
            
        if self.enable_graphql:
            self.graphql_interceptor = GraphQLInterceptor(log_level)
            
        if self.enable_client_routing:
            self.routing_handler = ClientRoutingHandler(log_level)
            
        self.logger.info("JSProcessingIntegrator инициализирован")
        
    def register_websocket_pattern(self, name: str, key_fields: List[str], 
                                 value_fields: List[str], type_identifier: Optional[str] = None) -> None:
        """
        Регистрирует шаблон для извлечения данных из сообщений WebSocket.
        
        Args:
            name: Имя шаблона
            key_fields: Ключевые поля для идентификации
            value_fields: Поля значений для извлечения
            type_identifier: Опциональный идентификатор типа
        """
        if not self.enable_websocket:
            self.logger.warning("Анализ WebSocket отключен")
            return
            
        self.websocket_analyzer.register_pattern(name, key_fields, value_fields, type_identifier)
        
    def register_graphql_pattern(self, name: str, field_path: str, 
                               operation_type: Optional[str] = None,
                               operation_name: Optional[str] = None) -> None:
        """
        Регистрирует шаблон для извлечения данных из ответов GraphQL.
        
        Args:
            name: Имя шаблона
            field_path: Путь к полю в нотации с точками
            operation_type: Тип операции
            operation_name: Имя операции
        """
        if not self.enable_graphql:
            self.logger.warning("Анализ GraphQL отключен")
            return
            
        self.graphql_interceptor.register_pattern(name, field_path, operation_type, operation_name)
        
    def register_route_pattern(self, pattern: str, name: str, params: Optional[List[str]] = None) -> None:
        """
        Регистрирует шаблон маршрута для распознавания.
        
        Args:
            pattern: Регулярное выражение для сопоставления с маршрутом
            name: Имя шаблона
            params: Список имен параметров
        """
        if not self.enable_client_routing:
            self.logger.warning("Обработка клиентской маршрутизации отключена")
            return
            
        self.routing_handler.register_route_pattern(pattern, name, params)
        
    def get_js_analysis_results(self) -> Dict[str, Any]:
        """
        Возвращает объединенные результаты анализа JavaScript.
        
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        results = {}
        
        # Добавляем результаты WebSocket
        if self.enable_websocket:
            results["websocket"] = {
                "statistics": self.websocket_analyzer.get_message_statistics(),
                "active_connections": self.websocket_analyzer.get_active_connections(),
                "extracted_data": self.websocket_analyzer.get_extracted_content()
            }
            
        # Добавляем результаты GraphQL
        if self.enable_graphql:
            results["graphql"] = {
                "statistics": self.graphql_interceptor.get_statistics(),
                "extracted_data": self.graphql_interceptor.get_extracted_data()
            }
            
        # Добавляем результаты маршрутизации
        if self.enable_client_routing:
            results["routing"] = {
                "statistics": self.routing_handler.get_route_statistics(),
                "history": self.routing_handler.get_route_history(),
                "unique_routes": self.routing_handler.get_unique_routes()
            }
            
        return results
    
    def clear_all_data(self) -> None:
        """Очищает все данные анализа."""
        if self.enable_websocket:
            self.websocket_analyzer.clear()
            
        if self.enable_graphql:
            self.graphql_interceptor.clear()
            
        if self.enable_client_routing:
            self.routing_handler.clear()
            
        self.logger.info("Все данные анализа JavaScript очищены")
