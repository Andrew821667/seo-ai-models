"""
GraphQL-интерцептор для проекта SEO AI Models.
Перехватывает и анализирует GraphQL-запросы для извлечения структурированных данных.
"""

import json
import re
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple, Union

class GraphQLOperation:
    """Структура для хранения GraphQL-операций"""
    
    def __init__(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        operation_type: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        self.query = query
        self.variables = variables or {}
        self.operation_name = operation_name
        self.operation_type = operation_type or self._extract_operation_type(query)
        self.endpoint = endpoint
        self.timestamp = time.time()
        
    def _extract_operation_type(self, query: str) -> Optional[str]:
        """Извлекает тип операции из строки запроса GraphQL"""
        if not query:
            return None
            
        # Простое регулярное выражение для извлечения типа операции
        match = re.search(r'^\s*(query|mutation|subscription)', query)
        if match:
            return match.group(1)
            
        # Если нет явного типа, это запрос по умолчанию
        if '{' in query:
            return "query"
            
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь"""
        return {
            "query": self.query,
            "variables": self.variables,
            "operation_name": self.operation_name,
            "operation_type": self.operation_type,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }

class GraphQLResponse:
    """Структура для хранения ответов GraphQL"""
    
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        status_code: Optional[int] = None,
        operation: Optional[GraphQLOperation] = None
    ):
        self.data = data or {}
        self.errors = errors or []
        self.status_code = status_code
        self.operation = operation
        self.timestamp = time.time()
        self.extracted_data = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь"""
        return {
            "data": self.data,
            "errors": self.errors,
            "status_code": self.status_code,
            "operation": self.operation.to_dict() if self.operation else None,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "extracted_data": self.extracted_data
        }

class GraphQLInterceptor:
    """
    Перехватывает и анализирует GraphQL-запросы для извлечения структурированных данных.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Инициализация GraphQL-интерцептора.
        
        Args:
            log_level: Уровень логирования
        """
        self.logger = logging.getLogger("GraphQLInterceptor")
        self.logger.setLevel(log_level)
        self.operations = []
        self.responses = []
        self.patterns = []
        self.endpoints = set()
        self.operation_types = set()
        self.operation_names = set()
        
    def register_pattern(self, name: str, field_path: str, operation_type: Optional[str] = None,
                       operation_name: Optional[str] = None) -> None:
        """
        Регистрирует шаблон для извлечения данных из ответов GraphQL.
        
        Args:
            name: Имя шаблона
            field_path: Путь к полю в нотации с точками (например, 'data.users.items')
            operation_type: Тип операции ('query', 'mutation', 'subscription') или None для всех
            operation_name: Имя операции или None для всех
        """
        self.patterns.append({
            "name": name,
            "field_path": field_path,
            "operation_type": operation_type,
            "operation_name": operation_name
        })
        
        self.logger.info(f"Зарегистрирован GraphQL-шаблон: {name}")
    
    def intercept_request(self, url: str, headers: Dict[str, str], body: Union[str, Dict, List]) -> List[GraphQLOperation]:
        """
        Перехватывает GraphQL-запрос.
        
        Args:
            url: URL-эндпоинт
            headers: Заголовки запроса
            body: Тело запроса (строка или объект)
            
        Returns:
            List[GraphQLOperation]: Список перехваченных операций
        """
        try:
            self.endpoints.add(url)
            intercepted_operations = []
            
            # Обработка тела запроса
            if isinstance(body, str):
                try:
                    parsed_body = json.loads(body)
                except json.JSONDecodeError:
                    self.logger.warning(f"Нераспознанное тело GraphQL-запроса: {body[:100]}...")
                    return []
            else:
                parsed_body = body
            
            # Поддержка пакетных запросов (массив операций)
            operations_data = []
            if isinstance(parsed_body, list):
                operations_data = parsed_body
            else:
                operations_data = [parsed_body]
                
            for op_data in operations_data:
                if not isinstance(op_data, dict):
                    continue
                    
                query = op_data.get("query", "")
                variables = op_data.get("variables", {})
                operation_name = op_data.get("operationName")
                
                if not query:
                    continue
                
                operation = GraphQLOperation(
                    query=query,
                    variables=variables,
                    operation_name=operation_name,
                    endpoint=url
                )
                
                # Обновляем статистику
                if operation.operation_type:
                    self.operation_types.add(operation.operation_type)
                if operation.operation_name:
                    self.operation_names.add(operation.operation_name)
                
                self.operations.append(operation)
                intercepted_operations.append(operation)
                
                self.logger.debug(
                    f"Перехвачен GraphQL-{operation.operation_type or 'unknown'}"
                    f"{' (' + operation.operation_name + ')' if operation.operation_name else ''}"
                    f" к {url}"
                )
            
            return intercepted_operations
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки GraphQL-запроса: {str(e)}")
            return []
    
    def intercept_response(self, response_body: Union[str, Dict, List], 
                          operations: Optional[List[GraphQLOperation]] = None,
                          status_code: Optional[int] = None) -> List[GraphQLResponse]:
        """
        Перехватывает ответ GraphQL.
        
        Args:
            response_body: Тело ответа (строка или объект)
            operations: Список операций, к которым относится ответ (если известно)
            status_code: HTTP-код статуса
            
        Returns:
            List[GraphQLResponse]: Список перехваченных ответов
        """
        try:
            # Обработка тела ответа
            if isinstance(response_body, str):
                try:
                    parsed_body = json.loads(response_body)
                except json.JSONDecodeError:
                    self.logger.warning(f"Нераспознанное тело GraphQL-ответа: {response_body[:100]}...")
                    return []
            else:
                parsed_body = response_body
            
            # Поддержка пакетных ответов (массив результатов)
            responses_data = []
            if isinstance(parsed_body, list):
                responses_data = parsed_body
            else:
                responses_data = [parsed_body]
                
            intercepted_responses = []
            
            for i, res_data in enumerate(responses_data):
                if not isinstance(res_data, dict):
                    continue
                
                # Сопоставление с операцией, если возможно
                operation = None
                if operations and i < len(operations):
                    operation = operations[i]
                
                response = GraphQLResponse(
                    data=res_data.get("data"),
                    errors=res_data.get("errors", []),
                    status_code=status_code,
                    operation=operation
                )
                
                # Применяем шаблоны для извлечения данных
                self._extract_response_data(response)
                
                self.responses.append(response)
                intercepted_responses.append(response)
                
                op_name = operation.operation_name if operation else "unknown"
                self.logger.debug(f"Перехвачен GraphQL-ответ для операции {op_name}")
            
            return intercepted_responses
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки GraphQL-ответа: {str(e)}")
            return []
    
    def _extract_response_data(self, response: GraphQLResponse) -> None:
        """
        Извлекает данные из ответа на основе зарегистрированных шаблонов.
        
        Args:
            response: Объект ответа
        """
        if not response.data:
            return
            
        # Применяем каждый шаблон
        for pattern in self.patterns:
            # Проверяем, соответствует ли шаблон типу операции
            if pattern["operation_type"] and response.operation:
                if response.operation.operation_type != pattern["operation_type"]:
                    continue
            
            # Проверяем, соответствует ли шаблон имени операции
            if pattern["operation_name"] and response.operation:
                if response.operation.operation_name != pattern["operation_name"]:
                    continue
            
            # Извлекаем данные по пути поля
            field_path = pattern["field_path"]
            extracted = self._extract_by_path(response.data, field_path)
            
            if extracted:
                pattern_name = pattern["name"]
                response.extracted_data[pattern_name] = extracted
    
    def _extract_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """
        Извлекает значение из структуры данных по пути.
        
        Args:
            data: Структура данных
            path: Путь в нотации с точками (например, 'users.items')
            
        Returns:
            Any: Извлеченное значение или None, если путь не найден
        """
        if not data or not path:
            return None
            
        parts = path.split('.')
        current = data
        
        try:
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        except:
            return None
    
    def get_operations_by_type(self, operation_type: str) -> List[GraphQLOperation]:
        """
        Возвращает операции определенного типа.
        
        Args:
            operation_type: Тип операции ('query', 'mutation', 'subscription')
            
        Returns:
            List[GraphQLOperation]: Список операций
        """
        return [op for op in self.operations if op.operation_type == operation_type]
    
    def get_operations_by_name(self, operation_name: str) -> List[GraphQLOperation]:
        """
        Возвращает операции с определенным именем.
        
        Args:
            operation_name: Имя операции
            
        Returns:
            List[GraphQLOperation]: Список операций
        """
        return [op for op in self.operations if op.operation_name == operation_name]
    
    def get_extracted_data(self, pattern_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Возвращает извлеченные данные из всех ответов.
        
        Args:
            pattern_name: Опциональное имя шаблона для фильтрации
            
        Returns:
            List[Dict[str, Any]]: Список извлеченных данных
        """
        result = []
        
        for response in self.responses:
            if pattern_name:
                if pattern_name in response.extracted_data:
                    result.append({
                        "data": response.extracted_data[pattern_name],
                        "operation": response.operation.to_dict() if response.operation else None,
                        "timestamp": response.timestamp
                    })
            else:
                for name, data in response.extracted_data.items():
                    result.append({
                        "pattern": name,
                        "data": data,
                        "operation": response.operation.to_dict() if response.operation else None,
                        "timestamp": response.timestamp
                    })
                    
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по перехваченным операциям и ответам.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        return {
            "total_operations": len(self.operations),
            "total_responses": len(self.responses),
            "endpoint_count": len(self.endpoints),
            "operation_types": list(self.operation_types),
            "operation_names": list(self.operation_names),
            "operations_by_type": {
                op_type: len(self.get_operations_by_type(op_type))
                for op_type in self.operation_types
            },
            "successful_responses": len([r for r in self.responses if not r.errors]),
            "error_responses": len([r for r in self.responses if r.errors])
        }
    
    def clear(self) -> None:
        """Очищает все перехваченные операции и ответы."""
        self.operations = []
        self.responses = []
        self.logger.info("Очищены все перехваченные GraphQL-операции и ответы")
