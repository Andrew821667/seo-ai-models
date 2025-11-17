"""
Улучшенный GraphQL-интерцептор для проекта SEO AI Models.
Обеспечивает продвинутое обнаружение и анализ GraphQL-запросов.
"""

import json
import re
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple, Union


class EnhancedGraphQLInterceptor:
    """
    Улучшенный перехватчик GraphQL с продвинутыми алгоритмами обнаружения.
    """

    # Сигнатуры популярных GraphQL клиентов
    GRAPHQL_CLIENT_SIGNATURES = {
        "apollo": ["apollographql", "apollo-client", "__apollo_client__", "ApolloClient", "gql`"],
        "relay": ["relay", "RelayEnvironment", "__RELAY_STORE__", "RelayModernEnvironment"],
        "urql": ["urql", "createClient", "useQuery", "useMutation"],
        "graphql_request": ["graphql-request", "request", "gql`"],
    }

    # Типичные схемы GraphQL запросов
    GRAPHQL_QUERY_PATTERNS = [
        # Простой запрос
        r"query\s+\w*\s*{[\s\S]*?}",
        # Запрос с параметрами
        r"query\s+\w*\s*\([\s\S]*?\)\s*{[\s\S]*?}",
        # Мутация
        r"mutation\s+\w*\s*\([\s\S]*?\)\s*{[\s\S]*?}",
        # Подписка
        r"subscription\s+\w*\s*{[\s\S]*?}",
        # Фрагменты
        r"fragment\s+\w+\s+on\s+\w+\s*{[\s\S]*?}",
    ]

    # Типичные поля ответов GraphQL
    GRAPHQL_RESPONSE_FIELDS = [
        {"data": {}, "errors": []},
        {"data": {}, "extensions": {}},
        {"data": None, "errors": [{"message": "", "locations": []}]},
        {"data": {}, "extensions": {"tracing": {}}},
    ]

    def __init__(self, log_level: int = logging.INFO):
        """
        Инициализация интерцептора.

        Args:
            log_level: Уровень логирования
        """
        self.logger = logging.getLogger("EnhancedGraphQLInterceptor")
        self.logger.setLevel(log_level)
        self.operations = []
        self.responses = []
        self.patterns = []
        self.endpoints = set()
        self.operation_types = set()
        self.operation_names = set()
        self.detected_clients = set()
        self.schema_fragments = {}

        self.logger.info("Инициализирован улучшенный GraphQL-интерцептор")

    def register_pattern(
        self,
        name: str,
        field_path: str,
        operation_type: Optional[str] = None,
        operation_name: Optional[str] = None,
    ) -> None:
        """
        Регистрирует шаблон для извлечения данных из ответов GraphQL.

        Args:
            name: Имя шаблона
            field_path: Путь к полю в нотации с точками
            operation_type: Тип операции ('query', 'mutation', 'subscription')
            operation_name: Имя операции
        """
        self.patterns.append(
            {
                "name": name,
                "field_path": field_path,
                "operation_type": operation_type,
                "operation_name": operation_name,
            }
        )

        self.logger.info(f"Зарегистрирован GraphQL-шаблон: {name}")

    def intercept_request(
        self, url: str, headers: Dict[str, str], body: Union[str, Dict, List], method: str = "POST"
    ) -> List[Dict[str, Any]]:
        """
        Перехватывает и анализирует GraphQL-запрос.

        Args:
            url: URL эндпоинта
            headers: Заголовки запроса
            body: Тело запроса
            method: HTTP метод

        Returns:
            List[Dict[str, Any]]: Список операций
        """
        # Проверяем, является ли это GraphQL-запросом
        if not self._is_graphql_request(url, method, headers, body):
            return []

        try:
            self.endpoints.add(url)
            intercepted_operations = []

            # Обработка тела запроса
            if isinstance(body, str):
                try:
                    parsed_body = json.loads(body)
                except json.JSONDecodeError:
                    # Проверяем, может быть это запрос в формате строки
                    parsed_body = self._extract_graphql_from_string(body)
                    if not parsed_body:
                        self.logger.warning(f"Нераспознанное тело GraphQL-запроса: {body[:100]}...")
                        return []
            else:
                parsed_body = body

            # Определение клиента GraphQL по заголовкам
            client_type = self._detect_graphql_client(
                headers, url, str(body) if isinstance(body, (dict, list)) else body
            )
            if client_type:
                self.detected_clients.add(client_type)
                self.logger.info(f"Обнаружен GraphQL-клиент: {client_type}")

            # Обработка операций
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

                # Извлекаем тип операции
                operation_type = self._extract_operation_type(query)

                # Извлекаем информацию о схеме
                self._extract_schema_info(query)

                operation = {
                    "query": query,
                    "variables": variables,
                    "operation_name": operation_name,
                    "operation_type": operation_type,
                    "endpoint": url,
                    "client_type": client_type,
                    "timestamp": time.time(),
                    "headers": headers,
                }

                # Обновляем статистику
                if operation_type:
                    self.operation_types.add(operation_type)
                if operation_name:
                    self.operation_names.add(operation_name)

                self.operations.append(operation)
                intercepted_operations.append(operation)

                self.logger.debug(
                    f"Перехвачен GraphQL-{operation_type or 'unknown'}"
                    f"{' (' + operation_name + ')' if operation_name else ''}"
                    f" к {url}"
                )

            return intercepted_operations

        except Exception as e:
            self.logger.error(f"Ошибка обработки GraphQL-запроса: {str(e)}")
            return []

    def intercept_response(
        self,
        response_body: Union[str, Dict, List],
        operations: Optional[List[Dict[str, Any]]] = None,
        status_code: Optional[int] = None,
        url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Перехватывает и анализирует ответ GraphQL.

        Args:
            response_body: Тело ответа
            operations: Связанные операции
            status_code: HTTP код статуса
            url: URL эндпоинта

        Returns:
            List[Dict[str, Any]]: Список ответов
        """
        try:
            # Проверяем, является ли это ответом GraphQL
            if not self._is_graphql_response(response_body):
                return []

            # Обработка тела ответа
            if isinstance(response_body, str):
                try:
                    parsed_body = json.loads(response_body)
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"Нераспознанное тело GraphQL-ответа: {response_body[:100]}..."
                    )
                    return []
            else:
                parsed_body = response_body

            # Поддержка пакетных ответов
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

                response = {
                    "data": res_data.get("data"),
                    "errors": res_data.get("errors", []),
                    "extensions": res_data.get("extensions", {}),
                    "status_code": status_code,
                    "operation": operation,
                    "timestamp": time.time(),
                    "url": url or (operation.get("endpoint") if operation else None),
                    "extracted_data": {},
                }

                # Применяем шаблоны для извлечения данных
                self._extract_response_data(response)

                # Извлекаем информацию о типах данных для схемы
                if response.get("data"):
                    self._extract_schema_from_response(response)

                self.responses.append(response)
                intercepted_responses.append(response)

                op_name = operation.get("operation_name") if operation else "unknown"
                self.logger.debug(f"Перехвачен GraphQL-ответ для операции {op_name}")

            return intercepted_responses

        except Exception as e:
            self.logger.error(f"Ошибка обработки GraphQL-ответа: {str(e)}")
            return []

    def _is_graphql_request(
        self, url: str, method: str, headers: Dict[str, str], body: Union[str, Dict, List]
    ) -> bool:
        """
        Определяет, является ли запрос GraphQL-запросом.

        Args:
            url: URL эндпоинта
            method: HTTP метод
            headers: Заголовки запроса
            body: Тело запроса

        Returns:
            bool: True, если это GraphQL-запрос
        """
        # Проверка по URL
        if "graphql" in url.lower() or "/api/graphql" in url.lower():
            return True

        # Проверка заголовков
        for header, value in headers.items():
            header_lower = header.lower()
            if "apollo" in header_lower or "graphql" in header_lower:
                return True

            # Проверка Content-Type
            if header_lower == "content-type" and "application/graphql" in value.lower():
                return True

            # Проверка на другие GraphQL клиенты в заголовках
            for client in self.GRAPHQL_CLIENT_SIGNATURES:
                for signature in self.GRAPHQL_CLIENT_SIGNATURES[client]:
                    if signature.lower() in header_lower or (
                        value and signature.lower() in value.lower()
                    ):
                        return True

        # Для POST-запросов проверяем содержимое
        if method.upper() == "POST" and body:
            body_str = str(body) if isinstance(body, (dict, list)) else body

            # Проверка на типичные поля GraphQL
            if isinstance(body, dict):
                if "query" in body and isinstance(body.get("query"), str):
                    return True
                if "operationName" in body:
                    return True

            # Проверка на пакетные запросы
            if isinstance(body, list) and len(body) > 0 and isinstance(body[0], dict):
                if "query" in body[0] or "operationName" in body[0]:
                    return True

            # Проверка на GraphQL синтаксис в теле
            for pattern in self.GRAPHQL_QUERY_PATTERNS:
                if re.search(pattern, body_str):
                    return True

        return False

    def _detect_graphql_client(self, headers: Dict[str, str], url: str, body: str) -> Optional[str]:
        """
        Определяет клиент GraphQL по заголовкам и URL.

        Args:
            headers: Заголовки запроса
            url: URL эндпоинта
            body: Тело запроса

        Returns:
            Optional[str]: Название клиента или None
        """
        # Проверка по URL
        url_lower = url.lower()
        if "apollo" in url_lower:
            return "apollo"
        if "relay" in url_lower:
            return "relay"
        if "urql" in url_lower:
            return "urql"

        # Проверка по заголовкам
        headers_str = json.dumps(headers).lower()
        for client, signatures in self.GRAPHQL_CLIENT_SIGNATURES.items():
            for signature in signatures:
                if signature.lower() in headers_str:
                    return client

        # Проверка по телу запроса
        for client, signatures in self.GRAPHQL_CLIENT_SIGNATURES.items():
            for signature in signatures:
                if signature.lower() in body.lower():
                    return client

        return None

    def _extract_operation_type(self, query: str) -> Optional[str]:
        """
        Извлекает тип операции из строки запроса GraphQL.

        Args:
            query: Запрос GraphQL

        Returns:
            Optional[str]: Тип операции ('query', 'mutation', 'subscription') или None
        """
        if not query:
            return None

        # Продвинутое регулярное выражение для определения типа операции
        match = re.search(r"^\s*(query|mutation|subscription)\b", query)
        if match:
            return match.group(1)

        # Проверка наличия фрагмента
        if re.search(r"\bfragment\s+\w+\s+on\b", query):
            return "fragment"

        # Если нет явного типа, это запрос по умолчанию
        if "{" in query:
            return "query"

        return None

    def _extract_graphql_from_string(self, body: str) -> Optional[Dict[str, Any]]:
        """
        Извлекает GraphQL-запрос из строки.

        Args:
            body: Строка с запросом

        Returns:
            Optional[Dict[str, Any]]: Извлеченный запрос или None
        """
        # Проверка на наличие GraphQL-запроса
        for pattern in self.GRAPHQL_QUERY_PATTERNS:
            match = re.search(pattern, body)
            if match:
                query = match.group(0)

                # Пытаемся извлечь имя операции
                operation_name_match = re.search(r"(query|mutation|subscription)\s+(\w+)", query)
                operation_name = operation_name_match.group(2) if operation_name_match else None

                # Пытаемся извлечь переменные
                variables_match = re.search(r"\(([$\w\s:]+)\)", query)
                variables = {}

                if variables_match:
                    args_str = variables_match.group(1)
                    args = args_str.split(",")

                    for arg in args:
                        arg = arg.strip()
                        if not arg:
                            continue

                        if ":" in arg:
                            name, value = arg.split(":", 1)
                            name = name.strip().lstrip("$")
                            value = value.strip()

                            # Конвертация значения
                            if value.lower() == "true":
                                variables[name] = True
                            elif value.lower() == "false":
                                variables[name] = False
                            elif value.isdigit():
                                variables[name] = int(value)
                            elif re.match(r"^\d+\.\d+$", value):
                                variables[name] = float(value)
                            else:
                                variables[name] = value

                return {"query": query, "operationName": operation_name, "variables": variables}

        return None

    def _extract_schema_info(self, query: str) -> None:
        """
        Извлекает информацию о схеме из запроса.

        Args:
            query: Запрос GraphQL
        """
        # Извлечение фрагментов
        fragment_matches = re.finditer(r"fragment\s+(\w+)\s+on\s+(\w+)\s*{([^}]*)}", query)
        for match in fragment_matches:
            fragment_name = match.group(1)
            type_name = match.group(2)
            fields = match.group(3).strip()

            # Сохраняем информацию о фрагменте
            self.schema_fragments[fragment_name] = {"type": type_name, "fields": fields}

        # Извлечение типов аргументов
        arg_matches = re.finditer(r"\(([$\w\s:]+)\)", query)
        for match in arg_matches:
            args_str = match.group(1)
            args = args_str.split(",")

            for arg in args:
                arg = arg.strip()
                if not arg:
                    continue

                if ":" in arg:
                    name, type_info = arg.split(":", 1)
                    name = name.strip().lstrip("$")
                    type_info = type_info.strip()

                    # Сохраняем информацию об аргументе
                    self.schema_fragments.setdefault("__arguments__", {})[name] = type_info

    def _extract_schema_from_response(self, response: Dict[str, Any]) -> None:
        """
        Извлекает информацию о схеме из ответа.

        Args:
            response: Ответ GraphQL
        """
        pass  # Будет реализовано позже

    def _extract_response_data(self, response: Dict[str, Any]) -> None:
        """
        Извлекает данные из ответа на основе зарегистрированных шаблонов.

        Args:
            response: Ответ GraphQL
        """
        if not response.get("data"):
            return

        # Применяем каждый шаблон
        for pattern in self.patterns:
            # Проверяем соответствие типу операции
            if pattern["operation_type"] and response.get("operation"):
                if response["operation"].get("operation_type") != pattern["operation_type"]:
                    continue

            # Проверяем соответствие имени операции
            if pattern["operation_name"] and response.get("operation"):
                if response["operation"].get("operation_name") != pattern["operation_name"]:
                    continue

            # Извлекаем данные по пути поля
            field_path = pattern["field_path"]
            extracted = self._extract_by_path(response["data"], field_path)

            if extracted:
                pattern_name = pattern["name"]
                response["extracted_data"][pattern_name] = extracted

    def _extract_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """
        Извлекает значение из структуры данных по пути.

        Args:
            data: Структура данных
            path: Путь в нотации с точками (например, 'users.items')

        Returns:
            Any: Извлеченное значение или None
        """
        if not data or not path:
            return None

        parts = path.split(".")
        current = data

        try:
            for part in parts:
                # Поддержка индексов массивов
                if "[" in part and "]" in part:
                    array_part, index_part = part.split("[")
                    index = int(index_part.replace("]", ""))

                    if array_part:
                        current = current.get(array_part)
                        if current is None:
                            return None

                    if isinstance(current, list) and 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        except Exception as e:
            self.logger.debug(f"Ошибка при извлечении по пути {path}: {str(e)}")
            return None

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
            "detected_clients": list(self.detected_clients),
            "operations_by_type": {
                op_type: len([op for op in self.operations if op.get("operation_type") == op_type])
                for op_type in self.operation_types
            },
            "operations_by_client": {
                client: len([op for op in self.operations if op.get("client_type") == client])
                for client in self.detected_clients
            },
            "successful_responses": len([r for r in self.responses if not r.get("errors")]),
            "error_responses": len([r for r in self.responses if r.get("errors")]),
            "schema_info": {
                "fragments_count": len(self.schema_fragments.get("__fragments__", {})),
                "types_count": len(self.schema_fragments.get("__types__", {})),
                "root_fields_count": len(self.schema_fragments.get("__root_fields__", {})),
            },
        }

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
                if "extracted_data" in response and pattern_name in response["extracted_data"]:
                    result.append(
                        {
                            "data": response["extracted_data"][pattern_name],
                            "operation": response.get("operation"),
                            "timestamp": response.get("timestamp"),
                        }
                    )
            else:
                if "extracted_data" in response:
                    for name, data in response["extracted_data"].items():
                        result.append(
                            {
                                "pattern": name,
                                "data": data,
                                "operation": response.get("operation"),
                                "timestamp": response.get("timestamp"),
                            }
                        )

        return result
