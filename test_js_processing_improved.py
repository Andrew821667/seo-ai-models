
"""
Тест улучшенных компонентов JavaScript-обработки.
"""

import sys
import os
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("test_js_processing")

# Добавляем корневую директорию проекта в путь импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем улучшенные компоненты
try:
    from seo_ai_models.parsers.unified.js_processing.enhanced_graphql_interceptor import EnhancedGraphQLInterceptor
    from seo_ai_models.parsers.unified.js_processing.enhanced_websocket_analyzer import EnhancedWebSocketAnalyzer, WebSocketMessage
except ImportError as e:
    logger.error(f"Ошибка импорта: {e}")
    sys.exit(1)

def test_graphql_detector():
    """Тестирует улучшенный детектор GraphQL"""
    logger.info("Тестирование улучшенного детектора GraphQL")
    
    interceptor = EnhancedGraphQLInterceptor()
    
    # Тестовые данные
    test_cases = [
        # Явный GraphQL запрос через URL
        {
            "url": "https://example.com/graphql",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": '{"query": "query { items { id name } }"}'
        },
        # GraphQL клиент Apollo в заголовках
        {
            "url": "https://api.example.com/v1",
            "method": "POST",
            "headers": {"apollographql-client-name": "web-app"},
            "body": '{"operationName":"GetItems","variables":{},"query":"query GetItems { items { id name } }"}'
        },
        # GraphQL запрос с фрагментом
        {
            "url": "https://api.example.com/data",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": '{"query": "query { items { ...ItemFields } } fragment ItemFields on Item { id name }"}'
        }
    ]
    
    # Проверка обнаружения GraphQL
    for i, test_case in enumerate(test_cases):
        operations = interceptor.intercept_request(
            test_case["url"], 
            test_case["headers"],
            test_case["body"],
            test_case["method"]
        )
        
        logger.info(f"Тест #{i+1}: {len(operations)} операций обнаружено")

def test_websocket_analyzer():
    """Тестирует улучшенный анализатор WebSocket"""
    logger.info("Тестирование улучшенного анализатора WebSocket")
    
    analyzer = EnhancedWebSocketAnalyzer()
    
    # Тестовые данные
    test_messages = [
        # Socket.IO
        {
            "content": '2{"data":["message",{"text":"Hello World","user":"test"}]}',
            "endpoint": "wss://example.com/socket.io/",
            "direction": "incoming"
        },
        # SignalR
        {
            "content": '{"type":1,"target":"ChatHub","arguments":["Hello World"]}',
            "endpoint": "wss://example.com/signalr",
            "direction": "outgoing"
        },
        # GraphQL WS
        {
            "content": '{"type":"next","id":"1","payload":{"data":{"users":[{"id":"1","name":"User 1"}]}}}',
            "endpoint": "wss://example.com/graphql",
            "direction": "incoming"
        }
    ]
    
    # Регистрация шаблонов
    analyzer.register_pattern(
        name="chat_messages",
        key_fields=["type", "target"],
        value_fields=["arguments"],
        protocol="signalr"
    )
    
    # Проверка обнаружения протоколов
    for i, msg in enumerate(test_messages):
        message = analyzer.capture_message(
            msg["content"],
            msg["direction"],
            msg["endpoint"]
        )
        
        if message:
            logger.info(f"Тест #{i+1}: Протокол: {message.protocol}, Тип: {message.message_type}")
            if message.extracted_data:
                logger.info(f"  Извлеченные данные: {json.dumps(message.extracted_data, ensure_ascii=False)}")

if __name__ == "__main__":
    test_graphql_detector()
    test_websocket_analyzer()
