"""
Интеграция расширенной обработки JavaScript с унифицированным парсером.
"""

import logging
from typing import Dict, List, Optional, Any

from seo_ai_models.parsers.unified.js_processing.websocket_analyzer import WebSocketAnalyzer
from seo_ai_models.parsers.unified.js_processing.graphql_interceptor import GraphQLInterceptor
from seo_ai_models.parsers.unified.js_processing.client_routing_handler import ClientRoutingHandler
from seo_ai_models.parsers.unified.parser_result import ParserResult, PageData

logger = logging.getLogger(__name__)

def enhance_parser_result(result: ParserResult, js_data: Dict[str, Any]) -> ParserResult:
    """
    Улучшает результат парсера данными расширенной обработки JavaScript.
    
    Args:
        result: Результат парсера
        js_data: Данные из JS-обработки
        
    Returns:
        ParserResult: Улучшенный результат парсера
    """
    if not result or not js_data:
        return result
    
    # Копируем метаданные и статистику
    if not hasattr(result, 'metadata'):
        result.metadata = {}
    
    # Добавляем данные WebSocket
    if 'websocket' in js_data:
        result.metadata['websocket'] = {
            'active_connections': len(js_data['websocket'].get('active_connections', [])),
            'message_count': js_data['websocket'].get('statistics', {}).get('total_messages', 0),
            'has_data': bool(js_data['websocket'].get('extracted_data'))
        }
        
        # Сохраняем детальные данные
        result.websocket_data = js_data['websocket']
    
    # Добавляем данные GraphQL
    if 'graphql' in js_data:
        result.metadata['graphql'] = {
            'operations_count': js_data['graphql'].get('statistics', {}).get('total_operations', 0),
            'response_count': js_data['graphql'].get('statistics', {}).get('total_responses', 0),
            'has_data': bool(js_data['graphql'].get('extracted_data'))
        }
        
        # Сохраняем детальные данные
        result.graphql_data = js_data['graphql']
    
    # Добавляем данные клиентской маршрутизации
    if 'routing' in js_data:
        result.metadata['routing'] = {
            'route_changes': js_data['routing'].get('statistics', {}).get('total_changes', 0),
            'unique_routes': js_data['routing'].get('statistics', {}).get('unique_routes', 0),
            'framework': js_data['routing'].get('statistics', {}).get('framework', {}).get('name')
        }
        
        # Сохраняем детальные данные
        result.routing_data = js_data['routing']
    
    # Добавляем флаг расширенной JS-обработки
    result.metadata['has_enhanced_js_processing'] = True
    
    return result

def enhance_page_data(page_data: PageData, js_data: Dict[str, Any], url: str) -> PageData:
    """
    Улучшает данные страницы информацией из расширенной обработки JavaScript.
    
    Args:
        page_data: Данные страницы
        js_data: Данные из JS-обработки
        url: URL страницы
        
    Returns:
        PageData: Улучшенные данные страницы
    """
    if not page_data or not js_data:
        return page_data
    
    # Добавляем данные обработки JavaScript в performance
    if not hasattr(page_data, 'performance'):
        page_data.performance = {}
    
    page_data.performance['js_processing'] = {
        'has_websocket': 'websocket' in js_data,
        'has_graphql': 'graphql' in js_data,
        'has_client_routing': 'routing' in js_data
    }
    
    # Добавляем дополнительные данные для URLs из client routing
    if 'routing' in js_data and 'unique_routes' in js_data['routing']:
        client_routes = []
        
        for route in js_data['routing']['unique_routes']:
            if route.get('url') and route.get('route_info'):
                client_routes.append({
                    'url': route['url'],
                    'path': route.get('path', ''),
                    'name': route.get('route_info', {}).get('name', ''),
                    'params': route.get('route_info', {}).get('params', {})
                })
        
        if client_routes:
            page_data.performance['client_routes'] = client_routes
    
    # Добавляем данные о GraphQL-операциях, относящихся к этой странице
    if 'graphql' in js_data and 'extracted_data' in js_data['graphql']:
        graphql_data = []
        
        for item in js_data['graphql']['extracted_data']:
            if 'operation' in item and item['operation'] and 'endpoint' in item['operation']:
                endpoint = item['operation']['endpoint']
                # Проверяем, относится ли операция к текущей странице
                if url in endpoint or endpoint in url:
                    graphql_data.append({
                        'pattern': item.get('pattern', ''),
                        'operation_name': item['operation'].get('operation_name', ''),
                        'operation_type': item['operation'].get('operation_type', '')
                    })
        
        if graphql_data:
            page_data.performance['graphql_operations'] = graphql_data
    
    return page_data

def extract_urls_from_client_routing(js_data: Dict[str, Any]) -> List[str]:
    """
    Извлекает дополнительные URL из данных клиентской маршрутизации.
    
    Args:
        js_data: Данные из JS-обработки
        
    Returns:
        List[str]: Список URL из клиентской маршрутизации
    """
    urls = []
    
    if 'routing' in js_data and 'unique_routes' in js_data['routing']:
        for route in js_data['routing']['unique_routes']:
            if route.get('url'):
                urls.append(route['url'])
    
    return urls
