
"""
Интегратор улучшенных JavaScript-компонентов для проекта SEO AI Models.
Объединяет все новые компоненты для удобного использования в парсере.
"""

import logging
from typing import Dict, List, Set, Optional, Any

from seo_ai_models.parsers.unified.js_processing.enhanced_websocket_analyzer import EnhancedWebSocketAnalyzer
from seo_ai_models.parsers.unified.js_processing.enhanced_graphql_interceptor import EnhancedGraphQLInterceptor
from seo_ai_models.parsers.unified.js_processing.client_routing_handler import ClientRoutingHandler
from seo_ai_models.parsers.unified.protection_bypass import BrowserFingerprint
from seo_ai_models.parsers.unified.crawlers.improved_spa_crawler import ImprovedSPACrawler

logger = logging.getLogger(__name__)

class JSIntegrator:
    """
    Интегратор для улучшенных JavaScript-компонентов.
    Предоставляет единый интерфейс для всех улучшенных возможностей.
    """
    
    def __init__(
        self,
        enable_websocket: bool = True,
        enable_graphql: bool = True,
        enable_client_routing: bool = True,
        emulate_user_behavior: bool = False,
        bypass_protection: bool = False,
        log_level: int = logging.INFO
    ):
        """
        Инициализация интегратора.
        
        Args:
            enable_websocket: Включить анализ WebSocket
            enable_graphql: Включить анализ GraphQL
            enable_client_routing: Включить обработку клиентской маршрутизации
            emulate_user_behavior: Эмулировать поведение пользователя
            bypass_protection: Использовать методы обхода защиты от ботов
            log_level: Уровень логирования
        """
        self.logger = logging.getLogger("JSIntegrator")
        self.logger.setLevel(log_level)
        
        self.enable_websocket = enable_websocket
        self.enable_graphql = enable_graphql
        self.enable_client_routing = enable_client_routing
        self.emulate_user_behavior = emulate_user_behavior
        self.bypass_protection = bypass_protection
        
        # Инициализация компонентов
        if self.enable_websocket:
            self.websocket_analyzer = EnhancedWebSocketAnalyzer(log_level)
            
        if self.enable_graphql:
            self.graphql_interceptor = EnhancedGraphQLInterceptor(log_level)
            
        if self.enable_client_routing:
            self.routing_handler = ClientRoutingHandler(log_level)
            
        if self.bypass_protection:
            self.fingerprint_generator = BrowserFingerprint(consistent=False)
            
        self.logger.info("Интегратор JavaScript-компонентов инициализирован")
    
    def create_crawler(self, base_url: str, **options) -> ImprovedSPACrawler:
        """
        Создает улучшенный SPA-краулер с текущими настройками.
        
        Args:
            base_url: Базовый URL для сканирования
            **options: Дополнительные опции для краулера
            
        Returns:
            ImprovedSPACrawler: Настроенный краулер
        """
        crawler = ImprovedSPACrawler(
            base_url=base_url,
            enable_websocket=self.enable_websocket,
            enable_graphql=self.enable_graphql,
            enable_client_routing=self.enable_client_routing,
            emulate_user_behavior=self.emulate_user_behavior,
            bypass_protection=self.bypass_protection,
            **options
        )
        
        self.logger.info(f"Создан улучшенный SPA-краулер для {base_url}")
        return crawler
    
    def parse_site(self, url: str, **options) -> Dict[str, Any]:
        """
        Парсит сайт с использованием улучшенного SPA-краулера.
        
        Args:
            url: URL для парсинга
            **options: Дополнительные опции
            
        Returns:
            Dict[str, Any]: Результаты парсинга
        """
        crawler = self.create_crawler(url, **options)
        result = crawler.crawl()
        
        self.logger.info(f"Завершен парсинг сайта {url}")
        return result
    
    def get_combined_results(self, parsing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Возвращает объединенные результаты анализа JavaScript.
        
        Args:
            parsing_result: Результат парсинга
            
        Returns:
            Dict[str, Any]: Объединенные результаты
        """
        results = {
            "summary": {
                "websocket_detected": "websocket" in parsing_result and parsing_result["websocket"]["statistics"]["total_messages"] > 0,
                "graphql_detected": "graphql" in parsing_result and parsing_result["graphql"]["statistics"]["total_operations"] > 0,
                "client_routing_detected": "routing" in parsing_result and parsing_result["routing"]["statistics"]["total_changes"] > 0
            }
        }
        
        # Добавляем детальные результаты
        if "websocket" in parsing_result:
            results["websocket"] = parsing_result["websocket"]
            
        if "graphql" in parsing_result:
            results["graphql"] = parsing_result["graphql"]
            
        if "routing" in parsing_result:
            results["routing"] = parsing_result["routing"]
            
        # Добавляем информацию о технологиях
        technologies = []
        
        if "routing" in parsing_result and parsing_result["routing"]["statistics"].get("framework"):
            framework = parsing_result["routing"]["statistics"]["framework"]
            technologies.append({
                "name": framework.get("name", "Unknown Framework"),
                "type": "client_routing",
                "confidence": framework.get("confidence", 0.5)
            })
            
        if "websocket" in parsing_result and parsing_result["websocket"]["statistics"].get("protocol_stats"):
            for protocol, count in parsing_result["websocket"]["statistics"]["protocol_stats"].items():
                if protocol != "raw" and count > 0:
                    technologies.append({
                        "name": protocol,
                        "type": "websocket_protocol",
                        "count": count
                    })
                    
        if "graphql" in parsing_result and parsing_result["graphql"]["statistics"].get("detected_clients"):
            for client in parsing_result["graphql"]["statistics"]["detected_clients"]:
                technologies.append({
                    "name": client,
                    "type": "graphql_client"
                })
                
        results["technologies"] = technologies
        
        return results
