
"""
Расширенная версия UnifiedParser с интеграцией улучшенных JavaScript-компонентов.
"""

import logging
from typing import Dict, List, Optional, Any

from seo_ai_models.parsers.unified.unified_parser import UnifiedParser
from seo_ai_models.parsers.unified.js_integrator import JSIntegrator

logger = logging.getLogger(__name__)

class JSEnhancedUnifiedParser(UnifiedParser):
    """
    Расширенная версия UnifiedParser с поддержкой продвинутой обработки JavaScript.
    Добавляет поддержку WebSocket, GraphQL и клиентской маршрутизации.
    """
    
    def __init__(
        self,
        user_agent: str = "SEOAIModels JSEnhancedUnifiedParser/1.0",
        respect_robots: bool = True,
        delay: float = 1.0,
        max_pages: int = 100,
        search_engine: str = "google",
        spa_settings: Optional[Dict[str, Any]] = None,
        auto_detect_spa: bool = True,
        force_spa_mode: bool = False,
        extract_readability: bool = True,
        extract_semantic: bool = True,
        parallel_parsing: bool = True,
        max_workers: int = 5,
        search_api_keys: Optional[List[str]] = None,
        enable_websocket: bool = True,
        enable_graphql: bool = True,
        enable_client_routing: bool = True,
        emulate_user_behavior: bool = False,
        bypass_protection: bool = False
    ):
        """
        Инициализация расширенного парсера.
        
        Args:
            user_agent: User-Agent для запросов
            respect_robots: Уважать ли robots.txt
            delay: Задержка между запросами в секундах
            max_pages: Максимальное количество страниц для сканирования
            search_engine: Поисковая система для использования
            spa_settings: Настройки для SPA-парсера
            auto_detect_spa: Автоматически определять SPA-сайты
            force_spa_mode: Всегда использовать режим SPA
            extract_readability: Извлекать метрики читаемости
            extract_semantic: Извлекать семантические метрики
            parallel_parsing: Использовать многопоточный парсинг
            max_workers: Максимальное количество потоков для параллельного парсинга
            search_api_keys: API-ключи для поисковых систем
            enable_websocket: Включить анализ WebSocket
            enable_graphql: Включить анализ GraphQL
            enable_client_routing: Включить обработку клиентской маршрутизации
            emulate_user_behavior: Эмулировать поведение пользователя
            bypass_protection: Использовать методы обхода защиты от ботов
        """
        # Инициализация базового парсера
        super().__init__(
            user_agent=user_agent,
            respect_robots=respect_robots,
            delay=delay,
            max_pages=max_pages,
            search_engine=search_engine,
            spa_settings=spa_settings,
            auto_detect_spa=auto_detect_spa,
            force_spa_mode=force_spa_mode,
            extract_readability=extract_readability,
            extract_semantic=extract_semantic,
            parallel_parsing=parallel_parsing,
            max_workers=max_workers,
            search_api_keys=search_api_keys
        )
        
        # Инициализация JavaScript-интегратора
        self.js_integrator = JSIntegrator(
            enable_websocket=enable_websocket,
            enable_graphql=enable_graphql,
            enable_client_routing=enable_client_routing,
            emulate_user_behavior=emulate_user_behavior,
            bypass_protection=bypass_protection
        )
        
        # Параметры JavaScript-обработки
        self.enable_websocket = enable_websocket
        self.enable_graphql = enable_graphql
        self.enable_client_routing = enable_client_routing
        self.emulate_user_behavior = emulate_user_behavior
        self.bypass_protection = bypass_protection
        
        logger.info("JSEnhancedUnifiedParser инициализирован с расширенной обработкой JavaScript")
    
    def parse_url(self, url: str, **options) -> Dict[str, Any]:
        """
        Парсинг одного URL с расширенной обработкой JavaScript.
        
        Args:
            url: URL для парсинга
            **options: Дополнительные опции парсинга
            
        Returns:
            Dict[str, Any]: Результат парсинга
        """
        # Получаем опцию для выбора парсера
        use_js_enhanced = options.pop("use_js_enhanced", True)
        
        # Выбираем между расширенным и обычным парсером
        if use_js_enhanced and (self.force_spa_mode or self.auto_detect_spa):
            logger.info(f"Используем расширенный JS-парсер для {url}")
            
            # Создаем опции для краулера
            crawler_options = {
                "max_pages": 1,
                "max_depth": 0,
                "delay": self.delay,
                "wait_for_timeout": self.spa_settings.get("wait_for_timeout", 10000) if self.spa_settings else 10000,
                "headless": self.spa_settings.get("headless", True) if self.spa_settings else True
            }
            
            # Добавляем дополнительные опции
            crawler_options.update(options)
            
            # Выполняем парсинг через JavaScript-интегратор
            result = self.js_integrator.parse_site(url, **crawler_options)
            
            # Преобразуем результат в формат базового парсера
            return self._convert_js_result_to_base_result(result, url)
        else:
            # Используем базовый парсер
            logger.info(f"Используем базовый парсер для {url}")
            return super().parse_url(url, **options)
    
    def crawl_site(self, base_url: str, **options) -> Dict[str, Any]:
        """
        Сканирование сайта с расширенной обработкой JavaScript.
        
        Args:
            base_url: Начальный URL для сканирования
            **options: Дополнительные опции сканирования
            
        Returns:
            Dict[str, Any]: Результаты сканирования
        """
        # Получаем опцию для выбора парсера
        use_js_enhanced = options.pop("use_js_enhanced", True)
        
        # Выбираем между расширенным и обычным парсером
        if use_js_enhanced and (self.force_spa_mode or self.auto_detect_spa):
            logger.info(f"Используем расширенный JS-парсер для сканирования {base_url}")
            
            # Создаем опции для краулера
            crawler_options = {
                "max_pages": options.get("max_pages", self.max_pages),
                "max_depth": options.get("max_depth", 3),
                "delay": options.get("delay", self.delay),
                "wait_for_timeout": self.spa_settings.get("wait_for_timeout", 10000) if self.spa_settings else 10000,
                "headless": self.spa_settings.get("headless", True) if self.spa_settings else True
            }
            
            # Добавляем дополнительные опции
            for key, value in options.items():
                if key not in crawler_options:
                    crawler_options[key] = value
            
            # Выполняем парсинг через JavaScript-интегратор
            result = self.js_integrator.parse_site(base_url, **crawler_options)
            
            # Преобразуем результат в формат базового парсера
            return self._convert_js_result_to_site_result(result, base_url)
        else:
            # Используем базовый парсер
            logger.info(f"Используем базовый парсер для сканирования {base_url}")
            return super().crawl_site(base_url, **options)
    
    def _convert_js_result_to_base_result(self, js_result: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Преобразует результат JavaScript-парсера в формат базового парсера.
        
        Args:
            js_result: Результат JavaScript-парсера
            url: URL страницы
            
        Returns:
            Dict[str, Any]: Результат в формате базового парсера
        """
        # Начинаем с базовой структуры
        result = {
            "success": True,
            "processing_time": js_result.get("duration", 0),
            "page_data": {
                "url": url,
                "content": {},
                "structure": {},
                "metadata": {},
                "html_stats": {},
                "performance": {}
            }
        }
        
        # Добавляем данные о контенте, если доступны
        if url in js_result.get("page_contents", {}):
            page_content = js_result["page_contents"][url]
            
            # Добавляем базовую информацию
            result["page_data"]["html_stats"]["html_size"] = len(page_content.get("html", ""))
            result["page_data"]["structure"]["title"] = page_content.get("title", "")
            
            # TODO: Извлечь больше информации из HTML
        
        # Добавляем данные о JavaScript-технологиях
        js_data = self.js_integrator.get_combined_results(js_result)
        
        if "technologies" in js_data:
            result["page_data"]["performance"]["javascript_technologies"] = js_data["technologies"]
            
        # Добавляем данные о клиентской маршрутизации
        if "routing" in js_result:
            routing_data = js_result["routing"]
            result["page_data"]["performance"]["client_routing"] = {
                "detected": True,
                "framework": routing_data.get("statistics", {}).get("framework", {}).get("name"),
                "route_changes": routing_data.get("statistics", {}).get("total_changes", 0),
                "unique_routes": routing_data.get("statistics", {}).get("unique_routes", 0)
            }
            
        # Добавляем данные о WebSocket
        if "websocket" in js_result:
            ws_data = js_result["websocket"]
            result["page_data"]["performance"]["websocket"] = {
                "detected": ws_data.get("statistics", {}).get("total_messages", 0) > 0,
                "connections": len(ws_data.get("connections", [])),
                "message_count": ws_data.get("statistics", {}).get("total_messages", 0),
                "protocols": list(ws_data.get("statistics", {}).get("protocol_stats", {}).keys())
            }
            
        # Добавляем данные о GraphQL
        if "graphql" in js_result:
            gql_data = js_result["graphql"]
            result["page_data"]["performance"]["graphql"] = {
                "detected": gql_data.get("statistics", {}).get("total_operations", 0) > 0,
                "operations_count": gql_data.get("statistics", {}).get("total_operations", 0),
                "clients": gql_data.get("statistics", {}).get("detected_clients", [])
            }
        
        return result
    
    def _convert_js_result_to_site_result(self, js_result: Dict[str, Any], base_url: str) -> Dict[str, Any]:
        """
        Преобразует результат JavaScript-парсера в формат результата сканирования сайта.
        
        Args:
            js_result: Результат JavaScript-парсера
            base_url: Базовый URL сайта
            
        Returns:
            Dict[str, Any]: Результат в формате базового парсера
        """
        # Начинаем с базовой структуры
        result = {
            "success": True,
            "processing_time": js_result.get("duration", 0),
            "site_data": {
                "base_url": base_url,
                "pages": {},
                "crawl_info": {
                    "start_time": js_result.get("timestamp", 0) - js_result.get("duration", 0),
                    "end_time": js_result.get("timestamp", 0),
                    "urls_found": len(js_result.get("found_urls", [])),
                    "urls_crawled": len(js_result.get("crawled_urls", [])),
                    "urls_failed": len(js_result.get("failed_urls", []))
                },
                "statistics": {},
                "site_structure": {},
                "javascript_technologies": {}
            }
        }
        
        # Добавляем данные о страницах
        for url in js_result.get("crawled_urls", []):
            # Преобразуем результат страницы
            page_result = self._convert_js_result_to_base_result(js_result, url)
            
            # Добавляем страницу в результат сайта
            if "page_data" in page_result:
                result["site_data"]["pages"][url] = page_result["page_data"]
        
        # Добавляем данные о JavaScript-технологиях
        js_data = self.js_integrator.get_combined_results(js_result)
        
        if "technologies" in js_data:
            result["site_data"]["javascript_technologies"] = js_data["technologies"]
            
        # Добавляем сводную информацию о клиентской маршрутизации
        if "routing" in js_result:
            routing_data = js_result["routing"]
            result["site_data"]["statistics"]["client_routing"] = {
                "detected": True,
                "framework": routing_data.get("statistics", {}).get("framework", {}).get("name"),
                "route_changes": routing_data.get("statistics", {}).get("total_changes", 0),
                "unique_routes": len(routing_data.get("unique_routes", []))
            }
            
        # Добавляем сводную информацию о WebSocket
        if "websocket" in js_result:
            ws_data = js_result["websocket"]
            result["site_data"]["statistics"]["websocket"] = {
                "detected": ws_data.get("statistics", {}).get("total_messages", 0) > 0,
                "connections": len(ws_data.get("connections", [])),
                "message_count": ws_data.get("statistics", {}).get("total_messages", 0),
                "protocols": list(ws_data.get("statistics", {}).get("protocol_stats", {}).keys())
            }
            
        # Добавляем сводную информацию о GraphQL
        if "graphql" in js_result:
            gql_data = js_result["graphql"]
            result["site_data"]["statistics"]["graphql"] = {
                "detected": gql_data.get("statistics", {}).get("total_operations", 0) > 0,
                "operations_count": gql_data.get("statistics", {}).get("total_operations", 0),
                "clients": gql_data.get("statistics", {}).get("detected_clients", [])
            }
        
        return result
