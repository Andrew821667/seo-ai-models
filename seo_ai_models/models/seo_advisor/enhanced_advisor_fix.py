
"""
Исправление для инициализации EnhancedSEOAdvisor.
"""

from seo_ai_models.models.seo_advisor.enhanced_advisor import EnhancedSEOAdvisor

# Сохраняем оригинальный метод инициализации
original_init = EnhancedSEOAdvisor.__init__

# Определяем исправленный метод инициализации
def fixed_init(self, llm_api_key=None, **kwargs):
    """
    Исправленная инициализация EnhancedSEOAdvisor с правильной передачей параметров базовому классу.
    
    Args:
        llm_api_key: API ключ для LLM-сервисов (опционально)
        **kwargs: Дополнительные параметры
    """
    # Извлекаем компоненты, но не передаем их напрямую в super().__init__
    content_analyzer = kwargs.pop('content_analyzer', None)
    semantic_analyzer = kwargs.pop('semantic_analyzer', None)
    eeat_analyzer = kwargs.pop('eeat_analyzer', None)
    predictor = kwargs.pop('predictor', None)
    suggester = kwargs.pop('suggester', None)
    
    # Вызываем базовый конструктор без компонентов
    super(EnhancedSEOAdvisor, self).__init__(**kwargs)
    
    # Сохраняем API ключ и инициализируем фабрику, если ключ предоставлен
    self.llm_factory = None
    self.llm_components = {}
    self.llm_api_key = llm_api_key
    
    # Устанавливаем компоненты, если они предоставлены
    if content_analyzer:
        self.content_analyzer = content_analyzer
    if semantic_analyzer:
        self.semantic_analyzer = semantic_analyzer
    if eeat_analyzer:
        self.eeat_analyzer = eeat_analyzer
    if predictor:
        self.predictor = predictor
    if suggester:
        self.suggester = suggester
    
    # Инициализируем фабрику LLM, если API ключ предоставлен
    if llm_api_key:
        try:
            from seo_ai_models.models.llm_integration.service.llm_service import LLMService
            from seo_ai_models.models.seo_advisor.llm_integration_adapter import LLMAdvisorFactory
            
            self.llm_factory = LLMAdvisorFactory(llm_api_key)
            self._initialize_llm_components()
        except ImportError as e:
            print(f"Модуль LLM-интеграции не найден: {e}")
            self.llm_factory = None
    
    # Флаг доступности LLM-функций
    self.llm_enabled = bool(self.llm_factory and self.llm_components)
    
    # Настройка логгирования
    import logging
    self.logger = logging.getLogger(__name__)
    
    if self.llm_enabled:
        self.logger.info("LLM-компоненты успешно инициализированы")
    else:
        self.logger.warning("LLM-компоненты недоступны")

# Заменяем метод инициализации
EnhancedSEOAdvisor.__init__ = fixed_init
