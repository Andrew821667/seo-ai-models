"""
Обновление AdaptiveParsingPipeline с поддержкой кэширования.
"""

from seo_ai_models.parsers.adaptive_parsing_pipeline import AdaptiveParsingPipeline
from seo_ai_models.parsers.utils.cache_manager import CacheManager

def add_caching_support(pipeline_class):
    """
    Добавляет поддержку кэширования в AdaptiveParsingPipeline.
    
    Args:
        pipeline_class: Класс AdaptiveParsingPipeline
        
    Returns:
        Обновленный класс с поддержкой кэширования
    """
    original_init = pipeline_class.__init__
    original_analyze_url = pipeline_class.analyze_url
    original_detect_site_type = pipeline_class.detect_site_type
    
    def __init_with_cache(self, *args, **kwargs):
        # Извлекаем параметры кэширования
        cache_dir = kwargs.pop('cache_dir', '.cache')
        cache_max_age = kwargs.pop('cache_max_age', 86400)
        cache_enabled = kwargs.pop('cache_enabled', True)
        
        # Вызываем оригинальный конструктор
        original_init(self, *args, **kwargs)
        
        # Инициализируем кэш-менеджер
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_age=cache_max_age,
            enabled=cache_enabled
        )
    
    def analyze_url_with_cache(self, url, detect_type=True, use_cache=True):
        """
        Анализ URL с поддержкой кэширования.
        
        Args:
            url: URL для анализа
            detect_type: Определять ли тип сайта
            use_cache: Использовать ли кэш
            
        Returns:
            Dict: Результаты анализа
        """
        if use_cache:
            # Параметры для кэша
            cache_params = {
                'detect_type': detect_type,
                'force_spa_mode': self.force_spa_mode
            }
            
            # Попытка получить из кэша
            cached_data, is_valid = self.cache_manager.get(url, cache_params)
            if cached_data and is_valid:
                return cached_data
        
        # Если кэш не используется или данных нет, выполняем обычный анализ
        result = original_analyze_url(self, url, detect_type)
        
        # Сохраняем результат в кэш, если он успешный
        if use_cache and result.get('success', False):
            cache_params = {
                'detect_type': detect_type,
                'force_spa_mode': self.force_spa_mode
            }
            self.cache_manager.set(url, result, cache_params)
        
        return result
    
    def detect_site_type_with_cache(self, url, use_cache=True):
        """
        Определение типа сайта с поддержкой кэширования.
        
        Args:
            url: URL для анализа
            use_cache: Использовать ли кэш
            
        Returns:
            Dict: Информация о типе сайта
        """
        if use_cache:
            # Параметры для кэша
            cache_params = {
                'force_spa_mode': self.force_spa_mode
            }
            
            # Попытка получить из кэша
            cached_data, is_valid = self.cache_manager.get(f"sitetype:{url}", cache_params)
            if cached_data and is_valid:
                return cached_data
        
        # Если кэш не используется или данных нет, определяем тип сайта
        result = original_detect_site_type(self, url)
        
        # Сохраняем результат в кэш
        if use_cache:
            cache_params = {
                'force_spa_mode': self.force_spa_mode
            }
            self.cache_manager.set(f"sitetype:{url}", result, cache_params)
        
        return result
    
    # Заменяем методы в классе
    pipeline_class.__init__ = __init_with_cache
    pipeline_class.analyze_url = analyze_url_with_cache
    pipeline_class.detect_site_type = detect_site_type_with_cache
    
    # Добавляем метод для управления кэшем
    def clear_cache(self, url=None):
        """
        Очищает кэш для URL или весь кэш.
        
        Args:
            url: URL для удаления из кэша (если None, очищает весь кэш)
            
        Returns:
            bool: True при успешной очистке
        """
        return self.cache_manager.clear(url)
    
    pipeline_class.clear_cache = clear_cache
    
    return pipeline_class
