"""
Обновление для ParsePipeline с добавлением поддержки SPA-сайтов.
Этот файл содержит функции для обновления существующего ParsePipeline.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Union

from seo_ai_models.parsers.adaptive_parsing_pipeline import AdaptiveParsingPipeline
from seo_ai_models.parsers.utils.spa_detector import SPADetector
from seo_ai_models.parsers.utils.request_utils import fetch_url_with_javascript_sync

logger = logging.getLogger(__name__)

def integrate_spa_support(pipeline_obj):
    """
    Интегрирует поддержку SPA в существующий объект ParsePipeline.
    
    Args:
        pipeline_obj: Экземпляр ParsePipeline для обновления
        
    Returns:
        Обновленный экземпляр ParsePipeline с поддержкой SPA
    """
    # Добавление необходимых атрибутов и методов
    pipeline_obj.spa_detector = SPADetector()
    pipeline_obj.support_spa = True
    pipeline_obj.force_spa_mode = False
    pipeline_obj.wait_for_idle = 2000  # мс
    pipeline_obj.wait_for_timeout = 10000  # мс
    pipeline_obj.headless = True
    pipeline_obj.browser_type = "chromium"
    
    # Оригинальный метод analyze_url
    original_analyze_url = pipeline_obj.analyze_url
    
    # Новый метод analyze_url с поддержкой SPA
    def analyze_url_with_spa(url, **kwargs):
        """
        Обертка для analyze_url с определением типа сайта.
        """
        # Определение, нужен ли режим SPA
        use_spa = kwargs.pop('use_spa', None)
        
        if use_spa is None and not pipeline_obj.force_spa_mode:
            # Определяем тип сайта автоматически
            site_type = detect_site_type(pipeline_obj, url)
            use_spa = site_type.get("is_spa", False)
        elif pipeline_obj.force_spa_mode:
            use_spa = True
            
        if use_spa:
            # Используем AdaptiveParsingPipeline для SPA-сайтов
            adaptive_pipeline = AdaptiveParsingPipeline(
                user_agent=pipeline_obj.user_agent,
                respect_robots=pipeline_obj.respect_robots,
                delay=pipeline_obj.delay,
                max_pages=pipeline_obj.max_pages,
                wait_for_idle=pipeline_obj.wait_for_idle,
                wait_for_timeout=pipeline_obj.wait_for_timeout,
                headless=pipeline_obj.headless,
                browser_type=pipeline_obj.browser_type,
                force_spa_mode=True
            )
            result = adaptive_pipeline.analyze_url(url, detect_type=False)
            return result
        else:
            # Используем оригинальный метод для обычных сайтов
            if 'delay' in kwargs: kwargs.pop('delay')
            print(f'DEBUG: kwargs = {kwargs}')
            # Удаляем delay из kwargs, если он есть
            kwargs_copy = kwargs.copy()
            if 'delay' in kwargs_copy:
                kwargs_copy.pop('delay')
            return original_analyze_url(url, **kwargs_copy)
            
    # Функция для определения типа сайта
    def detect_site_type(pipeline, url):
        """
        Определяет тип сайта (обычный или SPA).
        """
        logger.info(f"Detecting site type for {url}")
        
        # Создаем временный AdaptiveParsingPipeline
        adaptive_pipeline = AdaptiveParsingPipeline(
            user_agent=pipeline.user_agent,
            respect_robots=pipeline.respect_robots,
            max_pages=pipeline.max_pages,
            wait_for_idle=pipeline.wait_for_idle,
            wait_for_timeout=pipeline.wait_for_timeout,
            headless=pipeline.headless,
            browser_type=pipeline.browser_type
        )
        
        return adaptive_pipeline.detect_site_type(url)
    
    # Добавляем новые методы
    pipeline_obj.analyze_url_with_spa = analyze_url_with_spa
    pipeline_obj.detect_site_type = lambda url: detect_site_type(pipeline_obj, url)
    
    # Заменяем оригинальный метод analyze_url
    pipeline_obj.original_analyze_url = original_analyze_url
    pipeline_obj.analyze_url = analyze_url_with_spa
    
    # Добавляем новый метод для установки режима SPA
    def set_spa_mode(force_spa_mode=False):
        """
        Устанавливает режим SPA для парсера.
        
        Args:
            force_spa_mode: Принудительно использовать режим SPA
        """
        pipeline_obj.force_spa_mode = force_spa_mode
        
    pipeline_obj.set_spa_mode = set_spa_mode
    
    # Добавляем метод для настройки параметров SPA
    def configure_spa_options(
        wait_for_idle=None,
        wait_for_timeout=None,
        headless=None,
        browser_type=None
    ):
        """
        Настраивает параметры SPA-рендеринга.
        
        Args:
            wait_for_idle: Время ожидания в мс после событий 'networkidle'
            wait_for_timeout: Максимальное время ожидания в мс
            headless: Запускать ли браузер без интерфейса
            browser_type: Тип браузера ('chromium', 'firefox', 'webkit')
        """
        if wait_for_idle is not None:
            pipeline_obj.wait_for_idle = wait_for_idle
            
        if wait_for_timeout is not None:
            pipeline_obj.wait_for_timeout = wait_for_timeout
            
        if headless is not None:
            pipeline_obj.headless = headless
            
        if browser_type is not None and browser_type in ["chromium", "firefox", "webkit"]:
            pipeline_obj.browser_type = browser_type
    
    pipeline_obj.configure_spa_options = configure_spa_options
    
    return pipeline_obj
