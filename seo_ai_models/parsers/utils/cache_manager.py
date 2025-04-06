"""
Модуль для кэширования результатов парсинга.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Управляет кэшированием результатов парсинга для улучшения производительности.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        max_age: int = 86400,  # 24 часа в секундах
        enabled: bool = True
    ):
        """
        Инициализация CacheManager.

        Args:
            cache_dir: Директория для хранения кэша
            max_age: Максимальный возраст кэша в секундах
            enabled: Включено ли кэширование
        """
        self.cache_dir = cache_dir
        self.max_age = max_age
        self.enabled = enabled
        
        if self.enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, url: str, params: Dict = None) -> str:
        """
        Создает ключ кэша для URL и параметров.
        
        Args:
            url: URL для кэширования
            params: Дополнительные параметры, влияющие на содержимое
            
        Returns:
            str: Ключ кэша
        """
        # Создаем хэш на основе URL и параметров
        key_data = url
        if params:
            key_data += json.dumps(params, sort_keys=True)
            
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Получает путь к файлу кэша.
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            str: Путь к файлу кэша
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, url: str, params: Dict = None) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Получает данные из кэша.
        
        Args:
            url: URL для кэширования
            params: Дополнительные параметры
            
        Returns:
            Tuple[Optional[Dict[str, Any]], bool]: 
                - Данные из кэша (None, если не найдены)
                - Флаг, указывающий, актуален ли кэш
        """
        if not self.enabled:
            return None, False
        
        cache_key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None, False
        
        # Проверяем возраст кэша
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age > self.max_age:
            logger.info(f"Кэш для {url} устарел ({file_age:.0f} с > {self.max_age} с)")
            return None, False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Загружены данные из кэша для {url}")
                return data, True
        except Exception as e:
            logger.error(f"Ошибка при загрузке кэша для {url}: {str(e)}")
            return None, False
    
    def set(self, url: str, data: Dict[str, Any], params: Dict = None) -> bool:
        """
        Сохраняет данные в кэш.
        
        Args:
            url: URL для кэширования
            data: Данные для сохранения
            params: Дополнительные параметры
            
        Returns:
            bool: True при успешном сохранении
        """
        if not self.enabled:
            return False
        
        cache_key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Данные для {url} сохранены в кэш")
                return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении кэша для {url}: {str(e)}")
            return False
    
    def clear(self, url: str = None, params: Dict = None) -> bool:
        """
        Очищает кэш для URL или весь кэш.
        
        Args:
            url: URL для удаления из кэша (если None, очищает весь кэш)
            params: Дополнительные параметры
            
        Returns:
            bool: True при успешной очистке
        """
        if not self.enabled:
            return False
        
        if url:
            # Очистка кэша для конкретного URL
            cache_key = self._get_cache_key(url, params)
            cache_path = self._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    logger.info(f"Кэш для {url} очищен")
                    return True
                except Exception as e:
                    logger.error(f"Ошибка при очистке кэша для {url}: {str(e)}")
                    return False
        else:
            # Очистка всего кэша
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path) and filename.endswith('.json'):
                        os.remove(file_path)
                        
                logger.info("Весь кэш очищен")
                return True
            except Exception as e:
                logger.error(f"Ошибка при очистке всего кэша: {str(e)}")
                return False
