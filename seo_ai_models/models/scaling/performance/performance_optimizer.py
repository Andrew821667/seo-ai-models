# -*- coding: utf-8 -*-
"""
PerformanceOptimizer - Оптимизация производительности компонентов.
"""

import logging
import time
import functools
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

logger = logging.getLogger(__name__)

class CachingStrategy:
    """Стратегия кэширования для оптимизации производительности."""
    
    def __init__(self, cache_size: int = 100):
        """
        Инициализирует стратегию кэширования.
        
        Args:
            cache_size: Размер кэша
        """
        self.name = "Caching"
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def apply(self, component: Callable) -> Callable:
        """
        Применяет стратегию кэширования к компоненту.
        
        Args:
            component: Компонент для оптимизации
            
        Returns:
            Оптимизированный компонент
        """
        @functools.wraps(component)
        def wrapper(*args, **kwargs):
            # Создаем ключ для кэша
            key = str(args) + str(sorted(kwargs.items()))
            
            # Проверяем, есть ли результат в кэше
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
            
            # Если результата в кэше нет, вычисляем его
            self.cache_misses += 1
            result = component(*args, **kwargs)
            
            # Добавляем результат в кэш
            self.cache[key] = result
            
            # Если кэш превысил размер, удаляем самый старый элемент
            if len(self.cache) > self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            return result
        
        return wrapper
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику кэширования.
        
        Returns:
            Статистика кэширования
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = 0
        
        if total_requests > 0:
            hit_rate = (self.cache_hits / total_requests) * 100
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }

class PerformanceOptimizer:
    """
    Оптимизирует производительность компонентов системы.
    
    Анализирует и оптимизирует производительность компонентов системы
    с помощью различных стратегий оптимизации.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует PerformanceOptimizer.
        
        Args:
            config: Конфигурация
        """
        self.config = config or {}
        
        # Стратегии оптимизации
        self.strategies = {
            "Caching": CachingStrategy()
        }
    
    def analyze_component(
        self,
        component: Callable,
        component_name: str,
        test_data: Any = None,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Анализирует производительность компонента.
        
        Args:
            component: Компонент для анализа
            component_name: Имя компонента
            test_data: Тестовые данные
            iterations: Количество итераций
            
        Returns:
            Результаты анализа
        """
        # Результаты анализа
        results = {
            "component_name": component_name,
            "iterations": iterations,
            "execution_times": []
        }
        
        # Замеряем время выполнения
        for _ in range(iterations):
            start_time = time.time()
            
            try:
                if callable(test_data):
                    test_data_instance = test_data()
                    component(test_data_instance)
                else:
                    component(test_data)
            except TypeError:
                # Если компонент не принимает аргументы
                component()
            
            execution_time = time.time() - start_time
            results["execution_times"].append(execution_time)
        
        # Рассчитываем статистику
        execution_times = results["execution_times"]
        average_execution_time = sum(execution_times) / len(execution_times)
        min_execution_time = min(execution_times)
        max_execution_time = max(execution_times)
        
        results["average_execution_time"] = average_execution_time
        results["min_execution_time"] = min_execution_time
        results["max_execution_time"] = max_execution_time
        
        return results
    
    def optimize_component(
        self,
        component: Callable,
        component_name: str,
        strategy_names: List[str] = None,
        strategy_params: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Оптимизирует компонент с помощью указанных стратегий.
        
        Args:
            component: Компонент для оптимизации
            component_name: Имя компонента
            strategy_names: Имена стратегий
            strategy_params: Параметры стратегий
            
        Returns:
            Результаты оптимизации
        """
        strategy_names = strategy_names or []
        strategy_params = strategy_params or {}
        
        # Результаты оптимизации
        results = {
            "component_name": component_name,
            "strategies": [],
            "status": "success"
        }
        
        # Оптимизированный компонент
        optimized_component = component
        
        # Применяем стратегии
        for strategy_name in strategy_names:
            if strategy_name not in self.strategies:
                results["status"] = "error"
                results["message"] = f"Strategy {strategy_name} not found"
                return results
            
            strategy = self.strategies[strategy_name]
            
            # Применяем параметры стратегии, если они указаны
            if strategy_name in strategy_params:
                for param_name, param_value in strategy_params[strategy_name].items():
                    if hasattr(strategy, param_name):
                        setattr(strategy, param_name, param_value)
            
            # Применяем стратегию
            optimized_component = strategy.apply(optimized_component)
            
            # Добавляем информацию о стратегии в результаты
            results["strategies"].append({
                "name": strategy_name,
                "params": strategy_params.get(strategy_name, {})
            })
        
        # Добавляем оптимизированный компонент в результаты
        results["optimized_component"] = optimized_component
        
        return results
    
    def get_strategy_statistics(self, strategy_name: str) -> Dict[str, Any]:
        """
        Возвращает статистику стратегии.
        
        Args:
            strategy_name: Имя стратегии
            
        Returns:
            Статистика стратегии
        """
        if strategy_name not in self.strategies:
            return {
                "status": "error",
                "message": f"Strategy {strategy_name} not found"
            }
        
        strategy = self.strategies[strategy_name]
        
        if hasattr(strategy, "get_statistics"):
            return {
                "status": "success",
                "strategy": strategy_name,
                "statistics": strategy.get_statistics()
            }
        
        return {
            "status": "error",
            "message": f"Strategy {strategy_name} does not have statistics"
        }
    
    def get_available_strategies(self) -> Dict[str, Any]:
        """
        Возвращает список доступных стратегий.
        
        Returns:
            Список доступных стратегий
        """
        strategies = {}
        
        for name, strategy in self.strategies.items():
            strategies[name] = {
                "name": name,
                "description": getattr(strategy, "description", "")
            }
        
        return {
            "status": "success",
            "strategies": strategies
        }
