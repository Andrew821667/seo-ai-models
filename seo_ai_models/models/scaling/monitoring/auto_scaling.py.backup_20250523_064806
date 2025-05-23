# -*- coding: utf-8 -*-
"""
AutoScaling - Автоматическое масштабирование компонентов системы.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import uuid

logger = logging.getLogger(__name__)

class CPUUtilizationPolicy:
    """Политика масштабирования на основе загрузки CPU."""
    
    def __init__(
        self,
        component: str,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 20.0,
        cooldown_period: int = 300,
        min_instances: int = 1,
        max_instances: int = 10
    ):
        """
        Инициализирует политику масштабирования.
        
        Args:
            component: Имя компонента
            scale_up_threshold: Порог масштабирования вверх (%)
            scale_down_threshold: Порог масштабирования вниз (%)
            cooldown_period: Период охлаждения после масштабирования (секунды)
            min_instances: Минимальное количество экземпляров
            max_instances: Максимальное количество экземпляров
        """
        self.name = f"cpu-{component}-{str(uuid.uuid4())[:8]}"
        self.component = component
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.min_instances = min_instances
        self.max_instances = max_instances
        
        self.last_scale_time = 0
        self.last_action = None
    
    def check(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Проверяет метрики и определяет, нужно ли масштабирование.
        
        Args:
            metrics: Метрики
            
        Returns:
            Действие масштабирования или None
        """
        # Проверяем, прошел ли период охлаждения
        current_time = time.time()
        if current_time - self.last_scale_time < self.cooldown_period:
            return None
        
        # Получаем загрузку CPU
        cpu_utilization = metrics.get("cpu", {}).get("percent")
        
        if cpu_utilization is None:
            return None
        
        # Определяем, нужно ли масштабирование
        if cpu_utilization > self.scale_up_threshold:
            self.last_scale_time = current_time
            self.last_action = "scale_up"
            
            return {
                "action": "scale_up",
                "component": self.component,
                "reason": f"CPU utilization ({cpu_utilization}%) exceeded scale up threshold ({self.scale_up_threshold}%)"
            }
        elif cpu_utilization < self.scale_down_threshold:
            self.last_scale_time = current_time
            self.last_action = "scale_down"
            
            return {
                "action": "scale_down",
                "component": self.component,
                "reason": f"CPU utilization ({cpu_utilization}%) below scale down threshold ({self.scale_down_threshold}%)"
            }
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует политику в словарь.
        
        Returns:
            Словарь с данными политики
        """
        return {
            "name": self.name,
            "component": self.component,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "cooldown_period": self.cooldown_period,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "last_scale_time": self.last_scale_time,
            "last_action": self.last_action
        }

class AutoScaling:
    """
    Автоматическое масштабирование компонентов системы.
    
    Управляет масштабированием компонентов системы на основе метрик
    и политик масштабирования.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует AutoScaling.
        
        Args:
            config: Конфигурация
        """
        self.config = config or {}
        
        # Компоненты
        self.components = {}
        
        # Политики масштабирования
        self.policies = {}
        
        # Блокировка для доступа к компонентам и политикам
        self.lock = threading.Lock()
        
        # История масштабирования
        self.scaling_history = []
        
        # Максимальное количество элементов истории
        self.max_history_size = 1000
    
    def add_component(
        self,
        name: str,
        min_instances: int = 1,
        max_instances: int = 10,
        current_instances: int = 1
    ) -> Dict[str, Any]:
        """
        Добавляет компонент для масштабирования.
        
        Args:
            name: Имя компонента
            min_instances: Минимальное количество экземпляров
            max_instances: Максимальное количество экземпляров
            current_instances: Текущее количество экземпляров
            
        Returns:
            Результат добавления
        """
        with self.lock:
            if name in self.components:
                return {
                    "status": "error",
                    "message": f"Component {name} already exists"
                }
            
            self.components[name] = {
                "name": name,
                "min_instances": min_instances,
                "max_instances": max_instances,
                "current_instances": current_instances,
                "last_scale_time": None,
                "last_scale_action": None
            }
            
            return {
                "status": "success",
                "message": f"Component {name} added",
                "component": self.components[name]
            }
    
    def get_component_status(self, name: str) -> Dict[str, Any]:
        """
        Возвращает статус компонента.
        
        Args:
            name: Имя компонента
            
        Returns:
            Статус компонента
        """
        with self.lock:
            if name not in self.components:
                return {
                    "status": "error",
                    "message": f"Component {name} not found"
                }
            
            return {
                "status": "success",
                "component": self.components[name]
            }
    
    def update_component(
        self,
        name: str,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
        current_instances: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Обновляет компонент.
        
        Args:
            name: Имя компонента
            min_instances: Минимальное количество экземпляров
            max_instances: Максимальное количество экземпляров
            current_instances: Текущее количество экземпляров
            
        Returns:
            Результат обновления
        """
        with self.lock:
            if name not in self.components:
                return {
                    "status": "error",
                    "message": f"Component {name} not found"
                }
            
            component = self.components[name]
            
            if min_instances is not None:
                component["min_instances"] = min_instances
            
            if max_instances is not None:
                component["max_instances"] = max_instances
            
            if current_instances is not None:
                component["current_instances"] = current_instances
            
            return {
                "status": "success",
                "message": f"Component {name} updated",
                "component": component
            }
    
    def scale_component(
        self,
        component_name: str,
        action: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Масштабирует компонент.
        
        Args:
            component_name: Имя компонента
            action: Действие масштабирования ("scale_up" или "scale_down")
            reason: Причина масштабирования
            
        Returns:
            Результат масштабирования
        """
        with self.lock:
            if component_name not in self.components:
                return {
                    "status": "error",
                    "message": f"Component {component_name} not found"
                }
            
            component = self.components[component_name]
            current_instances = component["current_instances"]
            
            if action == "scale_up":
                # Проверяем, не превышено ли максимальное количество экземпляров
                if current_instances >= component["max_instances"]:
                    return {
                        "status": "error",
                        "message": f"Component {component_name} already at maximum instances"
                    }
                
                # Увеличиваем количество экземпляров
                component["current_instances"] += 1
            elif action == "scale_down":
                # Проверяем, не меньше ли минимального количества экземпляров
                if current_instances <= component["min_instances"]:
                    return {
                        "status": "error",
                        "message": f"Component {component_name} already at minimum instances"
                    }
                
                # Уменьшаем количество экземпляров
                component["current_instances"] -= 1
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
            
            # Обновляем время и действие последнего масштабирования
            component["last_scale_time"] = time.time()
            component["last_scale_action"] = action
            
            # Добавляем запись в историю масштабирования
            self._add_to_history({
                "timestamp": time.time(),
                "component": component_name,
                "action": action,
                "reason": reason,
                "instances_before": current_instances,
                "instances_after": component["current_instances"]
            })
            
            return {
                "status": "success",
                "message": f"Component {component_name} scaled {action}",
                "component": component
            }
    
    def _add_to_history(self, entry: Dict[str, Any]):
        """
        Добавляет запись в историю масштабирования.
        
        Args:
            entry: Запись
        """
        self.scaling_history.append(entry)
        
        # Ограничиваем размер истории
        if len(self.scaling_history) > self.max_history_size:
            self.scaling_history.pop(0)
    
    def get_scaling_history(
        self,
        component_name: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Возвращает историю масштабирования.
        
        Args:
            component_name: Имя компонента
            limit: Максимальное количество записей
            
        Returns:
            История масштабирования
        """
        with self.lock:
            # Фильтруем историю по имени компонента, если указано
            if component_name:
                filtered_history = [
                    entry for entry in self.scaling_history
                    if entry["component"] == component_name
                ]
            else:
                filtered_history = self.scaling_history
            
            # Сортируем по времени (от новых к старым)
            sorted_history = sorted(
                filtered_history,
                key=lambda entry: entry["timestamp"],
                reverse=True
            )
            
            # Применяем лимит
            limited_history = sorted_history[:limit]
            
            return {
                "status": "success",
                "history": limited_history,
                "total": len(filtered_history),
                "limit": limit
            }
    
    def add_policy(self, policy: Any) -> Dict[str, Any]:
        """
        Добавляет политику масштабирования.
        
        Args:
            policy: Политика масштабирования
            
        Returns:
            Результат добавления
        """
        with self.lock:
            if policy.name in self.policies:
                return {
                    "status": "error",
                    "message": f"Policy {policy.name} already exists"
                }
            
            # Проверяем, существует ли компонент
            if policy.component not in self.components:
                return {
                    "status": "error",
                    "message": f"Component {policy.component} not found"
                }
            
            self.policies[policy.name] = policy
            
            return {
                "status": "success",
                "message": f"Policy {policy.name} added",
                "policy": policy.to_dict()
            }
    
    def get_policy(self, name: str) -> Dict[str, Any]:
        """
        Возвращает политику масштабирования.
        
        Args:
            name: Имя политики
            
        Returns:
            Политика масштабирования
        """
        with self.lock:
            if name not in self.policies:
                return {
                    "status": "error",
                    "message": f"Policy {name} not found"
                }
            
            return {
                "status": "success",
                "policy": self.policies[name].to_dict()
            }
    
    def get_all_policies(self) -> Dict[str, Any]:
        """
        Возвращает все политики масштабирования.
        
        Returns:
            Политики масштабирования
        """
        with self.lock:
            return {
                "status": "success",
                "policies": {
                    name: policy.to_dict()
                    for name, policy in self.policies.items()
                }
            }
    
    def check_policies(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет политики масштабирования.
        
        Args:
            metrics: Метрики
            
        Returns:
            Результаты проверки
        """
        actions = []
        
        with self.lock:
            for name, policy in self.policies.items():
                action = policy.check(metrics)
                
                if action:
                    actions.append(action)
        
        return {
            "status": "success",
            "actions": actions
        }
    
    def apply_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Применяет действия масштабирования.
        
        Args:
            actions: Действия масштабирования
            
        Returns:
            Результаты применения
        """
        results = []
        
        for action in actions:
            component_name = action["component"]
            action_type = action["action"]
            reason = action["reason"]
            
            result = self.scale_component(component_name, action_type, reason)
            results.append(result)
        
        return {
            "status": "success",
            "results": results
        }
