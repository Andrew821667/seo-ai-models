# -*- coding: utf-8 -*-
"""
StepManager - Управляет шагами онбординга.
Обеспечивает динамическое управление шагами процесса онбординга.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class StepManager:
    """
    Управляет шагами процесса онбординга.
    
    Позволяет динамически создавать, настраивать и адаптировать
    шаги онбординга в зависимости от плана пользователя и его
    предпочтений.
    """
    
    def __init__(
        self,
        steps_config: Optional[Union[Dict[str, Any], str]] = None,
        user_id: Optional[str] = None,
        plan: Optional[str] = "free",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализирует StepManager.
        
        Args:
            steps_config: Конфигурация шагов (словарь или путь к JSON-файлу)
            user_id: Идентификатор пользователя
            plan: План пользователя
            config: Дополнительная конфигурация
        """
        self.user_id = user_id
        self.plan = plan
        self.config = config or {}
        
        # Загружаем конфигурацию шагов
        self.steps_config = self._load_steps_config(steps_config)
        
        # Адаптируем шаги для плана пользователя
        self._adapt_steps_for_plan()
    
    def _load_steps_config(self, steps_config: Optional[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
        """
        Загружает конфигурацию шагов из словаря или файла.
        
        Args:
            steps_config: Конфигурация шагов (словарь или путь к JSON-файлу)
            
        Returns:
            Загруженная конфигурация шагов
        """
        # Если конфигурация не указана, используем стандартную
        if steps_config is None:
            return self._get_default_steps_config()
        
        # Если конфигурация - строка, пытаемся загрузить JSON-файл
        if isinstance(steps_config, str):
            try:
                with open(steps_config, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading steps config from file {steps_config}: {e}")
                return self._get_default_steps_config()
        
        # Если конфигурация - словарь, используем её
        if isinstance(steps_config, dict):
            return steps_config
        
        # В остальных случаях используем стандартную конфигурацию
        return self._get_default_steps_config()
    
    def _get_default_steps_config(self) -> Dict[str, Any]:
        """
        Возвращает стандартную конфигурацию шагов.
        
        Returns:
            Стандартная конфигурация шагов
        """
        return {
            "steps": [
                {
                    "id": "welcome",
                    "title": "Добро пожаловать",
                    "description": "Знакомство с системой",
                    "required": True,
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"]
                },
                {
                    "id": "account_setup",
                    "title": "Настройка аккаунта",
                    "description": "Настройка профиля пользователя",
                    "required": True,
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"]
                },
                {
                    "id": "project_creation",
                    "title": "Создание проекта",
                    "description": "Создание первого проекта",
                    "required": True,
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"]
                },
                {
                    "id": "first_analysis",
                    "title": "Первый анализ",
                    "description": "Проведение первого анализа контента",
                    "required": True,
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"]
                },
                {
                    "id": "explore_features",
                    "title": "Изучение возможностей",
                    "description": "Знакомство с основными функциями",
                    "required": False,
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"]
                },
                {
                    "id": "invite_team",
                    "title": "Приглашение команды",
                    "description": "Добавление участников команды",
                    "required": False,
                    "applicable_plans": ["basic", "pro", "enterprise"]
                },
                {
                    "id": "api_integration",
                    "title": "API интеграция",
                    "description": "Настройка интеграции с API",
                    "required": False,
                    "applicable_plans": ["basic", "pro", "enterprise"]
                },
                {
                    "id": "advanced_features",
                    "title": "Расширенные возможности",
                    "description": "Знакомство с расширенными функциями",
                    "required": False,
                    "applicable_plans": ["pro", "enterprise"]
                },
                {
                    "id": "enterprise_setup",
                    "title": "Корпоративная настройка",
                    "description": "Настройка корпоративных функций",
                    "required": False,
                    "applicable_plans": ["enterprise"]
                },
                {
                    "id": "complete",
                    "title": "Завершение",
                    "description": "Завершение процесса онбординга",
                    "required": True,
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"]
                }
            ],
            "step_dependencies": {
                "welcome": [],
                "account_setup": ["welcome"],
                "project_creation": ["account_setup"],
                "first_analysis": ["project_creation"],
                "explore_features": ["first_analysis"],
                "invite_team": ["explore_features"],
                "api_integration": ["explore_features"],
                "advanced_features": ["explore_features"],
                "enterprise_setup": ["explore_features"],
                "complete": ["welcome", "account_setup", "project_creation", "first_analysis"]
            }
        }
    
    def _adapt_steps_for_plan(self):
        """Адаптирует шаги для плана пользователя."""
        # Фильтруем шаги, применимые для текущего плана
        applicable_steps = []
        
        for step in self.steps_config["steps"]:
            if self.plan in step["applicable_plans"]:
                applicable_steps.append(step)
        
        # Обновляем конфигурацию шагов
        self.steps_config["steps"] = applicable_steps
        
        # Адаптируем зависимости шагов
        self._adapt_step_dependencies()
    
    def _adapt_step_dependencies(self):
        """Адаптирует зависимости шагов с учетом отфильтрованных шагов."""
        new_dependencies = {}
        step_ids = [step["id"] for step in self.steps_config["steps"]]
        
        for step_id, dependencies in self.steps_config["step_dependencies"].items():
            # Если шаг не в отфильтрованных, пропускаем его
            if step_id not in step_ids:
                continue
            
            # Фильтруем зависимости, оставляя только существующие шаги
            filtered_dependencies = [dep for dep in dependencies if dep in step_ids]
            new_dependencies[step_id] = filtered_dependencies
        
        # Обновляем зависимости шагов
        self.steps_config["step_dependencies"] = new_dependencies
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """
        Возвращает список шагов для текущего плана.
        
        Returns:
            Список шагов
        """
        return self.steps_config["steps"]
    
    def get_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о конкретном шаге.
        
        Args:
            step_id: Идентификатор шага
            
        Returns:
            Информация о шаге или None, если шаг не найден
        """
        for step in self.steps_config["steps"]:
            if step["id"] == step_id:
                return step
        
        return None
    
    def get_required_steps(self) -> List[Dict[str, Any]]:
        """
        Возвращает список обязательных шагов.
        
        Returns:
            Список обязательных шагов
        """
        return [step for step in self.steps_config["steps"] if step["required"]]
    
    def get_optional_steps(self) -> List[Dict[str, Any]]:
        """
        Возвращает список необязательных шагов.
        
        Returns:
            Список необязательных шагов
        """
        return [step for step in self.steps_config["steps"] if not step["required"]]
    
    def get_step_dependencies(self, step_id: str) -> List[str]:
        """
        Возвращает список зависимостей для указанного шага.
        
        Args:
            step_id: Идентификатор шага
            
        Returns:
            Список идентификаторов шагов, от которых зависит указанный шаг
        """
        return self.steps_config["step_dependencies"].get(step_id, [])
    
    def check_step_availability(self, step_id: str, completed_steps: List[str]) -> Dict[str, Any]:
        """
        Проверяет доступность шага на основе завершенных шагов.
        
        Args:
            step_id: Идентификатор шага
            completed_steps: Список завершенных шагов
            
        Returns:
            Результат проверки доступности
        """
        # Проверяем, существует ли шаг
        step = self.get_step(step_id)
        if not step:
            return {
                "available": False,
                "message": f"Шаг {step_id} не найден",
                "missing_dependencies": []
            }
        
        # Получаем зависимости шага
        dependencies = self.get_step_dependencies(step_id)
        
        # Проверяем, все ли зависимости выполнены
        missing_dependencies = [dep for dep in dependencies if dep not in completed_steps]
        
        if missing_dependencies:
            return {
                "available": False,
                "message": f"Для доступа к шагу {step_id} необходимо завершить предыдущие шаги",
                "missing_dependencies": missing_dependencies
            }
        
        return {
            "available": True,
            "message": f"Шаг {step_id} доступен",
            "missing_dependencies": []
        }
    
    def get_next_step(self, current_step_id: str, completed_steps: List[str]) -> Optional[Dict[str, Any]]:
        """
        Определяет следующий шаг после текущего.
        
        Args:
            current_step_id: Идентификатор текущего шага
            completed_steps: Список завершенных шагов
            
        Returns:
            Информация о следующем шаге или None, если следующего шага нет
        """
        # Получаем список всех шагов
        steps = self.get_steps()
        step_ids = [step["id"] for step in steps]
        
        # Если текущего шага нет в списке, возвращаем None
        if current_step_id not in step_ids:
            return None
        
        # Получаем индекс текущего шага
        current_index = step_ids.index(current_step_id)
        
        # Если текущий шаг последний, возвращаем None
        if current_index == len(steps) - 1:
            return None
        
        # Проверяем доступность следующих шагов
        for i in range(current_index + 1, len(steps)):
            next_step_id = step_ids[i]
            availability = self.check_step_availability(next_step_id, completed_steps)
            
            if availability["available"]:
                return self.get_step(next_step_id)
        
        return None
    
    def get_onboarding_path(self, include_optional: bool = True) -> List[Dict[str, Any]]:
        """
        Возвращает оптимальный путь онбординга.
        
        Args:
            include_optional: Включать ли необязательные шаги
            
        Returns:
            Список шагов в оптимальном порядке
        """
        # Получаем список всех шагов
        all_steps = self.get_steps()
        
        # Фильтруем шаги, если нужно исключить необязательные
        if not include_optional:
            steps = [step for step in all_steps if step["required"]]
        else:
            steps = all_steps
        
        # Создаем граф зависимостей
        dependencies = self.steps_config["step_dependencies"]
        
        # Создаем список для результата
        result = []
        
        # Создаем множество для отслеживания добавленных шагов
        added_steps = set()
        
        # Функция для добавления шага и его зависимостей
        def add_step_with_dependencies(step_id):
            # Если шаг уже добавлен, пропускаем его
            if step_id in added_steps:
                return
            
            # Получаем зависимости шага
            step_dependencies = dependencies.get(step_id, [])
            
            # Добавляем зависимости
            for dep in step_dependencies:
                add_step_with_dependencies(dep)
            
            # Добавляем сам шаг, если он еще не добавлен
            if step_id not in added_steps:
                step = self.get_step(step_id)
                if step:
                    result.append(step)
                    added_steps.add(step_id)
        
        # Добавляем все шаги
        for step in steps:
            add_step_with_dependencies(step["id"])
        
        return result
    
    def customize_steps(self, customizations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Настраивает шаги онбординга на основе пользовательских предпочтений.
        
        Args:
            customizations: Настройки шагов
            
        Returns:
            Результат настройки
        """
        # Обрабатываем отключение шагов
        if "disabled_steps" in customizations:
            disabled_steps = customizations["disabled_steps"]
            
            # Фильтруем обязательные шаги из списка отключаемых
            required_steps = [step["id"] for step in self.get_required_steps()]
            invalid_disabled = [step_id for step_id in disabled_steps if step_id in required_steps]
            
            if invalid_disabled:
                return {
                    "status": "error",
                    "message": f"Невозможно отключить обязательные шаги: {', '.join(invalid_disabled)}"
                }
            
            # Отключаем необязательные шаги
            self.steps_config["steps"] = [step for step in self.steps_config["steps"] if step["id"] not in disabled_steps or step["required"]]
            
            # Адаптируем зависимости шагов
            self._adapt_step_dependencies()
        
        # Обрабатываем изменение порядка шагов
        if "steps_order" in customizations:
            steps_order = customizations["steps_order"]
            
            # Проверяем, что все шаги существуют
            existing_step_ids = [step["id"] for step in self.steps_config["steps"]]
            invalid_steps = [step_id for step_id in steps_order if step_id not in existing_step_ids]
            
            if invalid_steps:
                return {
                    "status": "error",
                    "message": f"Неизвестные шаги: {', '.join(invalid_steps)}"
                }
            
            # Проверяем, что все существующие шаги включены
            missing_steps = [step_id for step_id in existing_step_ids if step_id not in steps_order]
            
            if missing_steps:
                return {
                    "status": "error",
                    "message": f"Не все шаги включены в новый порядок: {', '.join(missing_steps)}"
                }
            
            # Изменяем порядок шагов
            ordered_steps = []
            for step_id in steps_order:
                for step in self.steps_config["steps"]:
                    if step["id"] == step_id:
                        ordered_steps.append(step)
                        break
            
            self.steps_config["steps"] = ordered_steps
        
        # Обрабатываем дополнительные настройки
        if "step_settings" in customizations:
            step_settings = customizations["step_settings"]
            
            for step_id, settings in step_settings.items():
                # Находим шаг
                step = self.get_step(step_id)
                
                if not step:
                    continue
                
                # Обновляем настройки шага
                for key, value in settings.items():
                    if key in ["title", "description"]:
                        step[key] = value
        
        return {
            "status": "success",
            "message": "Настройки шагов обновлены",
            "steps": self.get_steps()
        }
    
    def save_steps_config(self, file_path: str) -> Dict[str, Any]:
        """
        Сохраняет конфигурацию шагов в файл.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Результат сохранения
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.steps_config, f, indent=2, ensure_ascii=False)
            
            return {
                "status": "success",
                "message": f"Конфигурация шагов сохранена в файл {file_path}"
            }
        except IOError as e:
            logger.error(f"Error saving steps config to file {file_path}: {e}")
            
            return {
                "status": "error",
                "message": f"Ошибка сохранения конфигурации: {str(e)}"
            }
    
    def load_steps_config_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Загружает конфигурацию шагов из файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Результат загрузки
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.steps_config = json.load(f)
            
            # Адаптируем шаги для плана пользователя
            self._adapt_steps_for_plan()
            
            return {
                "status": "success",
                "message": f"Конфигурация шагов загружена из файла {file_path}",
                "steps": self.get_steps()
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading steps config from file {file_path}: {e}")
            
            return {
                "status": "error",
                "message": f"Ошибка загрузки конфигурации: {str(e)}"
            }
