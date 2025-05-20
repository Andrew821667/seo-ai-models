# -*- coding: utf-8 -*-
"""
OnboardingWizard - Мастер быстрого старта для новых пользователей.
Проводит пользователей через процесс начала работы с системой.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class OnboardingStep(Enum):
    """Шаги процесса онбординга."""
    WELCOME = "welcome"
    ACCOUNT_SETUP = "account_setup"
    PROJECT_CREATION = "project_creation"
    FIRST_ANALYSIS = "first_analysis"
    EXPLORE_FEATURES = "explore_features"
    INVITE_TEAM = "invite_team"
    COMPLETE = "complete"

class OnboardingWizard:
    """
    Мастер быстрого старта для новых пользователей.
    
    Предоставляет пошаговый процесс онбординга с интерактивными
    руководствами и подсказками для быстрого освоения системы.
    """
    
    def __init__(
        self,
        user_id: str,
        plan: Optional[str] = "free",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализирует OnboardingWizard.
        
        Args:
            user_id: Идентификатор пользователя
            plan: План пользователя
            config: Конфигурация
        """
        self.user_id = user_id
        self.plan = plan
        self.config = config or {}
        
        # Уникальный идентификатор процесса онбординга
        self.onboarding_id = str(uuid.uuid4())
        
        # Данные для отслеживания прогресса
        self.progress = {
            "start_time": datetime.now().isoformat(),
            "current_step": OnboardingStep.WELCOME.value,
            "completed_steps": [],
            "next_steps": [step.value for step in OnboardingStep if step != OnboardingStep.WELCOME],
            "completion_percentage": 0
        }
        
        # Настройки отображения шагов в зависимости от плана
        self._configure_steps_for_plan()
    
    def _configure_steps_for_plan(self):
        """Настраивает шаги онбординга в зависимости от плана пользователя."""
        # Для бесплатного плана не показываем шаг приглашения команды
        if self.plan == "free":
            if OnboardingStep.INVITE_TEAM.value in self.progress["next_steps"]:
                self.progress["next_steps"].remove(OnboardingStep.INVITE_TEAM.value)
        
        # Для Enterprise плана добавляем дополнительные шаги, если они будут в будущем
        if self.plan == "enterprise":
            # Здесь можно добавить дополнительные шаги для Enterprise
            pass
    
    def get_current_step(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущем шаге онбординга.
        
        Returns:
            Информация о текущем шаге
        """
        current_step = self.progress["current_step"]
        
        # Получаем содержимое текущего шага
        step_content = self._get_step_content(current_step)
        
        return {
            "onboarding_id": self.onboarding_id,
            "current_step": current_step,
            "progress": self.progress["completion_percentage"],
            "content": step_content,
            "next_step": self._get_next_step(),
            "prev_step": self._get_prev_step()
        }
    
    def _get_step_content(self, step: str) -> Dict[str, Any]:
        """
        Возвращает содержимое указанного шага.
        
        Args:
            step: Идентификатор шага
            
        Returns:
            Содержимое шага
        """
        # Содержимое шагов онбординга
        step_contents = {
            OnboardingStep.WELCOME.value: {
                "title": "Добро пожаловать в SEO AI Models!",
                "description": "Мы рады приветствовать вас в нашей системе анализа и оптимизации контента. Давайте настроим всё для успешного старта.",
                "image_url": "/static/img/onboarding/welcome.png",
                "video_url": "/static/videos/onboarding/welcome.mp4",
                "action": {
                    "type": "button",
                    "text": "Начать работу",
                    "next_step": OnboardingStep.ACCOUNT_SETUP.value
                }
            },
            
            OnboardingStep.ACCOUNT_SETUP.value: {
                "title": "Настройка вашего аккаунта",
                "description": "Давайте настроим ваш профиль для более персонализированного опыта.",
                "form": {
                    "fields": [
                        {
                            "name": "full_name",
                            "type": "text",
                            "label": "ФИО",
                            "required": True
                        },
                        {
                            "name": "company",
                            "type": "text",
                            "label": "Компания",
                            "required": False
                        },
                        {
                            "name": "industry",
                            "type": "select",
                            "label": "Отрасль",
                            "options": [
                                "E-commerce",
                                "Медиа и контент",
                                "Образование",
                                "Финансы",
                                "Здравоохранение",
                                "Технологии",
                                "Туризм и гостеприимство",
                                "Другое"
                            ],
                            "required": True
                        },
                        {
                            "name": "experience_level",
                            "type": "radio",
                            "label": "Опыт работы с SEO",
                            "options": [
                                "Новичок",
                                "Средний уровень",
                                "Эксперт"
                            ],
                            "required": True
                        }
                    ],
                    "submit_button": "Сохранить и продолжить"
                },
                "action": {
                    "type": "form_submit",
                    "next_step": OnboardingStep.PROJECT_CREATION.value
                }
            },
            
            OnboardingStep.PROJECT_CREATION.value: {
                "title": "Создание вашего первого проекта",
                "description": "Проекты помогают организовать вашу работу. Давайте создадим ваш первый проект.",
                "steps": [
                    "Введите название и описание вашего проекта",
                    "Добавьте URL вашего сайта для анализа",
                    "Выберите ключевые слова для отслеживания (опционально)",
                    "Настройте дополнительные параметры проекта"
                ],
                "form": {
                    "fields": [
                        {
                            "name": "project_name",
                            "type": "text",
                            "label": "Название проекта",
                            "required": True
                        },
                        {
                            "name": "project_description",
                            "type": "textarea",
                            "label": "Описание проекта",
                            "required": False
                        },
                        {
                            "name": "website_url",
                            "type": "url",
                            "label": "URL вашего сайта",
                            "required": True
                        },
                        {
                            "name": "keywords",
                            "type": "textarea",
                            "label": "Ключевые слова (по одному на строку)",
                            "required": False
                        }
                    ],
                    "submit_button": "Создать проект"
                },
                "tips": [
                    "Давайте проекту понятное название, которое отражает его цель",
                    "Добавляйте ключевые слова, по которым вы хотите ранжироваться",
                    "Не беспокойтесь, вы всегда сможете изменить эти настройки позже"
                ],
                "action": {
                    "type": "form_submit",
                    "next_step": OnboardingStep.FIRST_ANALYSIS.value
                }
            },
            
            OnboardingStep.FIRST_ANALYSIS.value: {
                "title": "Ваш первый анализ контента",
                "description": "Давайте проанализируем контент вашего сайта и получим рекомендации по оптимизации.",
                "steps": [
                    "Выберите страницу для анализа",
                    "Запустите анализ и дождитесь результатов",
                    "Изучите рекомендации и метрики"
                ],
                "demo": {
                    "type": "interactive",
                    "url": "/demo/first-analysis"
                },
                "tips": [
                    "Начните с анализа главной страницы или важной целевой страницы",
                    "Обратите внимание на показатели E-E-A-T",
                    "Рекомендации отсортированы по важности для быстрого внедрения"
                ],
                "action": {
                    "type": "button",
                    "text": "Перейти к анализу",
                    "next_step": OnboardingStep.EXPLORE_FEATURES.value
                }
            },
            
            OnboardingStep.EXPLORE_FEATURES.value: {
                "title": "Исследуйте возможности системы",
                "description": "Познакомьтесь с ключевыми функциями, которые помогут вам оптимизировать ваш контент.",
                "features": [
                    {
                        "name": "Анализ контента",
                        "description": "Глубокий анализ контента с оценкой качества, читабельности и релевантности",
                        "image_url": "/static/img/onboarding/content-analysis.png",
                        "demo_url": "/demo/content-analysis"
                    },
                    {
                        "name": "Рекомендации",
                        "description": "Персонализированные рекомендации по улучшению контента",
                        "image_url": "/static/img/onboarding/recommendations.png",
                        "demo_url": "/demo/recommendations"
                    },
                    {
                        "name": "E-E-A-T Анализ",
                        "description": "Оценка опыта, экспертизы, авторитетности и надежности контента",
                        "image_url": "/static/img/onboarding/eeat.png",
                        "demo_url": "/demo/eeat"
                    },
                    {
                        "name": "Отчеты",
                        "description": "Детальные отчеты с визуализацией данных",
                        "image_url": "/static/img/onboarding/reports.png",
                        "demo_url": "/demo/reports"
                    }
                ],
                "action": {
                    "type": "button",
                    "text": "Продолжить",
                    "next_step": OnboardingStep.INVITE_TEAM.value if OnboardingStep.INVITE_TEAM.value in self.progress["next_steps"] else OnboardingStep.COMPLETE.value
                }
            },
            
            OnboardingStep.INVITE_TEAM.value: {
                "title": "Пригласите вашу команду",
                "description": "Работайте вместе со своей командой для достижения лучших результатов.",
                "form": {
                    "fields": [
                        {
                            "name": "email",
                            "type": "email",
                            "label": "Email участника команды",
                            "required": True
                        },
                        {
                            "name": "role",
                            "type": "select",
                            "label": "Роль",
                            "options": [
                                "Администратор",
                                "Редактор",
                                "Аналитик",
                                "Наблюдатель"
                            ],
                            "required": True
                        }
                    ],
                    "submit_button": "Отправить приглашение"
                },
                "tips": [
                    "Администраторы имеют полный доступ к проекту",
                    "Редакторы могут вносить изменения, но не могут управлять пользователями",
                    "Аналитики могут просматривать данные и создавать отчеты",
                    "Наблюдатели имеют доступ только для чтения"
                ],
                "note": "Доступно только для платных планов. На вашем текущем плане вы можете пригласить до 3 участников.",
                "action": {
                    "type": "button",
                    "text": "Пропустить и завершить",
                    "next_step": OnboardingStep.COMPLETE.value
                }
            },
            
            OnboardingStep.COMPLETE.value: {
                "title": "Поздравляем! Вы готовы к работе!",
                "description": "Вы успешно завершили процесс онбординга и теперь можете в полной мере использовать возможности SEO AI Models.",
                "summary": [
                    "Вы настроили свой аккаунт",
                    "Создали свой первый проект",
                    "Провели первый анализ контента",
                    "Изучили основные возможности системы"
                ],
                "next_steps": [
                    {
                        "title": "Анализ конкурентов",
                        "description": "Изучите контент ваших конкурентов для выявления их стратегий",
                        "url": "/analysis/competitors"
                    },
                    {
                        "title": "Настройка отчетов",
                        "description": "Создайте регулярные отчеты для отслеживания прогресса",
                        "url": "/reports/setup"
                    },
                    {
                        "title": "Изучение руководств",
                        "description": "Ознакомьтесь с подробными руководствами по использованию системы",
                        "url": "/guides"
                    }
                ],
                "action": {
                    "type": "button",
                    "text": "Перейти к панели управления",
                    "url": "/dashboard"
                }
            }
        }
        
        # Возвращаем содержимое шага или пустой словарь, если шаг не найден
        return step_contents.get(step, {})
    
    def _get_next_step(self) -> Optional[str]:
        """
        Возвращает идентификатор следующего шага.
        
        Returns:
            Идентификатор следующего шага или None, если текущий шаг последний
        """
        current_step = self.progress["current_step"]
        
        try:
            next_steps = self.progress["next_steps"]
            if not next_steps:
                return None
            
            # Если текущий шаг COMPLETE, следующего шага нет
            if current_step == OnboardingStep.COMPLETE.value:
                return None
            
            # Если текущий шаг в next_steps, берем следующий за ним
            if current_step in next_steps:
                current_index = next_steps.index(current_step)
                if current_index < len(next_steps) - 1:
                    return next_steps[current_index + 1]
            
            # Иначе берем первый шаг из next_steps
            return next_steps[0]
        except (ValueError, IndexError):
            # Если что-то пошло не так, возвращаем None
            return None
    
    def _get_prev_step(self) -> Optional[str]:
        """
        Возвращает идентификатор предыдущего шага.
        
        Returns:
            Идентификатор предыдущего шага или None, если текущий шаг первый
        """
        current_step = self.progress["current_step"]
        completed_steps = self.progress["completed_steps"]
        
        # Если нет завершенных шагов, предыдущего шага нет
        if not completed_steps:
            return None
        
        # Возвращаем последний завершенный шаг
        return completed_steps[-1]
    
    def advance_to_step(self, step: str) -> Dict[str, Any]:
        """
        Переходит к указанному шагу.
        
        Args:
            step: Идентификатор шага
            
        Returns:
            Результат перехода
        """
        # Проверяем, существует ли указанный шаг
        try:
            target_step = OnboardingStep(step)
        except ValueError:
            return {
                "status": "error",
                "message": f"Неизвестный шаг: {step}"
            }
        
        current_step = self.progress["current_step"]
        
        # Если текущий шаг не COMPLETE, добавляем его в завершенные
        if current_step != OnboardingStep.COMPLETE.value:
            if current_step not in self.progress["completed_steps"]:
                self.progress["completed_steps"].append(current_step)
        
        # Обновляем текущий шаг
        self.progress["current_step"] = target_step.value
        
        # Удаляем текущий шаг из next_steps, если он там есть
        if target_step.value in self.progress["next_steps"]:
            self.progress["next_steps"].remove(target_step.value)
        
        # Обновляем процент завершения
        self._update_completion_percentage()
        
        return {
            "status": "success",
            "message": f"Переход к шагу: {target_step.value}",
            "current_step": self.get_current_step()
        }
    
    def _update_completion_percentage(self):
        """Обновляет процент завершения процесса онбординга."""
        # Получаем общее количество шагов (исключая COMPLETE)
        total_steps = len([step for step in OnboardingStep if step != OnboardingStep.COMPLETE])
        
        # Получаем количество завершенных шагов
        completed_steps = len(self.progress["completed_steps"])
        
        # Рассчитываем процент завершения
        if total_steps > 0:
            completion_percentage = round((completed_steps / total_steps) * 100)
        else:
            completion_percentage = 0
        
        # Если текущий шаг COMPLETE, процент завершения 100%
        if self.progress["current_step"] == OnboardingStep.COMPLETE.value:
            completion_percentage = 100
        
        self.progress["completion_percentage"] = completion_percentage
    
    def submit_step_data(self, step: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает данные, отправленные пользователем на определенном шаге.
        
        Args:
            step: Идентификатор шага
            data: Данные формы
            
        Returns:
            Результат обработки данных
        """
        # Проверяем, существует ли указанный шаг
        try:
            target_step = OnboardingStep(step)
        except ValueError:
            return {
                "status": "error",
                "message": f"Неизвестный шаг: {step}"
            }
        
        # Проверяем, что шаг соответствует текущему
        if self.progress["current_step"] != target_step.value:
            return {
                "status": "error",
                "message": f"Текущий шаг ({self.progress['current_step']}) не соответствует указанному ({target_step.value})"
            }
        
        # Обрабатываем данные в зависимости от шага
        if target_step == OnboardingStep.ACCOUNT_SETUP:
            result = self._process_account_setup(data)
        elif target_step == OnboardingStep.PROJECT_CREATION:
            result = self._process_project_creation(data)
        elif target_step == OnboardingStep.INVITE_TEAM:
            result = self._process_team_invite(data)
        else:
            # Для шагов без форм просто переходим к следующему шагу
            next_step = self._get_next_step()
            result = {
                "status": "success",
                "message": f"Шаг {target_step.value} завершен",
                "next_step": next_step
            }
        
        # Если обработка прошла успешно, переходим к следующему шагу
        if result["status"] == "success" and "next_step" in result:
            self.advance_to_step(result["next_step"])
        
        return result
    
    def _process_account_setup(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает данные настройки аккаунта.
        
        Args:
            data: Данные формы
            
        Returns:
            Результат обработки данных
        """
        # Проверяем наличие обязательных полей
        required_fields = ["full_name", "industry", "experience_level"]
        
        for field in required_fields:
            if field not in data or not data[field]:
                return {
                    "status": "error",
                    "message": f"Поле {field} обязательно для заполнения"
                }
        
        # В реальном приложении здесь будет сохранение данных пользователя
        # Например, вызов API для обновления профиля
        
        # Возвращаем успешный результат с указанием следующего шага
        return {
            "status": "success",
            "message": "Настройки аккаунта сохранены",
            "next_step": OnboardingStep.PROJECT_CREATION.value
        }
    
    def _process_project_creation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает данные создания проекта.
        
        Args:
            data: Данные формы
            
        Returns:
            Результат обработки данных
        """
        # Проверяем наличие обязательных полей
        required_fields = ["project_name", "website_url"]
        
        for field in required_fields:
            if field not in data or not data[field]:
                return {
                    "status": "error",
                    "message": f"Поле {field} обязательно для заполнения"
                }
        
        # Проверяем валидность URL
        website_url = data["website_url"]
        if not website_url.startswith(("http://", "https://")):
            return {
                "status": "error",
                "message": "URL должен начинаться с http:// или https://"
            }
        
        # В реальном приложении здесь будет создание проекта
        # Например, вызов API для создания проекта
        
        # Возвращаем успешный результат с указанием следующего шага
        return {
            "status": "success",
            "message": "Проект успешно создан",
            "next_step": OnboardingStep.FIRST_ANALYSIS.value,
            "project_id": "demo-project-123",  # В реальном приложении здесь будет ID созданного проекта
            "redirect_url": "/projects/demo-project-123"  # URL для перехода к проекту
        }
    
    def _process_team_invite(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает данные приглашения команды.
        
        Args:
            data: Данные формы
            
        Returns:
            Результат обработки данных
        """
        # Проверяем наличие обязательных полей
        required_fields = ["email", "role"]
        
        for field in required_fields:
            if field not in data or not data[field]:
                return {
                    "status": "error",
                    "message": f"Поле {field} обязательно для заполнения"
                }
        
        # Проверяем, что план позволяет приглашать пользователей
        if self.plan == "free":
            return {
                "status": "error",
                "message": "Бесплатный план не поддерживает приглашение пользователей. Обновите план для использования этой функции."
            }
        
        # В реальном приложении здесь будет отправка приглашения
        # Например, вызов API для отправки приглашения
        
        # Возвращаем успешный результат с указанием следующего шага
        return {
            "status": "success",
            "message": f"Приглашение отправлено на адрес {data['email']}",
            "next_step": OnboardingStep.COMPLETE.value
        }
    
    def skip_step(self, step: Optional[str] = None) -> Dict[str, Any]:
        """
        Пропускает указанный шаг.
        
        Args:
            step: Идентификатор шага (если None, пропускает текущий шаг)
            
        Returns:
            Результат пропуска шага
        """
        if step is None:
            step = self.progress["current_step"]
        
        # Проверяем, существует ли указанный шаг
        try:
            target_step = OnboardingStep(step)
        except ValueError:
            return {
                "status": "error",
                "message": f"Неизвестный шаг: {step}"
            }
        
        # Получаем следующий шаг
        next_step = self._get_next_step()
        
        if not next_step:
            # Если следующего шага нет, переходим к завершению
            next_step = OnboardingStep.COMPLETE.value
        
        # Переходим к следующему шагу
        return self.advance_to_step(next_step)
    
    def restart_onboarding(self) -> Dict[str, Any]:
        """
        Перезапускает процесс онбординга.
        
        Returns:
            Результат перезапуска
        """
        # Сохраняем старый ID
        old_onboarding_id = self.onboarding_id
        
        # Создаем новый ID
        self.onboarding_id = str(uuid.uuid4())
        
        # Сбрасываем прогресс
        self.progress = {
            "start_time": datetime.now().isoformat(),
            "current_step": OnboardingStep.WELCOME.value,
            "completed_steps": [],
            "next_steps": [step.value for step in OnboardingStep if step != OnboardingStep.WELCOME],
            "completion_percentage": 0
        }
        
        # Настраиваем шаги в зависимости от плана
        self._configure_steps_for_plan()
        
        return {
            "status": "success",
            "message": "Процесс онбординга перезапущен",
            "old_onboarding_id": old_onboarding_id,
            "new_onboarding_id": self.onboarding_id,
            "current_step": self.get_current_step()
        }
    
    def get_onboarding_status(self) -> Dict[str, Any]:
        """
        Возвращает статус процесса онбординга.
        
        Returns:
            Статус процесса онбординга
        """
        return {
            "onboarding_id": self.onboarding_id,
            "user_id": self.user_id,
            "plan": self.plan,
            "started_at": self.progress["start_time"],
            "current_step": self.progress["current_step"],
            "completed_steps": self.progress["completed_steps"],
            "remaining_steps": self.progress["next_steps"],
            "completion_percentage": self.progress["completion_percentage"],
            "is_completed": self.progress["current_step"] == OnboardingStep.COMPLETE.value
        }
