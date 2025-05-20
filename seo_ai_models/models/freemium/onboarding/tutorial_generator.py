# -*- coding: utf-8 -*-
"""
TutorialGenerator - Генератор обучающих материалов.
Создает персонализированные обучающие материалы для пользователей.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import random
import json

logger = logging.getLogger(__name__)

class TutorialType(Enum):
    """Типы обучающих материалов."""
    TEXT = "text"
    VIDEO = "video"
    INTERACTIVE = "interactive"
    SLIDESHOW = "slideshow"
    QUIZ = "quiz"

class TutorialLevel(Enum):
    """Уровни сложности обучающих материалов."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class TutorialGenerator:
    """
    Генератор обучающих материалов.
    
    Создает персонализированные обучающие материалы для пользователей
    с учетом их плана, опыта и предпочтений.
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        plan: str = "free",
        experience_level: str = "beginner",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализирует TutorialGenerator.
        
        Args:
            user_id: Идентификатор пользователя
            plan: План пользователя
            experience_level: Уровень опыта пользователя
            config: Дополнительная конфигурация
        """
        self.user_id = user_id
        self.plan = plan
        self.experience_level = experience_level
        self.config = config or {}
        
        # Загружаем базу знаний
        self.knowledge_base = self._load_knowledge_base()
        
        # Адаптируем материалы для плана пользователя
        self._adapt_materials_for_plan()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """
        Загружает базу знаний из файла или использует стандартную.
        
        Returns:
            База знаний
        """
        # В реальном приложении здесь будет загрузка базы знаний из файла или БД
        # Для примера используем встроенную базу знаний
        return self._get_default_knowledge_base()
    
    def _get_default_knowledge_base(self) -> Dict[str, Any]:
        """
        Возвращает стандартную базу знаний.
        
        Returns:
            Стандартная база знаний
        """
        return {
            "topics": [
                {
                    "id": "seo_basics",
                    "title": "Основы SEO",
                    "description": "Базовые понятия и принципы SEO",
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"],
                    "experience_levels": ["beginner", "intermediate", "advanced"],
                    "materials": [
                        {
                            "id": "seo_basics_intro",
                            "title": "Введение в SEO",
                            "description": "Общее представление о SEO и его значении",
                            "type": TutorialType.TEXT.value,
                            "level": TutorialLevel.BEGINNER.value,
                            "content": "Введение в SEO: базовые понятия и принципы...",
                            "estimated_time": 10  # в минутах
                        },
                        {
                            "id": "seo_basics_video",
                            "title": "Основы SEO в видеоформате",
                            "description": "Визуальное объяснение основ SEO",
                            "type": TutorialType.VIDEO.value,
                            "level": TutorialLevel.BEGINNER.value,
                            "content": "/static/videos/tutorials/seo_basics.mp4",
                            "estimated_time": 15
                        },
                        {
                            "id": "seo_basics_interactive",
                            "title": "Интерактивный урок по SEO",
                            "description": "Интерактивный урок с практическими заданиями",
                            "type": TutorialType.INTERACTIVE.value,
                            "level": TutorialLevel.INTERMEDIATE.value,
                            "content": "/tutorials/interactive/seo_basics",
                            "estimated_time": 30
                        },
                        {
                            "id": "seo_basics_advanced",
                            "title": "Продвинутые концепции SEO",
                            "description": "Углубленное изучение принципов SEO",
                            "type": TutorialType.TEXT.value,
                            "level": TutorialLevel.ADVANCED.value,
                            "content": "Продвинутые концепции SEO для экспертов...",
                            "estimated_time": 20
                        }
                    ]
                },
                {
                    "id": "content_analysis",
                    "title": "Анализ контента",
                    "description": "Методы и инструменты анализа контента",
                    "applicable_plans": ["free", "micro", "basic", "pro", "enterprise"],
                    "experience_levels": ["beginner", "intermediate", "advanced"],
                    "materials": [
                        {
                            "id": "content_analysis_basics",
                            "title": "Основы анализа контента",
                            "description": "Базовые принципы анализа контента",
                            "type": TutorialType.TEXT.value,
                            "level": TutorialLevel.BEGINNER.value,
                            "content": "Основы анализа контента для SEO...",
                            "estimated_time": 15
                        },
                        {
                            "id": "content_analysis_tools",
                            "title": "Инструменты анализа контента",
                            "description": "Обзор инструментов для анализа контента",
                            "type": TutorialType.SLIDESHOW.value,
                            "level": TutorialLevel.INTERMEDIATE.value,
                            "content": "/static/slideshows/content_analysis_tools.html",
                            "estimated_time": 20
                        },
                        {
                            "id": "content_analysis_quiz",
                            "title": "Тест по анализу контента",
                            "description": "Проверьте свои знания по анализу контента",
                            "type": TutorialType.QUIZ.value,
                            "level": TutorialLevel.INTERMEDIATE.value,
                            "content": "/tutorials/quizzes/content_analysis",
                            "estimated_time": 10
                        }
                    ]
                },
                {
                    "id": "llm_optimization",
                    "title": "Оптимизация для LLM",
                    "description": "Методы оптимизации контента для LLM-поисковиков",
                    "applicable_plans": ["micro", "basic", "pro", "enterprise"],
                    "experience_levels": ["intermediate", "advanced"],
                    "materials": [
                        {
                            "id": "llm_basics",
                            "title": "Основы LLM-оптимизации",
                            "description": "Введение в оптимизацию для LLM-поисковиков",
                            "type": TutorialType.TEXT.value,
                            "level": TutorialLevel.INTERMEDIATE.value,
                            "content": "Основы оптимизации контента для LLM-поисковиков...",
                            "estimated_time": 15
                        },
                        {
                            "id": "llm_advanced",
                            "title": "Продвинутая LLM-оптимизация",
                            "description": "Углубленные техники оптимизации для LLM",
                            "type": TutorialType.VIDEO.value,
                            "level": TutorialLevel.ADVANCED.value,
                            "content": "/static/videos/tutorials/llm_advanced.mp4",
                            "estimated_time": 25
                        }
                    ]
                },
                {
                    "id": "api_integration",
                    "title": "API интеграция",
                    "description": "Работа с API SEO AI Models",
                    "applicable_plans": ["basic", "pro", "enterprise"],
                    "experience_levels": ["intermediate", "advanced"],
                    "materials": [
                        {
                            "id": "api_basics",
                            "title": "Основы работы с API",
                            "description": "Введение в API SEO AI Models",
                            "type": TutorialType.TEXT.value,
                            "level": TutorialLevel.INTERMEDIATE.value,
                            "content": "Основы работы с API SEO AI Models...",
                            "estimated_time": 20
                        },
                        {
                            "id": "api_examples",
                            "title": "Примеры использования API",
                            "description": "Практические примеры работы с API",
                            "type": TutorialType.INTERACTIVE.value,
                            "level": TutorialLevel.ADVANCED.value,
                            "content": "/tutorials/interactive/api_examples",
                            "estimated_time": 30
                        }
                    ]
                },
                {
                    "id": "enterprise_features",
                    "title": "Корпоративные функции",
                    "description": "Расширенные функции для Enterprise-пользователей",
                    "applicable_plans": ["enterprise"],
                    "experience_levels": ["intermediate", "advanced"],
                    "materials": [
                        {
                            "id": "enterprise_overview",
                            "title": "Обзор корпоративных функций",
                            "description": "Общий обзор функций Enterprise-плана",
                            "type": TutorialType.SLIDESHOW.value,
                            "level": TutorialLevel.INTERMEDIATE.value,
                            "content": "/static/slideshows/enterprise_overview.html",
                            "estimated_time": 25
                        },
                        {
                            "id": "enterprise_case_studies",
                            "title": "Примеры использования для корпораций",
                            "description": "Реальные примеры использования в корпоративной среде",
                            "type": TutorialType.TEXT.value,
                            "level": TutorialLevel.ADVANCED.value,
                            "content": "Примеры использования SEO AI Models в корпоративной среде...",
                            "estimated_time": 30
                        }
                    ]
                }
            ]
        }
    
    def _adapt_materials_for_plan(self):
        """Адаптирует материалы для плана пользователя."""
        # Фильтруем топики, применимые для текущего плана
        filtered_topics = []
        
        for topic in self.knowledge_base["topics"]:
            if self.plan in topic["applicable_plans"]:
                # Фильтруем материалы по уровню опыта
                filtered_materials = [
                    material for material in topic["materials"]
                    if material["level"] in self.experience_level_mapping()
                ]
                
                if filtered_materials:
                    # Создаем копию топика с отфильтрованными материалами
                    filtered_topic = topic.copy()
                    filtered_topic["materials"] = filtered_materials
                    filtered_topics.append(filtered_topic)
        
        # Обновляем базу знаний
        self.knowledge_base["topics"] = filtered_topics
    
    def experience_level_mapping(self) -> List[str]:
        """
        Возвращает список уровней сложности материалов для текущего уровня опыта.
        
        Returns:
            Список подходящих уровней сложности
        """
        if self.experience_level == "beginner":
            return [TutorialLevel.BEGINNER.value]
        elif self.experience_level == "intermediate":
            return [TutorialLevel.BEGINNER.value, TutorialLevel.INTERMEDIATE.value]
        elif self.experience_level == "advanced":
            return [TutorialLevel.BEGINNER.value, TutorialLevel.INTERMEDIATE.value, TutorialLevel.ADVANCED.value]
        else:
            return [TutorialLevel.BEGINNER.value]
    
    def get_topics(self) -> List[Dict[str, Any]]:
        """
        Возвращает список доступных топиков.
        
        Returns:
            Список топиков
        """
        # Возвращаем основную информацию о топиках без материалов
        return [
            {
                "id": topic["id"],
                "title": topic["title"],
                "description": topic["description"],
                "materials_count": len(topic["materials"])
            }
            for topic in self.knowledge_base["topics"]
        ]
    
    def get_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о конкретном топике.
        
        Args:
            topic_id: Идентификатор топика
            
        Returns:
            Информация о топике или None, если топик не найден
        """
        for topic in self.knowledge_base["topics"]:
            if topic["id"] == topic_id:
                return topic
        
        return None
    
    def get_material(self, topic_id: str, material_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о конкретном материале.
        
        Args:
            topic_id: Идентификатор топика
            material_id: Идентификатор материала
            
        Returns:
            Информация о материале или None, если материал не найден
        """
        topic = self.get_topic(topic_id)
        
        if not topic:
            return None
        
        for material in topic["materials"]:
            if material["id"] == material_id:
                return material
        
        return None
    
    def generate_tutorial_path(self, max_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Генерирует оптимальный путь обучения.
        
        Args:
            max_time: Максимальное время обучения в минутах
            
        Returns:
            Список рекомендуемых материалов
        """
        # Создаем список всех доступных материалов
        all_materials = []
        
        for topic in self.knowledge_base["topics"]:
            for material in topic["materials"]:
                # Добавляем информацию о топике в материал
                material_with_topic = material.copy()
                material_with_topic["topic_id"] = topic["id"]
                material_with_topic["topic_title"] = topic["title"]
                all_materials.append(material_with_topic)
        
        # Сортируем материалы по уровню сложности
        level_order = {
            TutorialLevel.BEGINNER.value: 1,
            TutorialLevel.INTERMEDIATE.value: 2,
            TutorialLevel.ADVANCED.value: 3
        }
        
        sorted_materials = sorted(all_materials, key=lambda x: level_order[x["level"]])
        
        # Если указано максимальное время, ограничиваем список материалов
        if max_time is not None:
            result = []
            total_time = 0
            
            for material in sorted_materials:
                material_time = material["estimated_time"]
                
                if total_time + material_time <= max_time:
                    result.append(material)
                    total_time += material_time
                else:
                    break
            
            return result
        
        return sorted_materials
    
    def generate_topic_tutorial(self, topic_id: str) -> Dict[str, Any]:
        """
        Генерирует обучающие материалы для конкретного топика.
        
        Args:
            topic_id: Идентификатор топика
            
        Returns:
            Обучающие материалы по топику
        """
        topic = self.get_topic(topic_id)
        
        if not topic:
            return {
                "status": "error",
                "message": f"Топик {topic_id} не найден"
            }
        
        # Сортируем материалы по уровню сложности
        level_order = {
            TutorialLevel.BEGINNER.value: 1,
            TutorialLevel.INTERMEDIATE.value: 2,
            TutorialLevel.ADVANCED.value: 3
        }
        
        sorted_materials = sorted(topic["materials"], key=lambda x: level_order[x["level"]])
        
        return {
            "status": "success",
            "topic": {
                "id": topic["id"],
                "title": topic["title"],
                "description": topic["description"]
            },
            "materials": sorted_materials,
            "total_time": sum(material["estimated_time"] for material in sorted_materials)
        }
    
    def generate_personalized_tutorial(self) -> Dict[str, Any]:
        """
        Генерирует персонализированные обучающие материалы для пользователя.
        
        Returns:
            Персонализированные обучающие материалы
        """
        # Определяем приоритетные топики на основе плана и опыта
        priority_topics = []
        
        # Для начинающих пользователей приоритет - основы
        if self.experience_level == "beginner":
            priority_topics = ["seo_basics", "content_analysis"]
        # Для пользователей среднего уровня приоритет - дополнительные функции
        elif self.experience_level == "intermediate":
            if self.plan in ["micro", "basic", "pro", "enterprise"]:
                priority_topics = ["llm_optimization", "content_analysis"]
            else:
                priority_topics = ["content_analysis", "seo_basics"]
        # Для опытных пользователей приоритет - продвинутые функции
        elif self.experience_level == "advanced":
            if self.plan in ["basic", "pro", "enterprise"]:
                priority_topics = ["api_integration", "llm_optimization"]
            elif self.plan == "enterprise":
                priority_topics = ["enterprise_features", "api_integration"]
            else:
                priority_topics = ["content_analysis", "llm_optimization"]
        
        # Собираем материалы для приоритетных топиков
        priority_materials = []
        
        for topic_id in priority_topics:
            topic = self.get_topic(topic_id)
            
            if topic:
                for material in topic["materials"]:
                    # Добавляем информацию о топике в материал
                    material_with_topic = material.copy()
                    material_with_topic["topic_id"] = topic["id"]
                    material_with_topic["topic_title"] = topic["title"]
                    priority_materials.append(material_with_topic)
        
        # Добавляем материалы из других топиков
        additional_materials = []
        
        for topic in self.knowledge_base["topics"]:
            if topic["id"] not in priority_topics:
                for material in topic["materials"]:
                    # Добавляем информацию о топике в материал
                    material_with_topic = material.copy()
                    material_with_topic["topic_id"] = topic["id"]
                    material_with_topic["topic_title"] = topic["title"]
                    additional_materials.append(material_with_topic)
        
        # Сортируем материалы по уровню сложности
        level_order = {
            TutorialLevel.BEGINNER.value: 1,
            TutorialLevel.INTERMEDIATE.value: 2,
            TutorialLevel.ADVANCED.value: 3
        }
        
        priority_materials.sort(key=lambda x: level_order[x["level"]])
        additional_materials.sort(key=lambda x: level_order[x["level"]])
        
        return {
            "status": "success",
            "user_info": {
                "plan": self.plan,
                "experience_level": self.experience_level
            },
            "recommended_path": {
                "priority_materials": priority_materials,
                "additional_materials": additional_materials,
                "total_time": sum(material["estimated_time"] for material in priority_materials) +
                             sum(material["estimated_time"] for material in additional_materials)
            }
        }
    
    def generate_quick_start_guide(self) -> Dict[str, Any]:
        """
        Генерирует краткое руководство для быстрого старта.
        
        Returns:
            Краткое руководство
        """
        # Определяем базовые топики для быстрого старта
        basic_topics = ["seo_basics", "content_analysis"]
        
        # Добавляем продвинутые топики в зависимости от плана
        if self.plan in ["micro", "basic", "pro", "enterprise"]:
            basic_topics.append("llm_optimization")
        
        if self.plan in ["basic", "pro", "enterprise"]:
            basic_topics.append("api_integration")
        
        if self.plan == "enterprise":
            basic_topics.append("enterprise_features")
        
        # Собираем материалы для быстрого старта
        quick_start_materials = []
        
        for topic_id in basic_topics:
            topic = self.get_topic(topic_id)
            
            if topic:
                # Выбираем первый материал начального уровня для каждого топика
                beginner_materials = [m for m in topic["materials"] if m["level"] == TutorialLevel.BEGINNER.value]
                
                if beginner_materials:
                    material = beginner_materials[0]
                    # Добавляем информацию о топике в материал
                    material_with_topic = material.copy()
                    material_with_topic["topic_id"] = topic["id"]
                    material_with_topic["topic_title"] = topic["title"]
                    quick_start_materials.append(material_with_topic)
        
        return {
            "status": "success",
            "user_info": {
                "plan": self.plan,
                "experience_level": self.experience_level
            },
            "quick_start_guide": {
                "title": "Быстрый старт с SEO AI Models",
                "description": "Ознакомьтесь с основными возможностями SEO AI Models",
                "materials": quick_start_materials,
                "total_time": sum(material["estimated_time"] for material in quick_start_materials)
            }
        }
    
    def generate_feature_tutorial(self, feature_id: str) -> Dict[str, Any]:
        """
        Генерирует обучающие материалы для конкретной функции.
        
        Args:
            feature_id: Идентификатор функции
            
        Returns:
            Обучающие материалы по функции
        """
        # Карта соответствия функций и топиков
        feature_to_topic = {
            "content_analysis": "content_analysis",
            "recommendations": "seo_basics",
            "eeat_analysis": "content_analysis",
            "llm_integration": "llm_optimization",
            "api_access": "api_integration",
            "enterprise_features": "enterprise_features"
        }
        
        # Проверяем, существует ли функция в карте
        if feature_id not in feature_to_topic:
            return {
                "status": "error",
                "message": f"Функция {feature_id} не найдена"
            }
        
        # Получаем соответствующий топик
        topic_id = feature_to_topic[feature_id]
        topic = self.get_topic(topic_id)
        
        if not topic:
            return {
                "status": "error",
                "message": f"Топик для функции {feature_id} не найден"
            }
        
        # Фильтруем материалы по уровню опыта
        filtered_materials = [
            material for material in topic["materials"]
            if material["level"] in self.experience_level_mapping()
        ]
        
        # Если нет материалов для текущего уровня опыта, возвращаем ошибку
        if not filtered_materials:
            return {
                "status": "error",
                "message": f"Нет материалов для функции {feature_id} для уровня опыта {self.experience_level}"
            }
        
        return {
            "status": "success",
            "feature_id": feature_id,
            "topic": {
                "id": topic["id"],
                "title": topic["title"],
                "description": topic["description"]
            },
            "materials": filtered_materials,
            "total_time": sum(material["estimated_time"] for material in filtered_materials)
        }
