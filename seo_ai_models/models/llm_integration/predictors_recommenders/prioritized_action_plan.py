"""
План действий с приоритизацией по эффективности.

Модуль предоставляет функционал для создания плана действий
по оптимизации контента с приоритизацией задач по их эффективности,
стоимости и сложности внедрения.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Импортируем необходимые компоненты
from ..service.llm_service import LLMService
from ..service.prompt_generator import PromptGenerator
from .roi_calculator import ROICalculator
from .hybrid_recommender import HybridRecommender
from ..common.utils import parse_json_response


class PrioritizedActionPlan:
    """
    План действий с приоритизацией по эффективности.
    """
    
    def __init__(self, llm_service: LLMService, prompt_generator: PromptGenerator,
               roi_calculator: ROICalculator, 
               hybrid_recommender: Optional[HybridRecommender] = None):
        """
        Инициализирует класс для создания плана действий.
        
        Args:
            llm_service: Экземпляр LLMService для взаимодействия с LLM
            prompt_generator: Экземпляр PromptGenerator для генерации промптов
            roi_calculator: Экземпляр ROICalculator для расчета ROI
            hybrid_recommender: Экземпляр HybridRecommender для генерации рекомендаций (опционально)
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.roi_calculator = roi_calculator
        self.hybrid_recommender = hybrid_recommender
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
        
        # Определение фаз плана
        self.plan_phases = {
            "immediate": {
                "name": "Немедленные действия",
                "timeframe": "1-2 недели",
                "description": "Задачи, которые можно выполнить быстро и получить быстрые результаты"
            },
            "short_term": {
                "name": "Краткосрочные действия",
                "timeframe": "1-2 месяца",
                "description": "Задачи средней сложности с хорошим ROI"
            },
            "mid_term": {
                "name": "Среднесрочные действия",
                "timeframe": "3-6 месяцев",
                "description": "Комплексные задачи с высоким ROI, требующие больше времени и ресурсов"
            },
            "long_term": {
                "name": "Долгосрочные действия",
                "timeframe": "6-12 месяцев",
                "description": "Стратегические задачи с потенциально высокой отдачей в долгосрочной перспективе"
            }
        }
    
    def create_action_plan(self, content: str, query: str, 
                         recommendations: Optional[List[Dict[str, Any]]] = None,
                         industry: Optional[str] = None,
                         business_data: Optional[Dict[str, Any]] = None,
                         resources_level: str = "medium") -> Dict[str, Any]:
        """
        Создает план действий с приоритизацией по эффективности.
        
        Args:
            content: Контент для анализа
            query: Ключевой запрос
            recommendations: Список рекомендаций (опционально)
            industry: Отрасль (опционально)
            business_data: Бизнес-данные для расчета ROI (опционально)
            resources_level: Уровень доступных ресурсов ("low", "medium", "high")
            
        Returns:
            Dict[str, Any]: План действий
        """
        # Если рекомендации не предоставлены и доступен HybridRecommender, получаем рекомендации
        if recommendations is None and self.hybrid_recommender is not None:
            self.logger.info("Генерация рекомендаций с помощью HybridRecommender")
            recommendation_result = self.hybrid_recommender.generate_recommendations(
                content=content,
                query=query,
                industry=industry,
                balance_mode="balanced",
                max_recommendations=15
            )
            recommendations = recommendation_result.get("recommendations", [])
        elif recommendations is None:
            # Если рекомендации не предоставлены и HybridRecommender недоступен, генерируем их напрямую
            self.logger.info("Генерация рекомендаций с помощью LLM")
            recommendations = self._generate_recommendations(content, query, industry)
        
        # Рассчитываем ROI для рекомендаций
        roi_data = self.roi_calculator.calculate_detailed_roi(
            current_content=content,
            recommendations=recommendations,
            query=query,
            industry=industry,
            business_data=business_data,
            timeframe_months=12
        )
        
        # Анализируем рекомендации и группируем связанные задачи
        analyzed_recommendations = self._analyze_and_group_recommendations(
            recommendations, roi_data.get("recommendation_details", [])
        )
        
        # Создаем фазы плана с распределением задач
        plan_phases = self._create_plan_phases(analyzed_recommendations, resources_level)
        
        # Генерируем детальный план внедрения
        implementation_plan = self._generate_implementation_plan(
            plan_phases, resources_level
        )
        
        # Генерируем timeline для визуализации плана
        timeline = self._generate_timeline(plan_phases)
        
        # Генерируем сводку плана
        summary = self._generate_plan_summary(plan_phases, roi_data)
        
        # Формируем итоговый план
        action_plan = {
            "query": query,
            "industry": industry,
            "resources_level": resources_level,
            "phases": plan_phases,
            "implementation_plan": implementation_plan,
            "timeline": timeline,
            "summary": summary,
            "estimated_roi": {
                "total_implementation_costs": roi_data.get("implementation_costs", 0),
                "total_additional_revenue": roi_data.get("total", {}).get("additional_revenue", 0),
                "roi_percent": roi_data.get("roi_percent", 0),
                "payback_period_months": roi_data.get("payback_period_months", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return action_plan
    
    def _generate_recommendations(self, content: str, query: str,
                               industry: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации с помощью LLM.
        
        Args:
            content: Контент для анализа
            query: Ключевой запрос
            industry: Отрасль (опционально)
            
        Returns:
            List[Dict[str, Any]]: Список рекомендаций
        """
        # Формируем промпт для генерации рекомендаций
        prompt = f"""
        Ты эксперт по оптимизации контента для поисковых систем и LLM-поисковиков.
        
        Проанализируй следующий контент и предложи рекомендации по его оптимизации
        для поисковых систем и LLM-поисковиков для запроса: "{query}"
        
        {f"Отрасль: {industry}" if industry else ""}
        
        Предложи рекомендации по следующим категориям:
        1. Традиционное SEO:
           - on_page (оптимизация страницы)
           - technical (техническая оптимизация)
           - content (улучшение контента)
           - meta_tags (мета-теги)
           - user_experience (пользовательский опыт)
        
        2. LLM-оптимизация:
           - citability (цитируемость)
           - eeat (опыт, экспертиза, авторитетность, достоверность)
           - content_structure (структура контента)
           - information_quality (качество информации)
        
        Для каждой рекомендации укажи:
        - тип (traditional_seo или llm_optimization)
        - категорию
        - описание рекомендации
        - приоритет (от 1 до 5, где 5 - самый высокий)
        - ожидаемое влияние на ранжирование (от 1 до 5, где 5 - наибольшее влияние)
        - сложность внедрения (от 1 до 5, где 5 - самая сложная)
        
        Представь результат в формате JSON, например:
        [
            {{
                "type": "traditional_seo",
                "category": "content",
                "recommendation": "Добавить больше ключевых слов в заголовок",
                "priority": 4,
                "impact": 3,
                "implementation_difficulty": 1
            }},
            {{
                "type": "llm_optimization",
                "category": "citability",
                "recommendation": "Добавить больше фактических данных и ссылки на источники",
                "priority": 5,
                "impact": 4,
                "implementation_difficulty": 3
            }},
            ...
        ]
        
        Контент для анализа:
        {content}
        """
        
        # Выполняем запрос к LLM
        response = self.llm_service.query(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.5
        )
        
        # Извлекаем рекомендации из ответа
        recommendations_text = response.get("text", "")
        recommendations_data = parse_json_response(recommendations_text)
        
        # Если не удалось извлечь JSON, пытаемся структурировать ответ сами
        if not recommendations_data:
            recommendations_data = []
            
            # Пытаемся найти рекомендации в тексте
            recommendation_matches = re.findall(r'(\d+\.\s+)?(.*?):\s+(.*?)(?:\n|$)', recommendations_text)
            
            for match in recommendation_matches:
                recommendations_data.append({
                    "type": "traditional_seo",  # Тип по умолчанию
                    "category": "content",  # Категория по умолчанию
                    "recommendation": match[2],
                    "priority": 3,  # Приоритет по умолчанию
                    "impact": 3,  # Влияние по умолчанию
                    "implementation_difficulty": 2  # Сложность по умолчанию
                })
        
        # Преобразуем данные в нужный формат и добавляем токены и стоимость
        recommendations = []
        tokens_per_rec = response.get("tokens", {}).get("total", 0) / len(recommendations_data) if recommendations_data else 0
        cost_per_rec = response.get("cost", 0) / len(recommendations_data) if recommendations_data else 0
        
        for rec in recommendations_data:
            recommendations.append({
                "type": rec.get("type", "traditional_seo"),
                "category": rec.get("category", "content"),
                "recommendation": rec.get("recommendation", ""),
                "priority": rec.get("priority", 3),
                "impact": rec.get("impact", 3),
                "implementation_difficulty": rec.get("implementation_difficulty", 2),
                "tokens": {"total": tokens_per_rec},
                "cost": cost_per_rec
            })
        
        return recommendations
    
    def _analyze_and_group_recommendations(self, recommendations: List[Dict[str, Any]],
                                         roi_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Анализирует рекомендации и группирует связанные задачи.
        
        Args:
            recommendations: Список рекомендаций
            roi_details: Детали ROI для рекомендаций
            
        Returns:
            List[Dict[str, Any]]: Анализированные и сгруппированные рекомендации
        """
        # Создаем словарь для быстрого поиска деталей ROI по рекомендации
        roi_details_map = {}
        for detail in roi_details:
            roi_details_map[detail.get("recommendation", "")] = detail
        
        # Анализируем и расширяем рекомендации
        analyzed_recommendations = []
        for rec in recommendations:
            # Получаем детали ROI
            roi_detail = roi_details_map.get(rec.get("recommendation", ""), {})
            
            # Рассчитываем скор для приоритизации
            priority = rec.get("priority", 3)
            impact = rec.get("impact", 3)
            difficulty = rec.get("implementation_difficulty", 2)
            traffic_increase = roi_detail.get("total_traffic_increase", 0)
            
            # Формула приоритизации: (приоритет * влияние / сложность) * (1 + log(1 + увеличение трафика))
            # Это дает высокий скор задачам с высоким приоритетом, влиянием, низкой сложностью и большим увеличением трафика
            import math
            prioritization_score = (priority * impact / max(1, difficulty)) * (1 + math.log(1 + traffic_increase))
            
            # Определяем группу на основе категории рекомендации
            category = rec.get("category", "content")
            group = self._determine_recommendation_group(category, rec.get("type", "traditional_seo"))
            
            # Расширяем рекомендацию
            analyzed_recommendations.append({
                **rec,
                "roi_detail": roi_detail,
                "prioritization_score": prioritization_score,
                "group": group
            })
        
        # Сортируем по скору приоритизации
        analyzed_recommendations.sort(key=lambda x: x.get("prioritization_score", 0), reverse=True)
        
        return analyzed_recommendations
    
    def _determine_recommendation_group(self, category: str, rec_type: str) -> str:
        """
        Определяет группу для рекомендации.
        
        Args:
            category: Категория рекомендации
            rec_type: Тип рекомендации
            
        Returns:
            str: Группа рекомендации
        """
        # Группы для традиционного SEO
        if rec_type == "traditional_seo":
            if category in ["on_page", "meta_tags"]:
                return "on_page_optimization"
            elif category in ["technical"]:
                return "technical_optimization"
            elif category in ["content"]:
                return "content_optimization"
            elif category in ["user_experience"]:
                return "user_experience"
            else:
                return "other_traditional_seo"
        
        # Группы для LLM-оптимизации
        elif rec_type == "llm_optimization":
            if category in ["citability"]:
                return "citability_improvement"
            elif category in ["eeat"]:
                return "eeat_improvement"
            elif category in ["content_structure"]:
                return "content_structure_improvement"
            elif category in ["information_quality"]:
                return "information_quality_improvement"
            else:
                return "other_llm_optimization"
        
        # Группы для гибридных рекомендаций
        elif rec_type == "hybrid":
            if category in ["conflict_resolution"]:
                return "conflict_resolution"
            else:
                return "hybrid_optimization"
        
        # Группа по умолчанию
        else:
            return "other"
    
    def _create_plan_phases(self, analyzed_recommendations: List[Dict[str, Any]],
                          resources_level: str) -> Dict[str, Any]:
        """
        Создает фазы плана с распределением задач.
        
        Args:
            analyzed_recommendations: Анализированные рекомендации
            resources_level: Уровень доступных ресурсов
            
        Returns:
            Dict[str, Any]: Фазы плана с распределенными задачами
        """
        # Определяем множители для разных уровней ресурсов
        resource_multipliers = {
            "low": 0.5,  # Меньше задач в каждой фазе
            "medium": 1.0,  # Стандартное количество задач
            "high": 1.5  # Больше задач в каждой фазе
        }
        
        # Получаем множитель для указанного уровня ресурсов
        multiplier = resource_multipliers.get(resources_level, 1.0)
        
        # Определяем максимальное количество задач в каждой фазе
        immediate_tasks_max = round(3 * multiplier)
        short_term_tasks_max = round(5 * multiplier)
        mid_term_tasks_max = round(5 * multiplier)
        long_term_tasks_max = round(3 * multiplier)
        
        # Инициализируем фазы плана
        plan_phases = {
            "immediate": {
                **self.plan_phases["immediate"],
                "tasks": [],
                "total_costs": 0,
                "expected_benefits": 0
            },
            "short_term": {
                **self.plan_phases["short_term"],
                "tasks": [],
                "total_costs": 0,
                "expected_benefits": 0
            },
            "mid_term": {
                **self.plan_phases["mid_term"],
                "tasks": [],
                "total_costs": 0,
                "expected_benefits": 0
            },
            "long_term": {
                **self.plan_phases["long_term"],
                "tasks": [],
                "total_costs": 0,
                "expected_benefits": 0
            }
        }
        
        # Распределяем задачи по фазам
        for rec in analyzed_recommendations:
            # Определяем фазу для задачи
            difficulty = rec.get("implementation_difficulty", 2)
            
            # Правила распределения:
            # - Immediate: низкая сложность (1-2), высокий приоритет (4-5)
            # - Short-term: средняя сложность (2-3), высокий/средний приоритет (3-5)
            # - Mid-term: средняя/высокая сложность (3-4), средний/высокий приоритет (3-5)
            # - Long-term: высокая сложность (4-5), любой приоритет
            
            priority = rec.get("priority", 3)
            
            if difficulty <= 2 and priority >= 4 and len(plan_phases["immediate"]["tasks"]) < immediate_tasks_max:
                phase = "immediate"
            elif difficulty <= 3 and priority >= 3 and len(plan_phases["short_term"]["tasks"]) < short_term_tasks_max:
                phase = "short_term"
            elif difficulty <= 4 and priority >= 3 and len(plan_phases["mid_term"]["tasks"]) < mid_term_tasks_max:
                phase = "mid_term"
            elif len(plan_phases["long_term"]["tasks"]) < long_term_tasks_max:
                phase = "long_term"
            else:
                # Если все фазы заполнены, пропускаем задачу
                continue
            
            # Добавляем задачу в выбранную фазу
            task = {
                "id": f"task_{len(plan_phases[phase]['tasks']) + 1}",
                "recommendation": rec.get("recommendation", ""),
                "type": rec.get("type", "traditional_seo"),
                "category": rec.get("category", "content"),
                "group": rec.get("group", "other"),
                "priority": rec.get("priority", 3),
                "impact": rec.get("impact", 3),
                "implementation_difficulty": difficulty,
                "implementation_cost": rec.get("roi_detail", {}).get("implementation_cost", 0),
                "traffic_increase": rec.get("roi_detail", {}).get("total_traffic_increase", 0),
                "prioritization_score": rec.get("prioritization_score", 0)
            }
            
            plan_phases[phase]["tasks"].append(task)
            
            # Обновляем общую стоимость и ожидаемые выгоды фазы
            plan_phases[phase]["total_costs"] += task["implementation_cost"]
            plan_phases[phase]["expected_benefits"] += task["traffic_increase"]
        
        return plan_phases
    
    def _generate_implementation_plan(self, plan_phases: Dict[str, Any],
                                    resources_level: str) -> Dict[str, Any]:
        """
        Генерирует детальный план внедрения.
        
        Args:
            plan_phases: Фазы плана с распределенными задачами
            resources_level: Уровень доступных ресурсов
            
        Returns:
            Dict[str, Any]: Детальный план внедрения
        """
        # Определяем базовые параметры внедрения
        start_date = datetime.now()
        
        # Определяем длительность фаз в зависимости от уровня ресурсов
        phase_durations = {
            "low": {
                "immediate": 21,  # 3 недели
                "short_term": 90,  # 3 месяца
                "mid_term": 180,  # 6 месяцев
                "long_term": 365  # 12 месяцев
            },
            "medium": {
                "immediate": 14,  # 2 недели
                "short_term": 60,  # 2 месяца
                "mid_term": 120,  # 4 месяца
                "long_term": 270  # 9 месяцев
            },
            "high": {
                "immediate": 7,  # 1 неделя
                "short_term": 30,  # 1 месяц
                "mid_term": 90,  # 3 месяца
                "long_term": 180  # 6 месяцев
            }
        }
        
        # Получаем длительность фаз для указанного уровня ресурсов
        durations = phase_durations.get(resources_level, phase_durations["medium"])
        
        # Инициализируем план внедрения
        implementation_plan = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "resources_level": resources_level,
            "phases": {}
        }
        
        # Заполняем план внедрения
        current_date = start_date
        
        for phase_key, phase_data in plan_phases.items():
            # Определяем даты начала и окончания фазы
            phase_start_date = current_date
            phase_end_date = current_date + timedelta(days=durations[phase_key])
            
            # Создаем план для фазы
            phase_plan = {
                "name": phase_data["name"],
                "start_date": phase_start_date.strftime("%Y-%m-%d"),
                "end_date": phase_end_date.strftime("%Y-%m-%d"),
                "duration_days": durations[phase_key],
                "tasks": []
            }
            
            # Распределяем задачи по времени
            tasks = phase_data.get("tasks", [])
            
            if tasks:
                # Простое равномерное распределение
                days_per_task = durations[phase_key] / len(tasks)
                
                for i, task in enumerate(tasks):
                    # Определяем даты начала и окончания задачи
                    task_start_date = phase_start_date + timedelta(days=i * days_per_task)
                    
                    # Определяем длительность задачи в зависимости от сложности
                    difficulty = task.get("implementation_difficulty", 2)
                    task_duration = max(1, round(days_per_task * difficulty / 3))
                    
                    task_end_date = task_start_date + timedelta(days=task_duration)
                    
                    # Создаем план для задачи
                    task_plan = {
                        "id": task["id"],
                        "recommendation": task["recommendation"],
                        "start_date": task_start_date.strftime("%Y-%m-%d"),
                        "end_date": task_end_date.strftime("%Y-%m-%d"),
                        "duration_days": task_duration,
                        "implementation_cost": task["implementation_cost"],
                        "priority": task["priority"],
                        "group": task["group"]
                    }
                    
                    phase_plan["tasks"].append(task_plan)
            
            # Добавляем план фазы в общий план
            implementation_plan["phases"][phase_key] = phase_plan
            
            # Обновляем текущую дату для следующей фазы
            current_date = phase_end_date
        
        # Добавляем дату окончания плана
        implementation_plan["end_date"] = current_date.strftime("%Y-%m-%d")
        implementation_plan["total_duration_days"] = (current_date - start_date).days
        
        return implementation_plan
    
    def _generate_timeline(self, plan_phases: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует timeline для визуализации плана.
        
        Args:
            plan_phases: Фазы плана с распределенными задачами
            
        Returns:
            Dict[str, Any]: Timeline для визуализации
        """
        # Подготавливаем данные для timeline
        timeline_data = {
            "phases": []
        }
        
        # Формируем данные по фазам
        for phase_key, phase_data in plan_phases.items():
            timeline_data["phases"].append({
                "id": phase_key,
                "name": phase_data["name"],
                "timeframe": phase_data["timeframe"],
                "tasks_count": len(phase_data.get("tasks", [])),
                "total_costs": phase_data["total_costs"],
                "expected_benefits": phase_data["expected_benefits"]
            })
        
        # Формируем данные по группам задач
        task_groups = {}
        
        for phase_key, phase_data in plan_phases.items():
            for task in phase_data.get("tasks", []):
                group = task.get("group", "other")
                
                if group not in task_groups:
                    task_groups[group] = {
                        "name": self._get_group_display_name(group),
                        "tasks_count": 0,
                        "phases": {}
                    }
                
                # Увеличиваем счетчик задач в группе
                task_groups[group]["tasks_count"] += 1
                
                # Увеличиваем счетчик задач в группе для фазы
                if phase_key not in task_groups[group]["phases"]:
                    task_groups[group]["phases"][phase_key] = 0
                
                task_groups[group]["phases"][phase_key] += 1
        
        # Добавляем данные по группам в timeline
        timeline_data["groups"] = list(task_groups.values())
        
        return timeline_data
    
    def _get_group_display_name(self, group: str) -> str:
        """
        Возвращает отображаемое имя для группы задач.
        
        Args:
            group: Группа задач
            
        Returns:
            str: Отображаемое имя
        """
        # Таблица соответствия групп и отображаемых имен
        group_display_names = {
            "on_page_optimization": "Оптимизация страницы",
            "technical_optimization": "Техническая оптимизация",
            "content_optimization": "Улучшение контента",
            "user_experience": "Пользовательский опыт",
            "citability_improvement": "Улучшение цитируемости",
            "eeat_improvement": "Улучшение E-E-A-T",
            "content_structure_improvement": "Улучшение структуры контента",
            "information_quality_improvement": "Улучшение качества информации",
            "conflict_resolution": "Разрешение конфликтов",
            "hybrid_optimization": "Гибридная оптимизация",
            "other": "Другое",
            "other_traditional_seo": "Другие SEO-задачи",
            "other_llm_optimization": "Другие LLM-задачи"
        }
        
        return group_display_names.get(group, group)
    
    def _generate_plan_summary(self, plan_phases: Dict[str, Any],
                             roi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует сводку плана.
        
        Args:
            plan_phases: Фазы плана с распределенными задачами
            roi_data: Данные ROI
            
        Returns:
            Dict[str, Any]: Сводка плана
        """
        # Подсчитываем общее количество задач
        total_tasks = sum(len(phase_data.get("tasks", [])) for phase_data in plan_phases.values())
        
        # Подсчитываем общую стоимость внедрения
        total_costs = sum(phase_data["total_costs"] for phase_data in plan_phases.values())
        
        # Подсчитываем ожидаемые выгоды
        total_benefits = sum(phase_data["expected_benefits"] for phase_data in plan_phases.values())
        
        # Подсчитываем количество задач по типам
        traditional_seo_tasks = 0
        llm_optimization_tasks = 0
        hybrid_tasks = 0
        
        for phase_data in plan_phases.values():
            for task in phase_data.get("tasks", []):
                if task.get("type") == "traditional_seo":
                    traditional_seo_tasks += 1
                elif task.get("type") == "llm_optimization":
                    llm_optimization_tasks += 1
                elif task.get("type") == "hybrid":
                    hybrid_tasks += 1
        
        # Формируем сводку плана
        summary = {
            "total_tasks": total_tasks,
            "traditional_seo_tasks": traditional_seo_tasks,
            "llm_optimization_tasks": llm_optimization_tasks,
            "hybrid_tasks": hybrid_tasks,
            "total_implementation_costs": total_costs,
            "estimated_additional_revenue": roi_data.get("total", {}).get("additional_revenue", 0),
            "estimated_roi_percent": roi_data.get("roi_percent", 0),
            "payback_period_months": roi_data.get("payback_period_months", 0),
            "description": self._generate_summary_description(
                total_tasks, traditional_seo_tasks, llm_optimization_tasks, total_costs,
                roi_data.get("roi_percent", 0), roi_data.get("payback_period_months", 0)
            )
        }
        
        return summary
    
    def _generate_summary_description(self, total_tasks: int, traditional_seo_tasks: int,
                                    llm_optimization_tasks: int, total_costs: float,
                                    roi_percent: float, payback_period_months: float) -> str:
        """
        Генерирует текстовое описание сводки плана.
        
        Args:
            total_tasks: Общее количество задач
            traditional_seo_tasks: Количество задач по традиционному SEO
            llm_optimization_tasks: Количество задач по LLM-оптимизации
            total_costs: Общая стоимость внедрения
            roi_percent: Процент ROI
            payback_period_months: Срок окупаемости в месяцах
            
        Returns:
            str: Текстовое описание сводки плана
        """
        # Формируем текстовое описание
        description = f"План включает {total_tasks} задач: {traditional_seo_tasks} по традиционному SEO "
        description += f"и {llm_optimization_tasks} по LLM-оптимизации. "
        description += f"Общая стоимость внедрения составляет {total_costs:,.0f} руб. "
        description += f"Ожидаемый ROI: {roi_percent:,.0f}%. "
        
        if payback_period_months != float('inf'):
            description += f"Срок окупаемости: {payback_period_months:,.1f} месяцев."
        else:
            description += "Срок окупаемости не определен."
        
        return description
    
    def generate_resource_allocation(self, action_plan: Dict[str, Any],
                                   available_resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует план распределения ресурсов для внедрения.
        
        Args:
            action_plan: План действий
            available_resources: Доступные ресурсы
            
        Returns:
            Dict[str, Any]: План распределения ресурсов
        """
        # Определяем типы ресурсов и их доступность
        resource_types = available_resources.get("resource_types", {})
        
        # Инициализируем план распределения ресурсов
        resource_allocation = {
            "resource_types": resource_types,
            "phases": {}
        }
        
        # Распределяем ресурсы по фазам и задачам
        for phase_key, phase_data in action_plan.get("implementation_plan", {}).get("phases", {}).items():
            # Инициализируем распределение ресурсов для фазы
            phase_allocation = {
                "name": phase_data["name"],
                "start_date": phase_data["start_date"],
                "end_date": phase_data["end_date"],
                "duration_days": phase_data["duration_days"],
                "tasks": []
            }
            
            # Распределяем ресурсы по задачам
            for task in phase_data.get("tasks", []):
                # Определяем необходимые ресурсы для задачи
                task_resources = self._determine_task_resources(
                    task, resource_types, available_resources
                )
                
                # Добавляем распределение ресурсов для задачи
                task_allocation = {
                    "id": task["id"],
                    "recommendation": task["recommendation"],
                    "start_date": task["start_date"],
                    "end_date": task["end_date"],
                    "duration_days": task["duration_days"],
                    "resources": task_resources
                }
                
                phase_allocation["tasks"].append(task_allocation)
            
            # Добавляем распределение ресурсов для фазы
            resource_allocation["phases"][phase_key] = phase_allocation
        
        return resource_allocation
    
    def _determine_task_resources(self, task: Dict[str, Any],
                               resource_types: Dict[str, Any],
                               available_resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Определяет необходимые ресурсы для задачи.
        
        Args:
            task: Задача
            resource_types: Типы ресурсов
            available_resources: Доступные ресурсы
            
        Returns:
            Dict[str, Any]: Необходимые ресурсы для задачи
        """
        # В реальном проекте здесь был бы более сложный алгоритм
        # с учетом типа задачи, сложности, приоритета и доступности ресурсов
        
        # Для примера используем простое распределение
        task_resources = {}
        
        # Определяем тип задачи и группу
        task_type = task.get("type", "traditional_seo")
        task_group = task.get("group", "other")
        
        # Распределяем ресурсы в зависимости от типа задачи
        if task_type == "traditional_seo":
            # Для традиционного SEO нужны SEO-специалисты
            task_resources["seo_specialist"] = {
                "hours": task.get("duration_days", 1) * 4  # 4 часа в день
            }
            
            # Если это оптимизация контента, добавляем копирайтера
            if task_group == "content_optimization":
                task_resources["copywriter"] = {
                    "hours": task.get("duration_days", 1) * 2  # 2 часа в день
                }
            
            # Если это техническая оптимизация, добавляем разработчика
            if task_group == "technical_optimization":
                task_resources["developer"] = {
                    "hours": task.get("duration_days", 1) * 2  # 2 часа в день
                }
        
        elif task_type == "llm_optimization":
            # Для LLM-оптимизации нужны LLM-специалисты
            task_resources["llm_specialist"] = {
                "hours": task.get("duration_days", 1) * 4  # 4 часа в день
            }
            
            # Если это улучшение контента, добавляем копирайтера
            if task_group in ["citability_improvement", "information_quality_improvement"]:
                task_resources["copywriter"] = {
                    "hours": task.get("duration_days", 1) * 3  # 3 часа в день
                }
            
            # Если это улучшение E-E-A-T, добавляем эксперта
            if task_group == "eeat_improvement":
                task_resources["industry_expert"] = {
                    "hours": task.get("duration_days", 1) * 2  # 2 часа в день
                }
        
        elif task_type == "hybrid":
            # Для гибридных задач нужны и SEO, и LLM-специалисты
            task_resources["seo_specialist"] = {
                "hours": task.get("duration_days", 1) * 2  # 2 часа в день
            }
            
            task_resources["llm_specialist"] = {
                "hours": task.get("duration_days", 1) * 2  # 2 часа в день
            }
            
            # Если это разрешение конфликтов, добавляем PM
            if task_group == "conflict_resolution":
                task_resources["project_manager"] = {
                    "hours": task.get("duration_days", 1) * 1  # 1 час в день
                }
        
        return task_resources
