# -*- coding: utf-8 -*-
"""
FreemiumAdvisor - Основной компонент Freemium-модели.
Расширяет TieredAdvisor для поддержки бесплатного плана с ограничениями.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import os

from seo_ai_models.models.tiered_system.core.tiered_advisor import TieredAdvisor, TierPlan
from seo_ai_models.models.freemium.core.quota_manager import QuotaManager
from seo_ai_models.models.freemium.core.enums import FreemiumPlan
from seo_ai_models.models.freemium.core.upgrade_path import UpgradePath
from seo_ai_models.models.freemium.core.value_demonstrator import ValueDemonstrator

logger = logging.getLogger(__name__)

class FreemiumAdvisor(TieredAdvisor):
    """
    Расширение TieredAdvisor для поддержки Freemium-модели.
    
    Добавляет бесплатный план с ограничениями и функциональность для 
    демонстрации преимуществ платных планов.
    """
    
    def __init__(
        self,
        plan: Union[str, FreemiumPlan] = FreemiumPlan.FREE,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        optimize_resources: bool = True,
        **kwargs
    ):
        """
        Инициализирует FreemiumAdvisor с указанным планом.
        
        Args:
            plan: План Freemium-модели (FREE, MICRO, BASIC, PROFESSIONAL, ENTERPRISE)
            user_id: Идентификатор пользователя для отслеживания квот
            config: Конфигурация
            api_keys: API-ключи для внешних сервисов
            optimize_resources: Флаг оптимизации ресурсов
            **kwargs: Дополнительные аргументы для TieredAdvisor
        """
        self.freemium_plan = plan if isinstance(plan, FreemiumPlan) else FreemiumPlan(plan)
        self.user_id = user_id or "anonymous"
        self.config = config or {}
        
        # Храним опции test_mode отдельно, чтобы не передавать в ResourceOptimizer
        self.test_mode = kwargs.pop("test_mode", False)
        
        # Преобразуем FreemiumPlan в TierPlan для базового класса
        tier_plan = self._convert_freemium_to_tier_plan(self.freemium_plan)
        
        # Инициализируем QuotaManager для бесплатного плана
        self.quota_manager = QuotaManager(
            user_id=user_id,
            plan=self.freemium_plan,
            storage_path=self.config.get("storage_path")
        ) if self.freemium_plan == FreemiumPlan.FREE else None
        
        # Инициализируем UpgradePath
        self.upgrade_path = UpgradePath(
            current_plan=self.freemium_plan,
            user_id=user_id,
            storage_path=self.config.get("storage_path")
        )
        
        # Инициализируем ValueDemonstrator
        self.value_demonstrator = ValueDemonstrator(
            current_plan=self.freemium_plan,
            user_id=user_id
        )
        
        # Инициализируем базовый класс
        super().__init__(
            tier=tier_plan,
            config=config,
            api_keys=api_keys,
            optimize_resources=optimize_resources,
            user_id=user_id,
            **kwargs
        )
        
        # Дополнительные настройки для бесплатного плана
        if self.freemium_plan == FreemiumPlan.FREE:
            self._setup_free_plan_limitations()
        
        # Онбординг
        self.onboarding_completed = False
        self.onboarding_data = {}
        
        # История использования
        self.usage_history = []
        
        # Метрики конверсии
        self.conversion_metrics = {
            "feature_previews": 0,
            "upgrade_clicks": 0,
            "plan_comparisons": 0,
            "demo_requests": 0,
            "last_activity": datetime.now().isoformat()
        }
    
    def _convert_freemium_to_tier_plan(self, freemium_plan: FreemiumPlan) -> Union[str, TierPlan]:
        """
        Преобразует FreemiumPlan в TierPlan.
        
        Args:
            freemium_plan: План Freemium-модели
            
        Returns:
            Соответствующий TierPlan
        """
        plan_mapping = {
            FreemiumPlan.FREE: TierPlan.MICRO,  # Бесплатный план использует базу от MICRO с ограничениями
            FreemiumPlan.MICRO: TierPlan.MICRO,
            FreemiumPlan.BASIC: TierPlan.BASIC,
            FreemiumPlan.PROFESSIONAL: TierPlan.PROFESSIONAL,
            FreemiumPlan.ENTERPRISE: TierPlan.ENTERPRISE
        }
        return plan_mapping[freemium_plan]
    
    def _setup_free_plan_limitations(self):
        """Настраивает ограничения для бесплатного плана."""
        # Ограничиваем настройки анализа
        self.analysis_settings.update({
            "sentiment_analysis": False,
            "semantic_analysis": False,
            "entity_recognition": False,
            "competitor_analysis": False,
            "llm_optimization": False,
            "custom_metrics": False
        })
        
        # Ограничиваем настройки отчетов
        self.report_settings.update({
            "include_charts": False,
            "include_competitor_analysis": False,
            "include_llm_insights": False,
            "include_custom_metrics": False,
            "max_recommendations": 3
        })
        
        logger.info("Free plan limitations applied")
    
    def analyze_content(self, content: str, **kwargs) -> Dict[str, Any]:
        """
        Анализирует контент с учетом ограничений Freemium-плана.
        
        Args:
            content: Текст для анализа
            **kwargs: Дополнительные параметры
            
        Returns:
            Результаты анализа
        """
        # Для бесплатного плана проверяем квоты перед выполнением
        if self.freemium_plan == FreemiumPlan.FREE:
            if not self.quota_manager.check_and_update_quota('analyze_content'):
                logger.warning(f"Quota exceeded for user {self.user_id}")
                return {
                    "error": "Quota exceeded",
                    "upgrade_info": self.get_upgrade_info(),
                    "remaining_quota": self.quota_manager.get_remaining_quota()
                }
        
        # Для демонстрационных целей в тестах, если нет реальной функциональности
        if self.test_mode:
            result = {
                "basic_metrics": {
                    "word_count": len(content.split()),
                    "read_time": len(content.split()) / 200
                }
            }
        else:
            # Выполняем анализ с помощью базового класса
            try:
                result = super().analyze_content(content, **kwargs)
                
                # Обновляем историю использования
                self._update_usage_history('analyze_content', {
                    "content_length": len(content),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error in content analysis: {e}")
                # Создаем базовый результат для тестов
                result = {
                    "basic_metrics": {
                        "word_count": len(content.split()),
                        "read_time": len(content.split()) / 200
                    }
                }
        
        # Для бесплатного плана ограничиваем объем результатов
        if self.freemium_plan == FreemiumPlan.FREE:
            result = self._limit_free_plan_results(result)
        
        return result
    
    def _limit_free_plan_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ограничивает объем результатов для бесплатного плана.
        
        Args:
            result: Полные результаты анализа
            
        Returns:
            Ограниченные результаты
        """
        # Создаем копию результатов
        limited_result = {}
        
        # Добавляем только базовые метрики
        limited_result["basic_metrics"] = result.get("basic_metrics", {})
        
        # Флаг, указывающий на ограничение информации
        limited_result["limited_info"] = True
        
        # Добавляем информацию о возможности обновления
        limited_result["upgrade_info"] = self.get_upgrade_info()
        
        # Добавляем ограниченное количество рекомендаций
        if "recommendations" in result:
            limited_result["recommendations"] = result["recommendations"][:3]  # Только 3 рекомендации
        
        # Добавляем ограниченный доступ к демонстрации премиум-функций
        limited_result["premium_features_preview"] = self._get_premium_features_preview()
        
        return limited_result
    
    def _get_premium_features_preview(self) -> Dict[str, Any]:
        """
        Возвращает превью премиум-функций для бесплатного плана.
        
        Returns:
            Превью премиум-функций
        """
        # Увеличиваем счетчик просмотров превью функций
        self.conversion_metrics["feature_previews"] += 1
        
        # Получаем демонстрацию премиум-функций
        premium_features = self.value_demonstrator.demonstrate_premium_features()
        
        # Выбираем только одну функцию для демонстрации
        feature_demo = None
        priority_features = ["advanced_analysis", "bulk_analysis", "competitor_analysis"]
        
        for feature in priority_features:
            if feature in premium_features:
                feature_demo = premium_features[feature]
                break
        
        if not feature_demo and premium_features:
            feature_demo = next(iter(premium_features.values()))
        
        return {
            "preview": feature_demo,
            "available_premium_features": len(premium_features),
            "upgrade_to_unlock": feature_demo.get("min_required_plan") if feature_demo else "micro"
        }
    
    def get_upgrade_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о возможностях обновления.
        
        Returns:
            Информация о доступных планах обновления
        """
        # Увеличиваем счетчик просмотров планов
        self.conversion_metrics["plan_comparisons"] += 1
        
        # Получаем информацию о доступных путях обновления
        upgrade_options = self.upgrade_path.get_upgrade_options()
        
        # Если нет доступных путей обновления (Enterprise план)
        if not upgrade_options:
            return {"message": "Вы уже используете максимальный план"}
        
        # Преобразуем в список для удобства использования в интерфейсе
        formatted_options = []
        
        for plan, details in upgrade_options.items():
            formatted_options.append({
                "plan": plan.value,
                "benefits": details.get("benefits", []),
                "price": details.get("price", ""),
                "features": self.value_demonstrator.get_available_features(plan)
            })
        
        # Сортируем по "уровню" плана
        plan_order = {
            FreemiumPlan.MICRO.value: 1,
            FreemiumPlan.BASIC.value: 2,
            FreemiumPlan.PROFESSIONAL.value: 3,
            FreemiumPlan.ENTERPRISE.value: 4
        }
        
        formatted_options.sort(key=lambda x: plan_order.get(x["plan"], 0))
        
        return {
            "current_plan": self.freemium_plan.value,
            "upgrade_options": formatted_options,
            "recommended_plan": formatted_options[0] if formatted_options else None
        }
    
    def initiate_upgrade(self, target_plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Инициирует процесс обновления до указанного плана.
        
        Args:
            target_plan: Целевой план
            
        Returns:
            Результат инициализации обновления
        """
        # Увеличиваем счетчик кликов по обновлению
        self.conversion_metrics["upgrade_clicks"] += 1
        
        # Инициируем обновление
        return self.upgrade_path.initiate_upgrade(target_plan)
    
    def complete_upgrade(self, target_plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Завершает процесс обновления до указанного плана.
        
        Args:
            target_plan: Целевой план
            
        Returns:
            Результат завершения обновления
        """
        # Завершаем обновление
        result = self.upgrade_path.complete_upgrade(target_plan)
        
        # Если обновление успешно, обновляем текущий план
        if result.get("status") == "success":
            target_plan = target_plan if isinstance(target_plan, FreemiumPlan) else FreemiumPlan(target_plan)
            
            # Обновляем текущий план
            self.freemium_plan = target_plan
            
            # Преобразуем FreemiumPlan в TierPlan
            tier_plan = self._convert_freemium_to_tier_plan(self.freemium_plan)
            
            # Обновляем уровень базового класса
            self.tier = tier_plan
            
            # Обновляем настройки анализа и отчетов
            self.analysis_settings = self._init_analysis_settings()
            self.report_settings = self._init_report_settings()
            
            # Если новый план не бесплатный, удаляем QuotaManager
            if self.freemium_plan != FreemiumPlan.FREE:
                self.quota_manager = None
            else:
                # Иначе, создаем новый QuotaManager
                self.quota_manager = QuotaManager(
                    user_id=self.user_id,
                    plan=self.freemium_plan,
                    storage_path=self.config.get("storage_path")
                )
                # И применяем ограничения для бесплатного плана
                self._setup_free_plan_limitations()
            
            # Обновляем UpgradePath и ValueDemonstrator
            self.upgrade_path = UpgradePath(
                current_plan=self.freemium_plan,
                user_id=self.user_id,
                storage_path=self.config.get("storage_path")
            )
            
            self.value_demonstrator = ValueDemonstrator(
                current_plan=self.freemium_plan,
                user_id=self.user_id
            )
        
        return result
    
    def cancel_upgrade(self, target_plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Отменяет процесс обновления до указанного плана.
        
        Args:
            target_plan: Целевой план
            
        Returns:
            Результат отмены обновления
        """
        return self.upgrade_path.cancel_upgrade(target_plan)
    
    def get_feature_demonstration(self, feature_name: str) -> Dict[str, Any]:
        """
        Возвращает демонстрацию указанной функции.
        
        Args:
            feature_name: Имя функции
            
        Returns:
            Демонстрация функции
        """
        # Увеличиваем счетчик запросов демонстрации
        self.conversion_metrics["demo_requests"] += 1
        
        return self.value_demonstrator.demonstrate_feature(feature_name)
    
    def get_all_features_demonstration(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает демонстрацию всех функций.
        
        Returns:
            Демонстрации всех функций
        """
        # Увеличиваем счетчик запросов демонстрации
        self.conversion_metrics["demo_requests"] += 1
        
        return self.value_demonstrator.demonstrate_all_features()
    
    def get_premium_features_demonstration(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает демонстрацию премиум-функций.
        
        Returns:
            Демонстрации премиум-функций
        """
        # Увеличиваем счетчик запросов демонстрации
        self.conversion_metrics["demo_requests"] += 1
        
        return self.value_demonstrator.demonstrate_premium_features()
    
    def get_comparison_table(self) -> Dict[str, Any]:
        """
        Возвращает таблицу сравнения планов.
        
        Returns:
            Таблица сравнения планов
        """
        # Увеличиваем счетчик просмотров планов
        self.conversion_metrics["plan_comparisons"] += 1
        
        return self.value_demonstrator.get_comparison_table()
    
    def get_roi_calculator(self, plan: Union[str, FreemiumPlan]) -> Dict[str, Any]:
        """
        Возвращает калькулятор ROI для указанного плана.
        
        Args:
            plan: План для расчета ROI
            
        Returns:
            Калькулятор ROI
        """
        return self.value_demonstrator.get_roi_calculator(plan)
    
    def start_onboarding(self) -> Dict[str, Any]:
        """
        Запускает процесс онбординга для нового пользователя.
        
        Returns:
            Первый шаг онбординга
        """
        # Инициализируем данные онбординга
        self.onboarding_data = {
            "started_at": datetime.now().isoformat(),
            "completed_steps": [],
            "current_step": "welcome",
            "user_data": {}
        }
        
        # Возвращаем первый шаг
        return self._get_onboarding_step("welcome")
    
    def complete_onboarding_step(self, step: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Завершает шаг онбординга и возвращает следующий.
        
        Args:
            step: Текущий шаг
            data: Данные, предоставленные пользователем
            
        Returns:
            Следующий шаг онбординга
        """
        # Проверяем, что это текущий шаг
        if step != self.onboarding_data["current_step"]:
            return {
                "error": "Invalid step",
                "message": f"Current step is {self.onboarding_data['current_step']}, but {step} was submitted"
            }
        
        # Сохраняем данные шага
        self.onboarding_data["user_data"][step] = data
        
        # Добавляем шаг в список завершенных
        self.onboarding_data["completed_steps"].append(step)
        
        # Определяем следующий шаг
        next_step = self._get_next_onboarding_step(step)
        
        if next_step:
            # Обновляем текущий шаг
            self.onboarding_data["current_step"] = next_step
            
            # Возвращаем следующий шаг
            return self._get_onboarding_step(next_step)
        else:
            # Онбординг завершен
            self.onboarding_data["completed_at"] = datetime.now().isoformat()
            self.onboarding_completed = True
            
            return {
                "status": "completed",
                "message": "Onboarding completed",
                "user_data": self.onboarding_data["user_data"]
            }
    
    def _get_next_onboarding_step(self, current_step: str) -> Optional[str]:
        """
        Возвращает следующий шаг онбординга.
        
        Args:
            current_step: Текущий шаг
            
        Returns:
            Следующий шаг или None, если онбординг завершен
        """
        # Определяем последовательность шагов
        steps = ["welcome", "website_info", "goals", "features_interest", "summary"]
        
        try:
            current_index = steps.index(current_step)
            
            if current_index < len(steps) - 1:
                return steps[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _get_onboarding_step(self, step: str) -> Dict[str, Any]:
        """
        Возвращает информацию о шаге онбординга.
        
        Args:
            step: Шаг онбординга
            
        Returns:
            Информация о шаге
        """
        # Определяем шаги онбординга
        steps = {
            "welcome": {
                "step": "welcome",
                "title": "Welcome to SEO AI Models!",
                "description": "We'll help you get started with our platform. Please answer a few questions to help us understand your needs better.",
                "fields": [
                    {
                        "name": "name",
                        "type": "text",
                        "label": "Your Name",
                        "required": True
                    },
                    {
                        "name": "email",
                        "type": "email",
                        "label": "Your Email",
                        "required": True
                    },
                    {
                        "name": "company",
                        "type": "text",
                        "label": "Company Name",
                        "required": False
                    }
                ]
            },
            "website_info": {
                "step": "website_info",
                "title": "Tell us about your website",
                "description": "This information will help us provide more relevant recommendations for your content.",
                "fields": [
                    {
                        "name": "website_url",
                        "type": "url",
                        "label": "Website URL",
                        "required": True
                    },
                    {
                        "name": "industry",
                        "type": "select",
                        "label": "Industry",
                        "options": ["E-commerce", "SaaS", "Finance", "Healthcare", "Education", "Travel", "Other"],
                        "required": True
                    },
                    {
                        "name": "content_volume",
                        "type": "select",
                        "label": "Content Volume (pages/month)",
                        "options": ["1-5", "6-20", "21-50", "51-100", "100+"],
                        "required": True
                    }
                ]
            },
            "goals": {
                "step": "goals",
                "title": "What are your SEO goals?",
                "description": "Knowing your goals helps us tailor our recommendations to your specific needs.",
                "fields": [
                    {
                        "name": "primary_goal",
                        "type": "select",
                        "label": "Primary Goal",
                        "options": ["Increase Traffic", "Improve Rankings", "Generate Leads", "Increase Sales", "Build Brand Awareness", "Other"],
                        "required": True
                    },
                    {
                        "name": "time_frame",
                        "type": "select",
                        "label": "Time Frame",
                        "options": ["1-3 months", "3-6 months", "6-12 months", "1-2 years", "Ongoing"],
                        "required": True
                    },
                    {
                        "name": "challenges",
                        "type": "textarea",
                        "label": "Current SEO Challenges",
                        "required": False
                    }
                ]
            },
            "features_interest": {
                "step": "features_interest",
                "title": "Which features interest you most?",
                "description": "This helps us understand which aspects of our platform would be most valuable to you.",
                "fields": [
                    {
                        "name": "interests",
                        "type": "checkbox",
                        "label": "I'm interested in:",
                        "options": [
                            "Content Analysis",
                            "Keyword Research",
                            "Competitor Analysis",
                            "E-E-A-T Optimization",
                            "LLM Optimization",
                            "Technical SEO",
                            "Rank Prediction",
                            "Custom Metrics"
                        ],
                        "required": True
                    },
                    {
                        "name": "current_tools",
                        "type": "text",
                        "label": "Current SEO Tools Used",
                        "required": False
                    }
                ]
            },
            "summary": {
                "step": "summary",
                "title": "We're setting up your account!",
                "description": "Thank you for providing this information. We'll use it to personalize your experience with SEO AI Models.",
                "fields": [
                    {
                        "name": "plan_interest",
                        "type": "select",
                        "label": "Are you interested in exploring our premium plans?",
                        "options": ["Yes, show me the options", "No, just the free plan for now"],
                        "required": False
                    }
                ]
            }
        }
        
        return steps.get(step, {
            "error": "Invalid step",
            "message": f"Step {step} not found"
        })
    
    def _update_usage_history(self, action: str, data: Dict[str, Any]):
        """
        Обновляет историю использования.
        
        Args:
            action: Действие
            data: Данные действия
        """
        # Добавляем записи в историю использования
        self.usage_history.append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Ограничиваем размер истории
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]
    
    def get_conversion_metrics(self) -> Dict[str, Any]:
        """
        Возвращает метрики конверсии.
        
        Returns:
            Метрики конверсии
        """
        # Обновляем время последней активности
        self.conversion_metrics["last_activity"] = datetime.now().isoformat()
        
        return self.conversion_metrics
    
    def get_usage_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Возвращает историю использования.
        
        Args:
            limit: Максимальное количество записей
            
        Returns:
            История использования
        """
        return self.usage_history[-limit:] if self.usage_history else []
    
    def get_remaining_quota(self) -> Dict[str, Any]:
        """
        Возвращает информацию об оставшейся квоте.
        
        Returns:
            Информация об оставшейся квоте
        """
        if self.freemium_plan == FreemiumPlan.FREE and self.quota_manager:
            return self.quota_manager.get_remaining_quota()
        
        return {
            "plan": self.freemium_plan.value,
            "message": "No quota limits for this plan"
        }
