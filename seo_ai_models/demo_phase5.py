# -*- coding: utf-8 -*-
"""
Демонстрационный скрипт для компонентов Фазы 5 (Freemium-модель и Масштабирование/Мониторинг).
Этот скрипт показывает, как использовать компоненты, разработанные в Фазе 5.
"""

import os
import time
import logging
import json
from typing import Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорт компонентов Freemium-модели
from seo_ai_models.models.freemium.core.freemium_advisor import FreemiumAdvisor, FreemiumPlan
from seo_ai_models.models.freemium.core.quota_manager import QuotaManager
from seo_ai_models.models.freemium.core.upgrade_path import UpgradePath
from seo_ai_models.models.freemium.core.value_demonstrator import ValueDemonstrator
from seo_ai_models.models.freemium.onboarding.onboarding_wizard import OnboardingWizard
from seo_ai_models.models.freemium.onboarding.step_manager import StepManager
from seo_ai_models.models.freemium.onboarding.tutorial_generator import TutorialGenerator

# Импорт компонентов масштабирования и мониторинга
from seo_ai_models.models.scaling.performance.performance_optimizer import PerformanceOptimizer
from seo_ai_models.models.scaling.performance.distributed_processing import DistributedProcessing, Task, TaskPriority
from seo_ai_models.models.scaling.monitoring.system_monitor import SystemMonitor
from seo_ai_models.models.scaling.monitoring.auto_scaling import AutoScaling

class DemoPhase5:
    """
    Демонстрационный класс для компонентов Фазы 5 (Freemium-модель и Масштабирование/Мониторинг).
    """
    
    def __init__(self):
        """Инициализирует демонстрацию."""
        self.user_id = "demo_user_123"
    
    def demo_freemium_model(self):
        """Демонстрирует компоненты Freemium-модели."""
        logger.info("=== ДЕМОНСТРАЦИЯ КОМПОНЕНТОВ FREEMIUM-МОДЕЛИ ===")
        
        # 1. FreemiumAdvisor и QuotaManager
        logger.info("\n1. FreemiumAdvisor и QuotaManager:")
        
        # Создаем экземпляр FreemiumAdvisor
        freemium_advisor = FreemiumAdvisor(
            plan=FreemiumPlan.FREE,
            user_id=self.user_id
        )
        
        # Анализируем контент с учетом ограничений бесплатного плана
        sample_content = "Это пример контента для анализа SEO. Оптимизация для поисковых систем очень важна."
        analysis_result = freemium_advisor.analyze_content(content=sample_content)
        
        # Выводим информацию о результатах анализа и оставшихся квотах
        logger.info(f"Результат анализа для бесплатного плана: {'Ограниченный' if 'limited_info' in analysis_result and analysis_result['limited_info'] else 'Полный'}")
        
        if freemium_advisor.quota_manager:
            remaining_quota = freemium_advisor.quota_manager.get_remaining_quota('analyze_content')
            logger.info(f"Оставшаяся квота для анализа контента: {remaining_quota}")
        
        # 2. UpgradePath
        logger.info("\n2. UpgradePath:")
        
        # Создаем экземпляр UpgradePath
        upgrade_path = UpgradePath(user_id=self.user_id, current_plan=FreemiumPlan.FREE)
        
        # Получаем доступные пути обновления
        available_upgrades = upgrade_path.get_available_upgrade_paths()
        logger.info(f"Доступные планы для обновления: {[upgrade['code'] for upgrade in available_upgrades]}")
        
        # Имитируем процесс обновления до плана Micro
        upgrade_result = upgrade_path.initiate_upgrade(
            target_plan=FreemiumPlan.MICRO,
            payment_info={"card_last4": "1234", "amount": 1990}
        )
        logger.info(f"Результат инициализации обновления: {upgrade_result['status']}")
        
        # Проверяем статус обновления
        if "upgrade_id" in upgrade_result:
            upgrade_status = upgrade_path.get_upgrade_status(upgrade_result["upgrade_id"])
            logger.info(f"Статус обновления: {upgrade_status['status']}")
        
        # 3. ValueDemonstrator
        logger.info("\n3. ValueDemonstrator:")
        
        # Создаем экземпляр ValueDemonstrator
        value_demonstrator = ValueDemonstrator(
            current_plan=FreemiumPlan.FREE,
            user_id=self.user_id
        )
        
        # Демонстрируем функцию полных рекомендаций
        demo_result = value_demonstrator.demonstrate_feature(
            feature_name="full_recommendations",
            content=sample_content
        )
        
        # Выводим информацию о демонстрации
        logger.info(f"Демонстрация функции 'Полные рекомендации':")
        logger.info(f"  Количество рекомендаций в бесплатном плане: {demo_result['demo_result']['free_plan']['recommendations_count']}")
        logger.info(f"  Количество рекомендаций в платном плане: {demo_result['demo_result']['paid_plan']['recommendations_count']}")
        logger.info(f"  Улучшение: {demo_result['demo_result']['comparison']['improvement_percentage']:.1f}%")
        
        # 4. OnboardingWizard и StepManager
        logger.info("\n4. OnboardingWizard и StepManager:")
        
        # Создаем экземпляр OnboardingWizard
        onboarding_wizard = OnboardingWizard(
            user_id=self.user_id,
            plan="free"
        )
        
        # Получаем информацию о текущем шаге
        current_step = onboarding_wizard.get_current_step()
        logger.info(f"Текущий шаг онбординга: {current_step['current_step']}")
        logger.info(f"Заголовок шага: {current_step['content'].get('title', '')}")
        
        # Переходим к следующему шагу
        next_step_result = onboarding_wizard.advance_to_step(onboarding_wizard._get_next_step())
        logger.info(f"Переход к следующему шагу: {next_step_result['status']}")
        logger.info(f"Новый текущий шаг: {next_step_result['current_step']['current_step']}")
        
        # Создаем экземпляр StepManager
        step_manager = StepManager(
            user_id=self.user_id,
            plan="free"
        )
        
        # Получаем оптимальный путь онбординга
        onboarding_path = step_manager.get_onboarding_path(include_optional=False)
        logger.info(f"Оптимальный путь онбординга (только обязательные шаги): {[step['id'] for step in onboarding_path]}")
        
        # 5. TutorialGenerator
        logger.info("\n5. TutorialGenerator:")
        
        # Создаем экземпляр TutorialGenerator
        tutorial_generator = TutorialGenerator(
            user_id=self.user_id,
            plan="free",
            experience_level="beginner"
        )
        
        # Получаем список доступных топиков
        topics = tutorial_generator.get_topics()
        logger.info(f"Доступные топики для обучения: {[topic['title'] for topic in topics]}")
        
        # Генерируем руководство быстрого старта
        quick_start_guide = tutorial_generator.generate_quick_start_guide()
        logger.info(f"Руководство быстрого старта содержит {len(quick_start_guide['quick_start_guide']['materials'])} материалов")
        logger.info(f"Общее время изучения: {quick_start_guide['quick_start_guide']['total_time']} минут")
    
    def demo_scaling_components(self):
        """Демонстрирует компоненты масштабирования и мониторинга."""
        logger.info("\n=== ДЕМОНСТРАЦИЯ КОМПОНЕНТОВ МАСШТАБИРОВАНИЯ И МОНИТОРИНГА ===")
        
        # 1. PerformanceOptimizer
        logger.info("\n1. PerformanceOptimizer:")
        
        # Создаем экземпляр PerformanceOptimizer
        performance_optimizer = PerformanceOptimizer()
        
        # Создаем тестовую функцию для оптимизации
        def test_function(data):
            time.sleep(0.1)  # Имитация затратной операции
            return data * 2
        
        # Анализируем компонент
        analysis_result = performance_optimizer.analyze_component(test_function, "test_function", test_data=5)
        logger.info(f"Результат анализа компонента: среднее время выполнения {analysis_result['average_execution_time']:.4f} сек")
        
        # Оптимизируем компонент с использованием кэширования
        optimization_result = performance_optimizer.optimize_component(
            test_function,
            "test_function",
            strategy_names=["Caching"]
        )
        
        # Получаем оптимизированный компонент
        optimized_function = optimization_result["optimized_component"]
        
        # Проверяем производительность оптимизированного компонента
        start_time = time.time()
        result1 = optimized_function(5)  # Первый вызов (без кэша)
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        result2 = optimized_function(5)  # Второй вызов (из кэша)
        second_call_time = time.time() - start_time
        
        logger.info(f"Время первого вызова (без кэша): {first_call_time:.4f} сек")
        logger.info(f"Время второго вызова (из кэша): {second_call_time:.4f} сек")
        logger.info(f"Ускорение: {first_call_time / second_call_time:.1f}x")
        
        # 2. DistributedProcessing
        logger.info("\n2. DistributedProcessing:")
        
        # Создаем экземпляр DistributedProcessing
        distributed_processing = DistributedProcessing(worker_count=2)
        
        # Определяем тестовую функцию
        def process_data(data):
            time.sleep(0.1)  # Имитация затратной операции
            return {"result": data * 2}
        
        # Регистрируем функцию
        distributed_processing.register_function("process_data", process_data)
        
        # Запускаем систему
        distributed_processing.start()
        
        # Добавляем задачи
        tasks = []
        for i in range(5):
            task_result = distributed_processing.add_task(
                function_name="process_data",
                args=[i],
                priority=TaskPriority.NORMAL
            )
            tasks.append(task_result["task_id"])
        
        logger.info(f"Добавлено {len(tasks)} задач")
        
        # Ждем завершения всех задач
        distributed_processing.wait_for_all_tasks(timeout=5)
        
        # Получаем результаты задач
        for task_id in tasks:
            task_info = distributed_processing.get_task(task_id)
            if task_info and task_info.get("status") == "completed":
                logger.info(f"Задача {task_id}: результат = {task_info.get('result')}")
        
        # Останавливаем систему
        distributed_processing.stop()
        
        # 3. SystemMonitor
        logger.info("\n3. SystemMonitor:")
        
        # Создаем экземпляр SystemMonitor
        system_monitor = SystemMonitor(
            system_metrics_interval=2,
            application_metrics_interval=2
        )
        
        # Запускаем мониторинг
        system_monitor.start()
        
        # Записываем тестовые метрики приложения
        for _ in range(5):
            # Имитируем запросы
            system_monitor.record_request(successful=True, time_ms=100)
            
            # Имитируем запросы к базе данных
            system_monitor.record_query(successful=True, time_ms=50)
            
            time.sleep(0.5)
        
        # Ждем сбора метрик
        time.sleep(3)
        
        # Получаем последние метрики
        system_metrics = system_monitor.get_latest_metrics("system")
        application_metrics = system_monitor.get_latest_metrics("application")
        
        logger.info(f"Последние системные метрики:")
        if system_metrics["status"] == "success" and system_metrics["metrics"]:
            metrics = system_metrics["metrics"]
            if "cpu" in metrics:
                logger.info(f"  CPU: {metrics['cpu'].get('percent', 'Н/Д')}%")
            if "memory" in metrics:
                logger.info(f"  Memory: {metrics['memory'].get('percent', 'Н/Д')}%")
        
        logger.info(f"Последние метрики приложения:")
        if application_metrics["status"] == "success" and application_metrics["metrics"]:
            metrics = application_metrics["metrics"]
            if "requests" in metrics:
                logger.info(f"  Всего запросов: {metrics['requests'].get('total', 'Н/Д')}")
            if "performance" in metrics and "avg_processing_time" in metrics["performance"]:
                logger.info(f"  Среднее время обработки: {metrics['performance']['avg_processing_time']} мс")
        
        # Останавливаем мониторинг
        system_monitor.stop()
        
        # 4. AutoScaling
        logger.info("\n4. AutoScaling:")
        
        # Создаем экземпляр AutoScaling
        auto_scaling = AutoScaling()
        
        # Получаем информацию о компонентах
        components_info = auto_scaling.get_all_components()
        logger.info(f"Компоненты для масштабирования: {list(components_info['components'].keys())}")
        
        # Имитируем масштабирование компонента
        scaling_result = auto_scaling.scale_component(
            component_name="web_server",
            action="scale_up",
            reason="Ручное масштабирование для демонстрации"
        )
        
        logger.info(f"Результат масштабирования: {scaling_result['status']}")
        if scaling_result["status"] == "success":
            logger.info(f"  {scaling_result['message']}")
        
        # Получаем статус компонента после масштабирования
        component_status = auto_scaling.get_component_status("web_server")
        if component_status["status"] == "success":
            logger.info(f"  Текущее количество экземпляров: {component_status['component']['current_instances']}")
    
    def run_demo(self):
        """Запускает полную демонстрацию."""
        logger.info("Запуск демонстрации компонентов Фазы 5...")
        
        # Демонстрация Freemium-модели
        self.demo_freemium_model()
        
        # Демонстрация компонентов масштабирования
        self.demo_scaling_components()
        
        logger.info("\nДемонстрация завершена!")

if __name__ == "__main__":
    # Запускаем демонстрацию
    demo = DemoPhase5()
    demo.run_demo()
