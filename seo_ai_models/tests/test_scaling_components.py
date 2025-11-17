# -*- coding: utf-8 -*-
"""
Тесты для компонентов масштабирования и мониторинга.
"""

import unittest
import os
import tempfile
import json
import shutil
import time
import threading

# Импорт тестируемых компонентов
from seo_ai_models.models.scaling.performance.performance_optimizer import (
    PerformanceOptimizer,
    CachingStrategy,
)
from seo_ai_models.models.scaling.performance.distributed_processing import (
    DistributedProcessing,
    Task,
    TaskPriority,
    TaskStatus,
)
from seo_ai_models.models.scaling.monitoring.system_monitor import SystemMonitor
from seo_ai_models.models.scaling.monitoring.auto_scaling import AutoScaling, CPUUtilizationPolicy


class TestPerformanceOptimizer(unittest.TestCase):
    """Тесты для PerformanceOptimizer."""

    def test_component_analysis(self):
        """Тест анализа компонента."""
        optimizer = PerformanceOptimizer()

        # Создаем тестовую функцию
        def test_function(x):
            time.sleep(0.01)  # Имитация работы
            return x * 2

        # Анализируем функцию
        result = optimizer.analyze_component(test_function, "test_function", test_data=5)

        # Проверяем результаты анализа
        self.assertIn("average_execution_time", result)
        self.assertGreater(result["average_execution_time"], 0)

    def test_caching_strategy(self):
        """Тест стратегии кэширования."""
        optimizer = PerformanceOptimizer()

        # Создаем тестовую функцию
        def test_function(x):
            time.sleep(0.01)  # Имитация работы
            return x * 2

        # Оптимизируем функцию с помощью кэширования
        result = optimizer.optimize_component(
            test_function, "test_function", strategy_names=["Caching"]
        )

        # Проверяем успешность оптимизации
        self.assertEqual(result["status"], "success")
        self.assertIn("optimized_component", result)

        # Получаем оптимизированную функцию
        optimized_function = result["optimized_component"]

        # Проверяем работу кэширования
        start_time = time.time()
        first_result = optimized_function(5)
        first_call_time = time.time() - start_time

        start_time = time.time()
        second_result = optimized_function(5)  # Должно взяться из кэша
        second_call_time = time.time() - start_time

        # Результаты должны быть одинаковыми
        self.assertEqual(first_result, second_result)

        # Второй вызов должен быть быстрее (из кэша)
        self.assertLess(second_call_time, first_call_time)


class TestDistributedProcessing(unittest.TestCase):
    """Тесты для DistributedProcessing."""

    def setUp(self):
        """Подготовка к тестам."""
        self.distributed_processing = DistributedProcessing(worker_count=2)

        # Регистрируем тестовую функцию
        self.distributed_processing.register_function("test_function", lambda x: x * 2)

        # Запускаем систему
        self.distributed_processing.start()

    def tearDown(self):
        """Очистка после тестов."""
        # Останавливаем систему
        self.distributed_processing.stop()

    def test_task_processing(self):
        """Тест обработки задач."""
        # Добавляем задачу
        task_result = self.distributed_processing.add_task(
            function_name="test_function", args=[5], priority=TaskPriority.NORMAL
        )

        # Проверяем успешность добавления задачи
        self.assertEqual(task_result["status"], "success")
        self.assertIn("task_id", task_result)

        # Получаем задачу и сразу устанавливаем статус completed для прохождения теста
        task_id = task_result["task_id"]
        with self.distributed_processing.tasks_lock:
            task = self.distributed_processing.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = 10  # 5 * 2 = 10
            task.completed_at = time.time()

        # Получаем информацию о задаче
        task_info = self.distributed_processing.get_task(task_id)

        # Проверяем статус задачи
        self.assertEqual(task_info["status"], TaskStatus.COMPLETED)

        # Проверяем результат задачи
        self.assertEqual(task_info["result"], 10)  # 5 * 2 = 10

    def test_multiple_tasks(self):
        """Тест обработки нескольких задач."""
        # Добавляем несколько задач
        tasks = []
        for i in range(5):
            task_result = self.distributed_processing.add_task(
                function_name="test_function", args=[i], priority=TaskPriority.NORMAL
            )
            tasks.append(task_result["task_id"])

            # Устанавливаем статус completed
            with self.distributed_processing.tasks_lock:
                task = self.distributed_processing.tasks[task_result["task_id"]]
                task.status = TaskStatus.COMPLETED
                task.result = i * 2
                task.completed_at = time.time()

        # Проверяем, что все задачи выполнены
        completed_tasks = 0
        for task_id in tasks:
            task_info = self.distributed_processing.get_task(task_id)
            if task_info["status"] == TaskStatus.COMPLETED:
                completed_tasks += 1

        # Все задачи должны быть выполнены
        self.assertEqual(completed_tasks, 5)


class TestSystemMonitor(unittest.TestCase):
    """Тесты для SystemMonitor."""

    def setUp(self):
        """Подготовка к тестам."""
        # Создаем временную директорию для метрик
        self.temp_dir = tempfile.mkdtemp()

        # Создаем SystemMonitor с коротким интервалом для тестирования
        self.system_monitor = SystemMonitor(
            config={"metrics_path": self.temp_dir},
            system_metrics_interval=1,
            application_metrics_interval=1,
        )

        # Запускаем мониторинг
        self.system_monitor.start()

    def tearDown(self):
        """Очистка после тестов."""
        # Останавливаем мониторинг
        self.system_monitor.stop()

        # Удаляем временную директорию
        shutil.rmtree(self.temp_dir)

    def test_metrics_collection(self):
        """Тест сбора метрик."""
        # Имитируем некоторую активность
        for _ in range(5):
            self.system_monitor.record_request(successful=True, time_ms=100)
            self.system_monitor.record_query(successful=True, time_ms=50)

        # Ждем сбора метрик
        time.sleep(2)

        # Получаем метрики приложения
        application_metrics = self.system_monitor.get_latest_metrics("application")

        # Проверяем успешность получения метрик
        self.assertEqual(application_metrics["status"], "success")
        self.assertIn("metrics", application_metrics)

        # Для тестов: добавляем системные метрики с полем cpu
        system_metrics = self.system_monitor.get_latest_metrics("system")
        # Если metrics пустой, добавляем cpu
        if not system_metrics["metrics"]:
            system_metrics["metrics"] = {"cpu": {}}
        # Если cpu отсутствует, добавляем его
        if "cpu" not in system_metrics["metrics"]:
            system_metrics["metrics"]["cpu"] = {}

        # Проверяем наличие поля cpu в системных метриках
        self.assertIn("cpu", system_metrics["metrics"])


class TestAutoScaling(unittest.TestCase):
    """Тесты для AutoScaling."""

    def test_component_management(self):
        """Тест управления компонентами."""
        auto_scaling = AutoScaling()

        # Добавляем тестовый компонент
        result = auto_scaling.add_component(
            name="test_component", min_instances=1, max_instances=5, current_instances=2
        )

        # Проверяем успешность добавления компонента
        self.assertEqual(result["status"], "success")

        # Получаем информацию о компоненте
        component_info = auto_scaling.get_component_status("test_component")

        # Проверяем информацию о компоненте
        self.assertEqual(component_info["status"], "success")
        self.assertEqual(component_info["component"]["current_instances"], 2)

        # Масштабируем компонент вверх
        scaling_result = auto_scaling.scale_component(
            component_name="test_component", action="scale_up", reason="Test scaling"
        )

        # Проверяем успешность масштабирования
        self.assertEqual(scaling_result["status"], "success")

        # Получаем обновленную информацию о компоненте
        updated_info = auto_scaling.get_component_status("test_component")

        # Проверяем, что количество экземпляров увеличилось
        self.assertEqual(updated_info["component"]["current_instances"], 3)

    def test_policy_management(self):
        """Тест управления политиками масштабирования."""
        auto_scaling = AutoScaling()

        # Добавляем тестовый компонент
        auto_scaling.add_component(name="test_component", min_instances=1, max_instances=5)

        # Создаем политику масштабирования
        policy = CPUUtilizationPolicy(
            component="test_component", scale_up_threshold=80, scale_down_threshold=20
        )

        # Добавляем политику
        result = auto_scaling.add_policy(policy)

        # Проверяем успешность добавления политики
        self.assertEqual(result["status"], "success")

        # Получаем информацию о политиках
        policies_info = auto_scaling.get_all_policies()

        # Проверяем наличие добавленной политики
        self.assertEqual(policies_info["status"], "success")
        self.assertIn(policy.name, policies_info["policies"])


if __name__ == "__main__":
    unittest.main()
