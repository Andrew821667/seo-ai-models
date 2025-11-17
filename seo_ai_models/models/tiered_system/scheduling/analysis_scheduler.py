"""
Планировщик анализа с учетом приоритетов.

Модуль предоставляет функциональность для планирования анализа
с учетом приоритетов задач и доступных ресурсов.
"""

import logging
import time
import threading
import queue
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta

from seo_ai_models.models.tiered_system.core.feature_gating import TierPlan
from seo_ai_models.models.tiered_system.scheduling.credit_manager import CreditManager, CreditType


class TaskPriority(Enum):
    """Приоритеты задач."""

    HIGH = "high"  # Высокий приоритет
    MEDIUM = "medium"  # Средний приоритет
    LOW = "low"  # Низкий приоритет


class TaskStatus(Enum):
    """Статусы задач."""

    PENDING = "pending"  # Ожидание выполнения
    RUNNING = "running"  # Выполняется
    COMPLETED = "completed"  # Выполнена
    FAILED = "failed"  # Ошибка выполнения
    CANCELED = "canceled"  # Отменена


class AnalysisTask:
    """
    Задача анализа.

    Класс представляет задачу для анализа с приоритетом, параметрами и колбэками.
    """

    def __init__(
        self,
        task_id: str,
        operation: str,
        params: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        due_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализирует задачу анализа.

        Args:
            task_id: Идентификатор задачи
            operation: Название операции
            params: Параметры операции
            priority: Приоритет задачи
            callback: Функция обратного вызова при успешном выполнении
            error_callback: Функция обратного вызова при ошибке
            due_time: Время, к которому задача должна быть выполнена
            user_id: ID пользователя
            metadata: Дополнительные данные о задаче
        """
        self.task_id = task_id
        self.operation = operation
        self.params = params
        self.priority = priority
        self.callback = callback
        self.error_callback = error_callback
        self.due_time = due_time
        self.user_id = user_id
        self.metadata = metadata or {}

        # Статус задачи
        self.status = TaskStatus.PENDING

        # Время создания и выполнения
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None

        # Результат выполнения
        self.result = None
        self.error = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует задачу в словарь.

        Returns:
            Словарь с данными задачи
        """
        return {
            "task_id": self.task_id,
            "operation": self.operation,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "due_time": self.due_time.isoformat() if self.due_time else None,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    def __lt__(self, other):
        """
        Сравнивает задачи по приоритету и времени создания.

        Args:
            other: Другая задача

        Returns:
            True, если эта задача имеет более высокий приоритет
        """
        # Приоритет по важности (HIGH < MEDIUM < LOW)
        priority_order = {
            TaskPriority.HIGH: 0,
            TaskPriority.MEDIUM: 1,
            TaskPriority.LOW: 2,
        }

        # Проверяем срочность (если due_time задано)
        current_time = datetime.now()

        # Если у одной задачи есть due_time, а у другой нет,
        # задача с due_time имеет более высокий приоритет
        if self.due_time and not other.due_time:
            return True
        elif not self.due_time and other.due_time:
            return False

        # Если у обеих задач есть due_time, сравниваем по близости к дедлайну
        if self.due_time and other.due_time:
            self_time_left = (self.due_time - current_time).total_seconds()
            other_time_left = (other.due_time - current_time).total_seconds()

            # Если разница в дедлайнах существенная (более 10 минут),
            # задача с более близким дедлайном имеет более высокий приоритет
            if abs(self_time_left - other_time_left) > 600:
                return self_time_left < other_time_left

        # Если дедлайны близки или не заданы, сравниваем по приоритету
        if self.priority != other.priority:
            return priority_order[self.priority] < priority_order[other.priority]

        # Если приоритеты одинаковые, сравниваем по времени создания (FIFO)
        return self.created_at < other.created_at


class AnalysisScheduler:
    """
    Планировщик анализа с учетом приоритетов.

    Класс отвечает за планирование и выполнение задач анализа
    с учетом приоритетов и доступных ресурсов.
    """

    def __init__(
        self,
        tier: TierPlan,
        credit_manager: Optional[CreditManager] = None,
        max_workers: Optional[int] = None,
        executor: Optional[Any] = None,
        **kwargs,
    ):
        """
        Инициализирует планировщик анализа.

        Args:
            tier: План использования
            credit_manager: Менеджер кредитов
            max_workers: Максимальное количество одновременных задач
            executor: Исполнитель задач
            **kwargs: Дополнительные параметры
        """
        self.logger = logging.getLogger(__name__)
        self.tier = tier

        # Устанавливаем максимальное количество одновременных задач
        # в зависимости от плана, если не указано явно
        if max_workers is None:
            tier_workers = {
                TierPlan.MICRO: 1,
                TierPlan.BASIC: 2,
                TierPlan.PROFESSIONAL: 4,
                TierPlan.ENTERPRISE: 8,
            }
            max_workers = tier_workers.get(tier, 2)

        self.max_workers = max_workers

        # Устанавливаем менеджер кредитов
        self.credit_manager = credit_manager

        # Устанавливаем исполнитель задач
        self.executor = executor

        # Очередь задач с приоритетами
        self.task_queue = queue.PriorityQueue()

        # Словарь активных задач
        self.active_tasks = {}

        # Словарь выполненных задач
        self.completed_tasks = {}

        # Максимальное количество хранимых выполненных задач
        self.max_completed_tasks = kwargs.get("max_completed_tasks", 100)

        # Флаг работы планировщика
        self.is_running = False

        # Поток обработки очереди
        self.worker_thread = None

        # Справочник операций и их обработчиков
        self.operation_handlers = {}

        # Время хранения выполненных задач (30 дней)
        self.completed_tasks_retention_period = kwargs.get("completed_tasks_retention_period", 30)

        self.logger.info(
            f"AnalysisScheduler инициализирован с планом {tier.value} и {max_workers} рабочими потоками"
        )

    def register_operation_handler(
        self, operation: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """
        Регистрирует обработчик операции.

        Args:
            operation: Название операции
            handler: Функция-обработчик операции
        """
        self.operation_handlers[operation] = handler
        self.logger.info(f"Зарегистрирован обработчик для операции '{operation}'")

    def submit_task(self, task: AnalysisTask) -> bool:
        """
        Добавляет задачу в очередь.

        Args:
            task: Задача для добавления

        Returns:
            True, если задача успешно добавлена
        """
        # Проверяем, зарегистрирован ли обработчик для операции
        if task.operation not in self.operation_handlers:
            task.status = TaskStatus.FAILED
            task.error = ValueError(f"Неизвестная операция: {task.operation}")

            if task.error_callback:
                task.error_callback(task.error)

            return False

        # Проверяем доступность кредитов, если есть менеджер кредитов
        if self.credit_manager:
            if not self.credit_manager.check_credits_available(task.operation):
                task.status = TaskStatus.FAILED
                task.error = ValueError(f"Недостаточно кредитов для операции: {task.operation}")

                if task.error_callback:
                    task.error_callback(task.error)

                return False

        # Добавляем задачу в очередь
        self.active_tasks[task.task_id] = task
        self.task_queue.put(task)

        # Запускаем обработку очереди, если она не запущена
        if not self.is_running:
            self._start_processing()

        return True

    def submit_task_sync(
        self, task: AnalysisTask, timeout: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Добавляет задачу в очередь и ожидает ее выполнения.

        Args:
            task: Задача для добавления
            timeout: Таймаут ожидания в секундах

        Returns:
            Кортеж (успех, результат)
        """
        # Флаг завершения задачи
        task_completed = threading.Event()
        result_container = {"success": False, "result": None, "error": None}

        # Колбэки для задачи
        def success_callback(result):
            result_container["success"] = True
            result_container["result"] = result
            task_completed.set()

        def error_callback(error):
            result_container["success"] = False
            result_container["error"] = error
            task_completed.set()

        # Устанавливаем колбэки
        task.callback = success_callback
        task.error_callback = error_callback

        # Добавляем задачу в очередь
        success = self.submit_task(task)

        if not success:
            return False, {"error": task.error}

        # Ожидаем завершения задачи
        task_completed.wait(timeout)

        if not task_completed.is_set():
            return False, {"error": TimeoutError("Превышен таймаут ожидания выполнения задачи")}

        if result_container["success"]:
            return True, result_container["result"]
        else:
            return False, {"error": result_container["error"]}

    def cancel_task(self, task_id: str) -> bool:
        """
        Отменяет задачу.

        Args:
            task_id: Идентификатор задачи

        Returns:
            True, если задача успешно отменена
        """
        # Проверяем, есть ли задача в активных
        if task_id not in self.active_tasks:
            return False

        task = self.active_tasks[task_id]

        # Если задача уже выполняется или выполнена, нельзя ее отменить
        if task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]:
            return False

        # Отмечаем задачу как отмененную
        task.status = TaskStatus.CANCELED
        task.completed_at = datetime.now()

        # Перемещаем задачу в выполненные
        self.completed_tasks[task_id] = task
        del self.active_tasks[task_id]

        return True

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает статус задачи.

        Args:
            task_id: Идентификатор задачи

        Returns:
            Статус задачи или None, если задача не найдена
        """
        # Проверяем в активных задачах
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()

        # Проверяем в выполненных задачах
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()

        return None

    def get_tasks(
        self,
        status: Optional[TaskStatus] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список задач.

        Args:
            status: Статус задач для фильтрации
            user_id: ID пользователя для фильтрации
            operation: Название операции для фильтрации
            limit: Максимальное количество задач

        Returns:
            Список задач
        """
        tasks = []

        # Собираем задачи из активных и выполненных
        all_tasks = list(self.active_tasks.values()) + list(self.completed_tasks.values())

        for task in all_tasks:
            # Фильтрация по статусу
            if status and task.status != status:
                continue

            # Фильтрация по пользователю
            if user_id and task.user_id != user_id:
                continue

            # Фильтрация по операции
            if operation and task.operation != operation:
                continue

            tasks.append(task.to_dict())

            # Ограничение количества задач
            if limit and len(tasks) >= limit:
                break

        return tasks

    def stop(self) -> None:
        """Останавливает планировщик."""
        self.is_running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

        self.logger.info("AnalysisScheduler остановлен")

    def _start_processing(self) -> None:
        """Запускает обработку очереди задач."""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        self.logger.info("AnalysisScheduler запущен")

    def _process_queue(self) -> None:
        """Обрабатывает очередь задач."""
        while self.is_running:
            try:
                # Получаем активные задачи (в статусе RUNNING)
                active_count = sum(
                    1 for task in self.active_tasks.values() if task.status == TaskStatus.RUNNING
                )

                # Если достигнут лимит одновременных задач, ждем
                if active_count >= self.max_workers:
                    time.sleep(0.1)
                    continue

                # Получаем задачу из очереди (неблокирующий вызов)
                try:
                    task = self.task_queue.get(block=False)
                except queue.Empty:
                    time.sleep(0.1)
                    continue

                # Проверяем, не отменена ли задача
                if task.status == TaskStatus.CANCELED:
                    self.task_queue.task_done()
                    continue

                # Запускаем задачу в отдельном потоке
                thread = threading.Thread(target=self._execute_task, args=(task,))
                thread.daemon = True
                thread.start()

            except Exception as e:
                self.logger.error(f"Ошибка при обработке очереди задач: {e}")
                time.sleep(1.0)

    def _execute_task(self, task: AnalysisTask) -> None:
        """
        Выполняет задачу.

        Args:
            task: Задача для выполнения
        """
        # Отмечаем задачу как выполняемую
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        try:
            # Используем кредиты, если есть менеджер кредитов
            if self.credit_manager:
                self.credit_manager.use_credits(
                    operation=task.operation,
                    metadata={
                        "task_id": task.task_id,
                        "user_id": task.user_id,
                    },
                )

            # Получаем обработчик операции
            handler = self.operation_handlers.get(task.operation)

            if not handler:
                raise ValueError(f"Неизвестная операция: {task.operation}")

            # Выполняем задачу
            result = handler(task.params)

            # Отмечаем задачу как выполненную
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            # Вызываем колбэк, если задан
            if task.callback:
                task.callback(result)

        except Exception as e:
            # Отмечаем задачу как ошибочную
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = e

            # Вызываем колбэк ошибки, если задан
            if task.error_callback:
                task.error_callback(e)

            self.logger.error(f"Ошибка при выполнении задачи {task.task_id}: {e}")

        finally:
            # Перемещаем задачу в выполненные
            self.completed_tasks[task.task_id] = task
            del self.active_tasks[task.task_id]

            # Отмечаем задачу как выполненную в очереди
            self.task_queue.task_done()

            # Очищаем старые выполненные задачи
            self._cleanup_completed_tasks()

    def _cleanup_completed_tasks(self) -> None:
        """Очищает старые выполненные задачи."""
        if len(self.completed_tasks) <= self.max_completed_tasks:
            return

        # Удаляем старые задачи по времени выполнения
        retention_date = datetime.now() - timedelta(days=self.completed_tasks_retention_period)
        tasks_to_remove = []

        for task_id, task in self.completed_tasks.items():
            if task.completed_at and task.completed_at < retention_date:
                tasks_to_remove.append(task_id)

        # Удаляем задачи
        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]

        # Если все еще превышен лимит, удаляем самые старые
        if len(self.completed_tasks) > self.max_completed_tasks:
            sorted_tasks = sorted(
                self.completed_tasks.items(), key=lambda x: x[1].completed_at or datetime.now()
            )

            tasks_to_remove = sorted_tasks[: len(self.completed_tasks) - self.max_completed_tasks]

            for task_id, _ in tasks_to_remove:
                del self.completed_tasks[task_id]

        removed_count = len(tasks_to_remove)
        if removed_count > 0:
            self.logger.info(f"Удалено {removed_count} устаревших задач")

    def update_tier(self, new_tier: TierPlan) -> None:
        """
        Обновляет план использования.

        Args:
            new_tier: Новый план
        """
        old_tier = self.tier
        self.tier = new_tier

        # Обновляем максимальное количество одновременных задач
        tier_workers = {
            TierPlan.MICRO: 1,
            TierPlan.BASIC: 2,
            TierPlan.PROFESSIONAL: 4,
            TierPlan.ENTERPRISE: 8,
        }
        self.max_workers = tier_workers.get(new_tier, 2)

        # Обновляем менеджер кредитов, если он есть
        if self.credit_manager:
            self.credit_manager.update_tier(new_tier)

        self.logger.info(
            f"Обновлен план использования: {old_tier.value} -> {new_tier.value}, "
            f"максимум рабочих потоков: {self.max_workers}"
        )
