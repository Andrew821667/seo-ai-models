# -*- coding: utf-8 -*-
"""
DistributedProcessing - Распределенная обработка для масштабирования системы.
Обеспечивает масштабирование через распределение задач между несколькими процессами или узлами.
"""

import logging
import time
import threading
import queue
import multiprocessing
import uuid
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import socket

logger = logging.getLogger(__name__)


class TaskStatus:
    """Статусы задач в системе распределенной обработки."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority:
    """Приоритеты задач в системе распределенной обработки."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Task:
    """Представляет задачу в системе распределенной обработки."""

    def __init__(
        self,
        task_id: Optional[str] = None,
        function_name: str = "",
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: int = TaskPriority.NORMAL,
        timeout: Optional[int] = None,
    ):
        """
        Инициализирует задачу.

        Args:
            task_id: Идентификатор задачи (если None, генерируется автоматически)
            function_name: Имя функции для выполнения
            args: Позиционные аргументы функции
            kwargs: Именованные аргументы функции
            priority: Приоритет задачи
            timeout: Тайм-аут выполнения задачи в секундах
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.function_name = function_name
        self.args = args or []
        self.kwargs = kwargs or {}
        self.priority = priority
        self.timeout = timeout

        self.status = TaskStatus.PENDING
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.worker_id = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует задачу в словарь.

        Returns:
            Словарь с данными задачи
        """
        return {
            "task_id": self.task_id,
            "function_name": self.function_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "priority": self.priority,
            "timeout": self.timeout,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "worker_id": self.worker_id,
        }

    # Добавляем метод для сравнения задач
    def __lt__(self, other):
        """
        Определяет поведение оператора < для сравнения задач.
        Используется для приоритетной очереди, где задачи с
        более высоким приоритетом (CRITICAL > HIGH > NORMAL > LOW)
        извлекаются первыми.

        Args:
            other: Другая задача для сравнения

        Returns:
            True, если self < other по приоритету
        """
        if not isinstance(other, Task):
            return NotImplemented
        return self.priority < other.priority

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """
        Создает задачу из словаря.

        Args:
            data: Словарь с данными задачи

        Returns:
            Задача
        """
        task = cls(
            task_id=data.get("task_id"),
            function_name=data.get("function_name", ""),
            args=data.get("args", []),
            kwargs=data.get("kwargs", {}),
            priority=data.get("priority", TaskPriority.NORMAL),
            timeout=data.get("timeout"),
        )

        task.status = data.get("status", TaskStatus.PENDING)
        task.created_at = data.get("created_at", datetime.now().isoformat())
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.result = data.get("result")
        task.error = data.get("error")
        task.worker_id = data.get("worker_id")

        return task


class Worker:
    """Представляет рабочий процесс в системе распределенной обработки."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        host: str = "localhost",
        process_count: int = 1,
        task_queue: Optional[queue.PriorityQueue] = None,
        result_queue: Optional[queue.Queue] = None,
        functions: Optional[Dict[str, Callable]] = None,
    ):
        """
        Инициализирует рабочий процесс.

        Args:
            worker_id: Идентификатор рабочего процесса (если None, генерируется автоматически)
            host: Хост рабочего процесса
            process_count: Количество процессов для обработки задач
            task_queue: Очередь задач
            result_queue: Очередь результатов
            functions: Словарь доступных функций
        """
        self.worker_id = worker_id or f"{host}-{str(uuid.uuid4())[:8]}"
        self.host = host
        self.process_count = process_count
        self.task_queue = task_queue or queue.PriorityQueue()
        self.result_queue = result_queue or queue.Queue()
        self.functions = functions or {}

        self.processes = []
        self.running = False
        self.total_tasks_processed = 0
        self.failed_tasks = 0
        self.start_time = None
        self.active_tasks = {}  # Задачи, которые сейчас обрабатываются

    def start(self):
        """Запускает рабочий процесс."""
        if self.running:
            return

        self.running = True
        self.start_time = datetime.now().isoformat()

        # Запускаем процессы
        for i in range(self.process_count):
            process_id = f"{self.worker_id}-{i}"
            process = threading.Thread(target=self._process_worker, args=(process_id,))
            process.daemon = True
            process.start()
            self.processes.append(process)

        logger.info(f"Worker {self.worker_id} started with {self.process_count} processes")

    def stop(self):
        """Останавливает рабочий процесс."""
        if not self.running:
            return

        self.running = False

        # Ждем завершения всех процессов
        for process in self.processes:
            process.join(timeout=5)

        self.processes = []
        logger.info(f"Worker {self.worker_id} stopped")

    def _process_worker(self, process_id: str):
        """
        Рабочий метод для обработки задач.

        Args:
            process_id: Идентификатор процесса
        """
        logger.info(f"Process {process_id} started")

        while self.running:
            try:
                # Пытаемся получить задачу из очереди с тайм-аутом
                try:
                    # Задачи в очереди хранятся в виде кортежей (приоритет, задача)
                    priority, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Проверяем, что задача не была отменена
                if task.status == TaskStatus.CANCELLED:
                    self.task_queue.task_done()
                    continue

                # Обновляем статус задачи
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now().isoformat()
                task.worker_id = process_id

                # Добавляем задачу в список активных задач
                self.active_tasks[task.task_id] = task

                # Получаем функцию для выполнения
                function = self.functions.get(task.function_name)

                if function is None:
                    task.status = TaskStatus.FAILED
                    task.error = f"Function {task.function_name} not found"
                    task.completed_at = datetime.now().isoformat()
                    self.failed_tasks += 1
                else:
                    # Выполняем функцию
                    try:
                        # Если указан тайм-аут, используем его
                        if task.timeout is not None:
                            timeout_thread = threading.Thread(
                                target=self._function_with_timeout, args=(task, function)
                            )
                            timeout_thread.daemon = True
                            timeout_thread.start()
                            timeout_thread.join(timeout=task.timeout)

                            if timeout_thread.is_alive():
                                task.status = TaskStatus.FAILED
                                task.error = (
                                    f"Task execution timed out after {task.timeout} seconds"
                                )
                                task.completed_at = datetime.now().isoformat()
                                self.failed_tasks += 1
                            # Иначе задача уже должна быть завершена
                        else:
                            # Выполняем функцию без тайм-аута
                            result = function(*task.args, **task.kwargs)
                            task.result = result
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.now().isoformat()
                            self.total_tasks_processed += 1
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.completed_at = datetime.now().isoformat()
                        self.failed_tasks += 1
                        logger.error(f"Error executing task {task.task_id}: {e}")

                # Удаляем задачу из списка активных задач
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]

                # Отправляем результат в очередь результатов
                self.result_queue.put(task)

                # Сообщаем очереди задач, что задача обработана
                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"Error in process worker {process_id}: {e}")

        logger.info(f"Process {process_id} stopped")

    def _function_with_timeout(self, task: Task, function: Callable):
        """
        Выполняет функцию с тайм-аутом.

        Args:
            task: Задача
            function: Функция для выполнения
        """
        try:
            result = function(*task.args, **task.kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            self.total_tasks_processed += 1
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            self.failed_tasks += 1
            logger.error(f"Error executing task {task.task_id}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает статус рабочего процесса.

        Returns:
            Статус рабочего процесса
        """
        uptime = None

        if self.start_time:
            start_time_obj = datetime.fromisoformat(self.start_time)
            uptime_seconds = (datetime.now() - start_time_obj).total_seconds()

            # Форматируем uptime в формате "дни:часы:минуты:секунды"
            days, remainder = divmod(uptime_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)

            uptime = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "process_count": self.process_count,
            "running": self.running,
            "start_time": self.start_time,
            "uptime": uptime,
            "total_tasks_processed": self.total_tasks_processed,
            "failed_tasks": self.failed_tasks,
            "active_tasks": len(self.active_tasks),
            "pending_tasks": self.task_queue.qsize(),
        }

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Возвращает список активных задач.

        Returns:
            Список активных задач
        """
        return [task.to_dict() for task in self.active_tasks.values()]

    def register_function(self, name: str, function: Callable):
        """
        Регистрирует функцию.

        Args:
            name: Имя функции
            function: Функция
        """
        self.functions[name] = function
        logger.info(f"Function {name} registered in worker {self.worker_id}")

    def register_functions(self, functions: Dict[str, Callable]):
        """
        Регистрирует несколько функций.

        Args:
            functions: Словарь функций
        """
        self.functions.update(functions)
        logger.info(f"{len(functions)} functions registered in worker {self.worker_id}")


class DistributedProcessing:
    """
    Обеспечивает распределенную обработку для масштабирования системы.

    Управляет распределением задач между несколькими процессами или узлами
    для повышения производительности и масштабируемости системы.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        worker_count: int = 1,
        process_count_per_worker: int = 2,
    ):
        """
        Инициализирует DistributedProcessing.

        Args:
            config: Конфигурация
            worker_count: Количество рабочих процессов
            process_count_per_worker: Количество процессов на каждый рабочий процесс
        """
        self.config = config or {}
        self.worker_count = worker_count
        self.process_count_per_worker = process_count_per_worker

        # Очереди для задач и результатов
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()

        # Рабочие процессы
        self.workers = {}

        # Функции, доступные для выполнения
        self.registered_functions = {}

        # Задачи
        self.tasks = {}

        # Блокировка для доступа к задачам
        self.tasks_lock = threading.Lock()

        # Поток для обработки результатов
        self.result_thread = None

        # Флаг для управления работой
        self.running = False

        # Обработчики событий
        self.event_handlers = {
            "task_completed": [],
            "task_failed": [],
            "worker_added": [],
            "worker_removed": [],
        }

        # Инициализируем рабочие процессы
        self._init_workers()

    def _init_workers(self):
        """Инициализирует рабочие процессы."""
        host = socket.gethostname()

        for i in range(self.worker_count):
            worker_id = f"{host}-worker-{i}"

            worker = Worker(
                worker_id=worker_id,
                host=host,
                process_count=self.process_count_per_worker,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                functions=self.registered_functions,
            )

            self.workers[worker_id] = worker

    def start(self):
        """Запускает систему распределенной обработки."""
        if self.running:
            return

        self.running = True

        # Запускаем рабочие процессы
        for worker in self.workers.values():
            worker.start()

        # Запускаем поток для обработки результатов
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()

        logger.info(f"Distributed processing started with {len(self.workers)} workers")

    def stop(self):
        """Останавливает систему распределенной обработки."""
        if not self.running:
            return

        self.running = False

        # Останавливаем рабочие процессы
        for worker in self.workers.values():
            worker.stop()

        # Ждем завершения потока обработки результатов
        if self.result_thread:
            self.result_thread.join(timeout=5)
            self.result_thread = None

        logger.info("Distributed processing stopped")

    def _process_results(self):
        """Обрабатывает результаты выполнения задач."""
        while self.running:
            try:
                # Пытаемся получить результат из очереди с тайм-аутом
                try:
                    task = self.result_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Обновляем задачу в списке задач
                with self.tasks_lock:
                    if task.task_id in self.tasks:
                        self.tasks[task.task_id] = task

                # Вызываем обработчики событий
                if task.status == TaskStatus.COMPLETED:
                    self._trigger_event("task_completed", task)
                elif task.status == TaskStatus.FAILED:
                    self._trigger_event("task_failed", task)

                # Сообщаем очереди результатов, что результат обработан
                self.result_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing results: {e}")

    def register_function(self, name: str, function: Callable):
        """
        Регистрирует функцию.

        Args:
            name: Имя функции
            function: Функция
        """
        self.registered_functions[name] = function

        # Регистрируем функцию в рабочих процессах
        for worker in self.workers.values():
            worker.register_function(name, function)

        logger.info(f"Function {name} registered")

    def register_functions(self, functions: Dict[str, Callable]):
        """
        Регистрирует несколько функций.

        Args:
            functions: Словарь функций
        """
        self.registered_functions.update(functions)

        # Регистрируем функции в рабочих процессах
        for worker in self.workers.values():
            worker.register_functions(functions)

        logger.info(f"{len(functions)} functions registered")

    def add_task(
        self,
        function_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        priority: int = TaskPriority.NORMAL,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Добавляет задачу в очередь.

        Args:
            function_name: Имя функции для выполнения
            args: Позиционные аргументы функции
            kwargs: Именованные аргументы функции
            task_id: Идентификатор задачи (если None, генерируется автоматически)
            priority: Приоритет задачи
            timeout: Тайм-аут выполнения задачи в секундах

        Returns:
            Информация о добавленной задаче
        """
        # Проверяем, что функция зарегистрирована
        if function_name not in self.registered_functions:
            return {"status": "error", "message": f"Function {function_name} not registered"}

        # Создаем задачу
        task = Task(
            task_id=task_id,
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout,
        )

        # Добавляем задачу в список задач
        with self.tasks_lock:
            self.tasks[task.task_id] = task

        # Добавляем задачу в очередь задач - используем приоритет задачи напрямую
        # в приоритетной очереди, вместо инвертирования
        self.task_queue.put((priority, task))

        return {"status": "success", "message": "Task added", "task_id": task.task_id}

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о задаче.

        Args:
            task_id: Идентификатор задачи

        Returns:
            Информация о задаче или None, если задача не найдена
        """
        with self.tasks_lock:
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()

        return None

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Отменяет задачу.

        Args:
            task_id: Идентификатор задачи

        Returns:
            Результат отмены
        """
        with self.tasks_lock:
            if task_id not in self.tasks:
                return {"status": "error", "message": f"Task {task_id} not found"}

            task = self.tasks[task_id]

            # Проверяем, что задача еще не выполнена
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return {"status": "error", "message": f"Task {task_id} already {task.status}"}

            # Отменяем задачу
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now().isoformat()

        return {"status": "success", "message": f"Task {task_id} cancelled"}

    def _handle_task_failure(self, task_id: str, error: Exception, worker_id: str = None) -> None:
        """
        Обрабатывает сбой выполнения задачи.

        Args:
            task_id: Идентификатор задачи
            error: Исключение, вызвавшее сбой
            worker_id: Идентификатор рабочего процесса (опционально)
        """
        try:
            with self.tasks_lock:
                if task_id not in self.tasks:
                    return

                task = self.tasks[task_id]
                current_retry = getattr(task, "retry_count", 0)
                max_retries = self.config.get("max_retries", 3)

                # Логируем ошибку
                logger = logging.getLogger(__name__)
                logger.error(f"Task {task_id} failed: {str(error)}")
                if worker_id:
                    logger.error(f"Worker {worker_id} reported task failure")

                # Увеличиваем счетчик попыток
                task.retry_count = current_retry + 1
                task.status = "failed"
                task.error = str(error)
                task.last_failure_time = time.time()

                # Если превышено количество попыток - помечаем как окончательно проваленную
                if task.retry_count >= max_retries:
                    task.status = "permanently_failed"
                    logger.error(f"Task {task_id} permanently failed after {max_retries} retries")

                    # Уведомляем о критическом сбое
                    self._trigger_event(
                        "task_permanently_failed",
                        {
                            "task_id": task_id,
                            "error": str(error),
                            "retry_count": task.retry_count,
                            "worker_id": worker_id,
                        },
                    )
                else:
                    # Планируем повторную попытку
                    retry_delay = self.config.get("retry_delay", 5) * (
                        2**current_retry
                    )  # Экспоненциальная задержка
                    task.scheduled_retry_time = time.time() + retry_delay
                    task.status = "scheduled_for_retry"

                    logger.info(
                        f"Task {task_id} scheduled for retry #{task.retry_count} in {retry_delay} seconds"
                    )

                    # Уведомляем о повторной попытке
                    self._trigger_event(
                        "task_retry_scheduled",
                        {
                            "task_id": task_id,
                            "retry_count": task.retry_count,
                            "retry_delay": retry_delay,
                            "worker_id": worker_id,
                        },
                    )

                # Обновляем статистику воркера, если указан
                if worker_id and worker_id in self.workers:
                    worker = self.workers[worker_id]
                    worker.failed_tasks_count = getattr(worker, "failed_tasks_count", 0) + 1
                    worker.last_failure_time = time.time()

                    # Если воркер часто падает - помечаем его как проблемный
                    failure_rate = worker.failed_tasks_count / max(
                        worker.completed_tasks_count + worker.failed_tasks_count, 1
                    )
                    if failure_rate > self.config.get("worker_failure_threshold", 0.3):
                        logger.warning(
                            f"Worker {worker_id} has high failure rate: {failure_rate:.2%}"
                        )
                        worker.status = "problematic"

        except Exception as e:
            # Если произошла ошибка в обработчике ошибок - логируем и продолжаем
            logger = logging.getLogger(__name__)
            logger.critical(f"Error in _handle_task_failure: {str(e)}")

    def get_tasks(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """
        Возвращает список задач.

        Args:
            status: Фильтр по статусу задачи
            limit: Максимальное количество задач
            offset: Смещение

        Returns:
            Список задач
        """
        with self.tasks_lock:
            # Фильтруем задачи по статусу
            if status:
                filtered_tasks = [task for task in self.tasks.values() if task.status == status]
            else:
                filtered_tasks = list(self.tasks.values())

            # Сортируем задачи по времени создания (от новых к старым)
            sorted_tasks = sorted(filtered_tasks, key=lambda task: task.created_at, reverse=True)

            # Применяем смещение и лимит
            paginated_tasks = sorted_tasks[offset : offset + limit]

            return {
                "total": len(filtered_tasks),
                "limit": limit,
                "offset": offset,
                "tasks": [task.to_dict() for task in paginated_tasks],
            }

    def add_worker(
        self, host: str, process_count: int = 2, worker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Добавляет новый рабочий процесс.

        Args:
            host: Хост рабочего процесса
            process_count: Количество процессов
            worker_id: Идентификатор рабочего процесса

        Returns:
            Результат добавления
        """
        # Генерируем идентификатор рабочего процесса, если он не указан
        worker_id = worker_id or f"{host}-worker-{len(self.workers)}"

        # Проверяем, что рабочий процесс с таким идентификатором еще не существует
        if worker_id in self.workers:
            return {"status": "error", "message": f"Worker {worker_id} already exists"}

        # Создаем рабочий процесс
        worker = Worker(
            worker_id=worker_id,
            host=host,
            process_count=process_count,
            task_queue=self.task_queue,
            result_queue=self.result_queue,
            functions=self.registered_functions,
        )

        # Добавляем рабочий процесс в список рабочих процессов
        self.workers[worker_id] = worker

        # Запускаем рабочий процесс, если система запущена
        if self.running:
            worker.start()

        # Вызываем обработчики события
        self._trigger_event("worker_added", worker)

        return {"status": "success", "message": f"Worker {worker_id} added", "worker_id": worker_id}

    def remove_worker(self, worker_id: str) -> Dict[str, Any]:
        """
        Удаляет рабочий процесс.

        Args:
            worker_id: Идентификатор рабочего процесса

        Returns:
            Результат удаления
        """
        # Проверяем, что рабочий процесс существует
        if worker_id not in self.workers:
            return {"status": "error", "message": f"Worker {worker_id} not found"}

        # Получаем рабочий процесс
        worker = self.workers[worker_id]

        # Останавливаем рабочий процесс
        worker.stop()

        # Удаляем рабочий процесс из списка рабочих процессов
        del self.workers[worker_id]

        # Вызываем обработчики события
        self._trigger_event("worker_removed", worker)

        return {"status": "success", "message": f"Worker {worker_id} removed"}

    def get_workers(self) -> Dict[str, Any]:
        """
        Возвращает список рабочих процессов.

        Returns:
            Список рабочих процессов
        """
        return {
            "total": len(self.workers),
            "workers": [worker.get_status() for worker in self.workers.values()],
        }

    def _cleanup_completed_tasks(self, max_completed_age: int = 3600) -> Dict[str, Any]:
        """
        Очищает завершенные задачи для освобождения памяти.

        Args:
            max_completed_age: Максимальный возраст завершенных задач в секундах (по умолчанию 1 час)

        Returns:
            Статистика очистки
        """
        try:
            cleanup_stats = {
                "cleaned_tasks": 0,
                "freed_memory_mb": 0,
                "cleanup_time": time.time(),
                "errors": [],
            }

            current_time = time.time()
            tasks_to_remove = []

            with self.tasks_lock:
                for task_id, task in self.tasks.items():
                    # Проверяем, нужно ли удалить задачу
                    should_remove = False

                    # Удаляем старые завершенные задачи
                    if task.status in ["completed", "permanently_failed"]:
                        completion_time = getattr(task, "completion_time", 0)
                        if completion_time and (current_time - completion_time) > max_completed_age:
                            should_remove = True

                    # Удаляем очень старые задачи в любом статусе (защита от утечек памяти)
                    creation_time = getattr(task, "creation_time", current_time)
                    max_task_age = self.config.get("max_task_age", 86400)  # 24 часа по умолчанию
                    if (current_time - creation_time) > max_task_age:
                        should_remove = True
                        cleanup_stats["errors"].append(f"Task {task_id} exceeded maximum age")

                    if should_remove:
                        tasks_to_remove.append(task_id)

                # Удаляем найденные задачи
                for task_id in tasks_to_remove:
                    try:
                        task = self.tasks[task_id]

                        # Оцениваем освобожденную память (приблизительно)
                        task_memory = len(str(task.to_dict())) / 1024 / 1024  # MB
                        cleanup_stats["freed_memory_mb"] += task_memory

                        # Удаляем задачу
                        del self.tasks[task_id]
                        cleanup_stats["cleaned_tasks"] += 1

                    except Exception as e:
                        cleanup_stats["errors"].append(f"Error removing task {task_id}: {str(e)}")

            # Очищаем результаты
            if hasattr(self, "results"):
                with self.results_lock if hasattr(self, "results_lock") else self.tasks_lock:
                    results_to_remove = []
                    for result_id in list(self.results.keys()):
                        # Удаляем результаты для которых нет задач
                        if result_id not in self.tasks:
                            results_to_remove.append(result_id)

                    for result_id in results_to_remove:
                        try:
                            del self.results[result_id]
                            cleanup_stats["cleaned_tasks"] += 1
                        except Exception as e:
                            cleanup_stats["errors"].append(
                                f"Error removing result {result_id}: {str(e)}"
                            )

            # Вызываем сборку мусора Python
            import gc

            collected = gc.collect()
            cleanup_stats["gc_collected"] = collected

            # Логируем результаты
            logger = logging.getLogger(__name__)
            if cleanup_stats["cleaned_tasks"] > 0:
                logger.info(
                    f"Cleanup completed: {cleanup_stats['cleaned_tasks']} tasks removed, "
                    f"{cleanup_stats['freed_memory_mb']:.2f} MB freed"
                )

            if cleanup_stats["errors"]:
                logger.warning(f"Cleanup errors: {len(cleanup_stats['errors'])}")
                for error in cleanup_stats["errors"][:5]:  # Показываем первые 5 ошибок
                    logger.warning(f"  {error}")

            # Уведомляем о завершении очистки
            self._trigger_event("cleanup_completed", cleanup_stats)

            return cleanup_stats

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Critical error in _cleanup_completed_tasks: {str(e)}")
            return {
                "cleaned_tasks": 0,
                "freed_memory_mb": 0,
                "cleanup_time": time.time(),
                "errors": [f"Critical error: {str(e)}"],
                "gc_collected": 0,
            }

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о рабочем процессе.

        Args:
            worker_id: Идентификатор рабочего процесса

        Returns:
            Информация о рабочем процессе или None, если рабочий процесс не найден
        """
        if worker_id in self.workers:
            return self.workers[worker_id].get_status()

        return None

    def get_active_tasks_by_worker(self, worker_id: str) -> Dict[str, Any]:
        """
        Возвращает список активных задач рабочего процесса.

        Args:
            worker_id: Идентификатор рабочего процесса

        Returns:
            Список активных задач
        """
        if worker_id not in self.workers:
            return {"status": "error", "message": f"Worker {worker_id} not found"}

        return {"worker_id": worker_id, "active_tasks": self.workers[worker_id].get_active_tasks()}

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику системы.

        Returns:
            Статистика системы
        """
        total_tasks = len(self.tasks)

        # Считаем количество задач в разных статусах
        pending_tasks = 0
        processing_tasks = 0
        completed_tasks = 0
        failed_tasks = 0
        cancelled_tasks = 0

        with self.tasks_lock:
            for task in self.tasks.values():
                if task.status == TaskStatus.PENDING:
                    pending_tasks += 1
                elif task.status == TaskStatus.PROCESSING:
                    processing_tasks += 1
                elif task.status == TaskStatus.COMPLETED:
                    completed_tasks += 1
                elif task.status == TaskStatus.FAILED:
                    failed_tasks += 1
                elif task.status == TaskStatus.CANCELLED:
                    cancelled_tasks += 1

        # Считаем общее количество обработанных задач
        total_processed = sum(worker.total_tasks_processed for worker in self.workers.values())

        # Считаем общее количество неудачных задач
        total_failed = sum(worker.failed_tasks for worker in self.workers.values())

        return {
            "workers": {
                "total": len(self.workers),
                "active": sum(1 for worker in self.workers.values() if worker.running),
            },
            "tasks": {
                "total": total_tasks,
                "pending": pending_tasks,
                "processing": processing_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "cancelled": cancelled_tasks,
                "total_processed": total_processed,
                "total_failed": total_failed,
            },
            "queues": {
                "task_queue_size": self.task_queue.qsize(),
                "result_queue_size": self.result_queue.qsize(),
            },
        }

    def on(self, event: str, handler: Callable):
        """
        Регистрирует обработчик события.

        Args:
            event: Название события
            handler: Обработчик события
        """
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable):
        """
        Удаляет обработчик события.

        Args:
            event: Название события
            handler: Обработчик события
        """
        if event in self.event_handlers and handler in self.event_handlers[event]:
            self.event_handlers[event].remove(handler)

    def _trigger_event(self, event: str, data: Any):
        """
        Вызывает обработчики события.

        Args:
            event: Название события
            data: Данные события
        """
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {e}")

    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        """
        Ожидает завершения всех задач.

        Args:
            timeout: Тайм-аут в секундах

        Returns:
            True, если все задачи завершены, иначе False
        """
        # Ждем завершения задач в очереди задач
        try:
            self.task_queue.join(timeout=timeout)
            return True
        except Exception:
            return False

    def clear_completed_tasks(self, clear_failed: bool = True) -> Dict[str, Any]:
        """
        Очищает список завершенных задач.

        Args:
            clear_failed: Очищать ли неудачные задачи

        Returns:
            Результат очистки
        """
        with self.tasks_lock:
            # Получаем список идентификаторов задач для удаления
            tasks_to_remove = []

            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.COMPLETED or (
                    clear_failed and task.status == TaskStatus.FAILED
                ):
                    tasks_to_remove.append(task_id)

            # Удаляем задачи
            for task_id in tasks_to_remove:
                del self.tasks[task_id]

            return {"status": "success", "message": f"Cleared {len(tasks_to_remove)} tasks"}

    def save_tasks(self, file_path: str) -> Dict[str, Any]:
        """
        Сохраняет список задач в файл.

        Args:
            file_path: Путь к файлу

        Returns:
            Результат сохранения
        """
        try:
            with self.tasks_lock:
                # Формируем список задач для сохранения
                tasks_data = [task.to_dict() for task in self.tasks.values()]

            # Сохраняем задачи в файл
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(tasks_data, f, indent=2, ensure_ascii=False)

            return {
                "status": "success",
                "message": f"Tasks saved to {file_path}",
                "count": len(tasks_data),
            }
        except Exception as e:
            logger.error(f"Error saving tasks to file {file_path}: {e}")
            return {"status": "error", "message": f"Error saving tasks: {str(e)}"}

    def load_tasks(self, file_path: str) -> Dict[str, Any]:
        """
        Загружает список задач из файла.

        Args:
            file_path: Путь к файлу

        Returns:
            Результат загрузки
        """
        try:
            # Загружаем задачи из файла
            with open(file_path, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)

            # Преобразуем данные в объекты Task
            loaded_tasks = []

            for task_data in tasks_data:
                task = Task.from_dict(task_data)
                loaded_tasks.append(task)

            # Добавляем задачи в список задач
            with self.tasks_lock:
                for task in loaded_tasks:
                    self.tasks[task.task_id] = task

            return {
                "status": "success",
                "message": f"Tasks loaded from {file_path}",
                "count": len(loaded_tasks),
            }
        except Exception as e:
            logger.error(f"Error loading tasks from file {file_path}: {e}")
            return {"status": "error", "message": f"Error loading tasks: {str(e)}"}
