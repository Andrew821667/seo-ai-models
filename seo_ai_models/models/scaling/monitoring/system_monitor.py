# -*- coding: utf-8 -*-
"""
SystemMonitor - Мониторинг производительности и использования ресурсов системы.
Отслеживает и анализирует ключевые метрики производительности системы.
"""

import logging
import time
import threading
import os
import json
import datetime
import socket
import queue
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import psutil

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class MetricCollector:
    """Базовый класс для сбора метрик."""

    def __init__(self, name: str, interval: int = 60):
        """
        Инициализирует сборщик метрик.

        Args:
            name: Имя сборщика метрик
            interval: Интервал сбора метрик в секундах
        """
        self.name = name
        self.interval = interval
        self.metrics = []
        self.timestamps = []
        self.max_data_points = 1000  # Максимальное количество сохраняемых точек данных

    def collect(self) -> Dict[str, Any]:
        """
        Собирает метрики.

        Returns:
            Собранные метрики
        """
        raise NotImplementedError("Метод должен быть переопределен в подклассе")

    def add_metrics(self, metrics: Dict[str, Any]):
        """
        Добавляет метрики в историю.

        Args:
            metrics: Метрики для добавления
        """
        self.metrics.append(metrics)
        self.timestamps.append(time.time())

        # Ограничиваем количество сохраняемых метрик
        if len(self.metrics) > self.max_data_points:
            self.metrics.pop(0)
            self.timestamps.pop(0)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Возвращает последние собранные метрики.

        Returns:
            Последние метрики или пустой словарь, если метрики еще не собраны
        """
        if self.metrics:
            return self.metrics[-1]
        return {}

    def get_metrics_history(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Возвращает историю метрик за указанный период.

        Args:
            start_time: Начальное время (если None, с начала сбора)
            end_time: Конечное время (если None, до текущего момента)

        Returns:
            Кортеж (временные метки, метрики)
        """
        if not self.metrics:
            return [], []

        # Если начальное время не указано, берем время первой метрики
        if start_time is None:
            start_time = self.timestamps[0]

        # Если конечное время не указано, берем текущее время
        if end_time is None:
            end_time = time.time()

        # Фильтруем метрики по времени
        filtered_indices = [
            i for i, ts in enumerate(self.timestamps) if start_time <= ts <= end_time
        ]

        filtered_timestamps = [self.timestamps[i] for i in filtered_indices]
        filtered_metrics = [self.metrics[i] for i in filtered_indices]

        return filtered_timestamps, filtered_metrics

    def calculate_statistics(
        self, metric_name: str, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Рассчитывает статистику для указанной метрики.

        Args:
            metric_name: Имя метрики
            start_time: Начальное время
            end_time: Конечное время

        Returns:
            Статистика (мин, макс, среднее, медиана, стандартное отклонение)
        """
        timestamps, metrics_list = self.get_metrics_history(start_time, end_time)

        if not metrics_list:
            return {"min": None, "max": None, "avg": None, "median": None, "std": None}

        # Извлекаем значения метрики
        values = []

        for metrics in metrics_list:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    values.append(value)

        if not values:
            return {"min": None, "max": None, "avg": None, "median": None, "std": None}

        # Рассчитываем статистику
        result = {"min": min(values), "max": max(values), "avg": sum(values) / len(values)}

        # Если доступен numpy, используем его для расчета медианы и стандартного отклонения
        if np is not None:
            result["median"] = float(np.median(values))
            result["std"] = float(np.std(values))
        else:
            # Простой расчет медианы
            sorted_values = sorted(values)
            mid = len(sorted_values) // 2
            if len(sorted_values) % 2 == 0:
                result["median"] = (sorted_values[mid - 1] + sorted_values[mid]) / 2
            else:
                result["median"] = sorted_values[mid]

            # Простой расчет стандартного отклонения
            mean = result["avg"]
            sum_squared_diff = sum((x - mean) ** 2 for x in values)
            result["std"] = (sum_squared_diff / len(values)) ** 0.5

        return result


class SystemMetricsCollector(MetricCollector):
    """Сборщик системных метрик."""

    def __init__(self, interval: int = 60):
        """
        Инициализирует сборщик системных метрик.

        Args:
            interval: Интервал сбора метрик в секундах
        """
        super().__init__(name="system", interval=interval)

    def collect(self) -> Dict[str, Any]:
        """
        Собирает системные метрики.

        Returns:
            Собранные метрики
        """
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_times = psutil.cpu_times_percent(interval=1)

        # Память
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Диск
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                }
            except PermissionError:
                # Некоторые точки монтирования могут быть недоступны
                pass

        # Сеть
        net_io = psutil.net_io_counters()

        # Процессы
        process_count = len(psutil.pids())

        metrics = {
            "timestamp": time.time(),
            "hostname": socket.gethostname(),
            "cpu": {
                "percent": cpu_percent,
                "user": cpu_times.user,
                "system": cpu_times.system,
                "idle": cpu_times.idle,
                "count": psutil.cpu_count(),
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            },
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent,
            },
            "disk": disk_usage,
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            },
            "processes": {"count": process_count},
        }

        # Добавляем метрики в историю
        self.add_metrics(metrics)

        return metrics


class ApplicationMetricsCollector(MetricCollector):
    """Сборщик метрик приложения."""

    def __init__(self, interval: int = 60):
        """
        Инициализирует сборщик метрик приложения.

        Args:
            interval: Интервал сбора метрик в секундах
        """
        super().__init__(name="application", interval=interval)

        # Счетчики производительности приложения
        self.counters = {
            "requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "processing_time_sum": 0,
            "processing_time_count": 0,
            "errors": 0,
        }

        # Блокировка для доступа к счетчикам
        self.counters_lock = threading.Lock()

    def collect(self) -> Dict[str, Any]:
        """
        Собирает метрики приложения.

        Returns:
            Собранные метрики
        """
        # Создаем копию счетчиков
        with self.counters_lock:
            counters_copy = self.counters.copy()

            # Рассчитываем среднее время обработки
            avg_processing_time = None
            if counters_copy["processing_time_count"] > 0:
                avg_processing_time = (
                    counters_copy["processing_time_sum"] / counters_copy["processing_time_count"]
                )

            # Рассчитываем процент успешных запросов
            success_rate = None
            if counters_copy["requests"] > 0:
                success_rate = (
                    counters_copy["successful_requests"] / counters_copy["requests"]
                ) * 100

            metrics = {
                "timestamp": time.time(),
                "requests": {
                    "total": counters_copy["requests"],
                    "successful": counters_copy["successful_requests"],
                    "failed": counters_copy["failed_requests"],
                    "success_rate": success_rate,
                },
                "performance": {"avg_processing_time": avg_processing_time},
                "errors": counters_copy["errors"],
            }

        # Добавляем метрики в историю
        self.add_metrics(metrics)

        return metrics

    def increment_counter(self, counter_name: str, value: int = 1):
        """
        Увеличивает значение счетчика.

        Args:
            counter_name: Имя счетчика
            value: Значение для увеличения
        """
        with self.counters_lock:
            if counter_name in self.counters:
                self.counters[counter_name] += value

    def record_processing_time(self, time_ms: float):
        """
        Записывает время обработки запроса.

        Args:
            time_ms: Время обработки в миллисекундах
        """
        with self.counters_lock:
            self.counters["processing_time_sum"] += time_ms
            self.counters["processing_time_count"] += 1

    def record_request(self, successful: bool = True):
        """
        Записывает запрос.

        Args:
            successful: Успешен ли запрос
        """
        with self.counters_lock:
            self.counters["requests"] += 1

            if successful:
                self.counters["successful_requests"] += 1
            else:
                self.counters["failed_requests"] += 1

    def record_error(self):
        """Записывает ошибку."""
        with self.counters_lock:
            self.counters["errors"] += 1


class DatabaseMetricsCollector(MetricCollector):
    """Сборщик метрик базы данных."""

    def __init__(self, interval: int = 60):
        """
        Инициализирует сборщик метрик базы данных.

        Args:
            interval: Интервал сбора метрик в секундах
        """
        super().__init__(name="database", interval=interval)

        # Счетчики производительности базы данных
        self.counters = {
            "queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "query_time_sum": 0,
            "query_time_count": 0,
        }

        # Блокировка для доступа к счетчикам
        self.counters_lock = threading.Lock()

    def collect(self) -> Dict[str, Any]:
        """
        Собирает метрики базы данных.

        Returns:
            Собранные метрики
        """
        # Создаем копию счетчиков
        with self.counters_lock:
            counters_copy = self.counters.copy()

            # Рассчитываем среднее время выполнения запроса
            avg_query_time = None
            if counters_copy["query_time_count"] > 0:
                avg_query_time = counters_copy["query_time_sum"] / counters_copy["query_time_count"]

            # Рассчитываем процент успешных запросов
            success_rate = None
            if counters_copy["queries"] > 0:
                success_rate = (
                    counters_copy["successful_queries"] / counters_copy["queries"]
                ) * 100

            metrics = {
                "timestamp": time.time(),
                "queries": {
                    "total": counters_copy["queries"],
                    "successful": counters_copy["successful_queries"],
                    "failed": counters_copy["failed_queries"],
                    "success_rate": success_rate,
                },
                "performance": {"avg_query_time": avg_query_time},
            }

        # Здесь можно добавить код для получения метрик из базы данных
        # Например, количество активных соединений, размер базы данных и т.д.

        # Добавляем метрики в историю
        self.add_metrics(metrics)

        return metrics

    def record_query(self, successful: bool = True, time_ms: Optional[float] = None):
        """
        Записывает запрос к базе данных.

        Args:
            successful: Успешен ли запрос
            time_ms: Время выполнения запроса в миллисекундах
        """
        with self.counters_lock:
            self.counters["queries"] += 1

            if successful:
                self.counters["successful_queries"] += 1
            else:
                self.counters["failed_queries"] += 1

            if time_ms is not None:
                self.counters["query_time_sum"] += time_ms
                self.counters["query_time_count"] += 1


class AlertRule:
    """Правило для генерации оповещений."""

    def __init__(
        self,
        name: str,
        metric_collector: str,
        metric_path: str,
        condition: str,
        threshold: float,
        severity: str = "warning",
        description: str = "",
        cooldown: int = 300,  # Период охлаждения в секундах
    ):
        """
        Инициализирует правило оповещения.

        Args:
            name: Имя правила
            metric_collector: Имя сборщика метрик
            metric_path: Путь к метрике (например, "cpu.percent")
            condition: Условие (">", "<", ">=", "<=", "==", "!=")
            threshold: Пороговое значение
            severity: Уровень важности ("info", "warning", "error", "critical")
            description: Описание правила
            cooldown: Период охлаждения в секундах
        """
        self.name = name
        self.metric_collector = metric_collector
        self.metric_path = metric_path
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.description = description
        self.cooldown = cooldown

        # Время последнего оповещения
        self.last_alert_time = 0

    def check(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Проверяет метрики на соответствие правилу.

        Args:
            metrics: Метрики для проверки

        Returns:
            Оповещение или None, если метрики не соответствуют правилу
        """
        # Получаем значение метрики
        metric_value = self._get_metric_value(metrics, self.metric_path)

        if metric_value is None:
            return None

        # Проверяем условие
        if self._check_condition(metric_value):
            # Проверяем период охлаждения
            current_time = time.time()
            if current_time - self.last_alert_time < self.cooldown:
                return None

            # Обновляем время последнего оповещения
            self.last_alert_time = current_time

            # Создаем оповещение
            return {
                "name": self.name,
                "severity": self.severity,
                "metric_collector": self.metric_collector,
                "metric_path": self.metric_path,
                "metric_value": metric_value,
                "threshold": self.threshold,
                "condition": self.condition,
                "description": self.description,
                "timestamp": current_time,
            }

        return None

    def _get_metric_value(self, metrics: Dict[str, Any], path: str) -> Optional[float]:
        """
        Получает значение метрики по пути.

        Args:
            metrics: Метрики
            path: Путь к метрике

        Returns:
            Значение метрики или None, если метрика не найдена
        """
        keys = path.split(".")
        value = metrics

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        if isinstance(value, (int, float)):
            return value

        return None

    def _check_condition(self, value: float) -> bool:
        """
        Проверяет условие.

        Args:
            value: Значение метрики

        Returns:
            True, если условие выполняется, иначе False
        """
        if self.condition == ">":
            return value > self.threshold
        elif self.condition == "<":
            return value < self.threshold
        elif self.condition == ">=":
            return value >= self.threshold
        elif self.condition == "<=":
            return value <= self.threshold
        elif self.condition == "==":
            return value == self.threshold
        elif self.condition == "!=":
            return value != self.threshold

        return False


class AlertManager:
    """Менеджер оповещений."""

    def __init__(self):
        """Инициализирует менеджер оповещений."""
        self.rules = []
        self.alerts = []
        self.alert_handlers = []

        # Максимальное количество сохраняемых оповещений
        self.max_alerts = 1000

    def add_rule(self, rule: AlertRule):
        """
        Добавляет правило оповещения.

        Args:
            rule: Правило оповещения
        """
        self.rules.append(rule)

    def add_rules(self, rules: List[AlertRule]):
        """
        Добавляет несколько правил оповещения.

        Args:
            rules: Список правил оповещения
        """
        self.rules.extend(rules)

    def check_metrics(self, collector_name: str, metrics: Dict[str, Any]):
        """
        Проверяет метрики на соответствие правилам.

        Args:
            collector_name: Имя сборщика метрик
            metrics: Метрики для проверки
        """
        # Фильтруем правила по имени сборщика метрик
        applicable_rules = [rule for rule in self.rules if rule.metric_collector == collector_name]

        # Проверяем каждое правило
        for rule in applicable_rules:
            alert = rule.check(metrics)

            if alert:
                # Добавляем оповещение в список
                self.alerts.append(alert)

                # Ограничиваем количество сохраняемых оповещений
                if len(self.alerts) > self.max_alerts:
                    self.alerts.pop(0)

                # Вызываем обработчики оповещений
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in alert handler: {e}")

    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Добавляет обработчик оповещений.

        Args:
            handler: Обработчик оповещений
        """
        self.alert_handlers.append(handler)

    def get_alerts(
        self,
        severity: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список оповещений.

        Args:
            severity: Фильтр по уровню важности
            start_time: Начальное время
            end_time: Конечное время

        Returns:
            Список оповещений
        """
        # Фильтруем оповещения
        filtered_alerts = self.alerts

        if severity:
            filtered_alerts = [alert for alert in filtered_alerts if alert["severity"] == severity]

        if start_time:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert["timestamp"] >= start_time
            ]

        if end_time:
            filtered_alerts = [alert for alert in filtered_alerts if alert["timestamp"] <= end_time]

        return filtered_alerts


class SystemMonitor:
    """
    Мониторинг производительности и использования ресурсов.

    Отслеживает и анализирует ключевые метрики производительности системы,
    приложения и базы данных.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        system_metrics_interval: int = 60,
        application_metrics_interval: int = 60,
        database_metrics_interval: int = 60,
    ):
        """
        Инициализирует SystemMonitor.

        Args:
            config: Конфигурация
            system_metrics_interval: Интервал сбора системных метрик в секундах
            application_metrics_interval: Интервал сбора метрик приложения в секундах
            database_metrics_interval: Интервал сбора метрик базы данных в секундах
        """
        self.config = config or {}

        # Сборщики метрик
        self.collectors = {
            "system": SystemMetricsCollector(interval=system_metrics_interval),
            "application": ApplicationMetricsCollector(interval=application_metrics_interval),
            "database": DatabaseMetricsCollector(interval=database_metrics_interval),
        }

        # Менеджер оповещений
        self.alert_manager = AlertManager()

        # Добавляем стандартные правила оповещений
        self._add_default_alert_rules()

        # Очередь событий для сохранения метрик
        self.event_queue = queue.Queue()

        # Поток для сбора метрик
        self.collector_threads = {}

        # Поток для обработки событий
        self.event_thread = None

        # Флаг для управления работой
        self.running = False

        # Путь для сохранения метрик
        self.metrics_path = self.config.get("metrics_path", "metrics")

    def _add_default_alert_rules(self):
        """Добавляет стандартные правила оповещений."""
        # Правила для системных метрик
        self.alert_manager.add_rules(
            [
                AlertRule(
                    name="High CPU Usage",
                    metric_collector="system",
                    metric_path="cpu.percent",
                    condition=">",
                    threshold=90,
                    severity="warning",
                    description="CPU usage is above 90%",
                ),
                AlertRule(
                    name="High Memory Usage",
                    metric_collector="system",
                    metric_path="memory.percent",
                    condition=">",
                    threshold=90,
                    severity="warning",
                    description="Memory usage is above 90%",
                ),
                AlertRule(
                    name="High Swap Usage",
                    metric_collector="system",
                    metric_path="swap.percent",
                    condition=">",
                    threshold=80,
                    severity="warning",
                    description="Swap usage is above 80%",
                ),
            ]
        )

        # Правила для метрик приложения
        self.alert_manager.add_rules(
            [
                AlertRule(
                    name="High Request Failure Rate",
                    metric_collector="application",
                    metric_path="requests.success_rate",
                    condition="<",
                    threshold=95,
                    severity="warning",
                    description="Request success rate is below 95%",
                ),
                AlertRule(
                    name="High Average Processing Time",
                    metric_collector="application",
                    metric_path="performance.avg_processing_time",
                    condition=">",
                    threshold=1000,
                    severity="warning",
                    description="Average processing time is above 1000ms",
                ),
            ]
        )

        # Правила для метрик базы данных
        self.alert_manager.add_rules(
            [
                AlertRule(
                    name="High Query Failure Rate",
                    metric_collector="database",
                    metric_path="queries.success_rate",
                    condition="<",
                    threshold=95,
                    severity="warning",
                    description="Query success rate is below 95%",
                ),
                AlertRule(
                    name="High Average Query Time",
                    metric_collector="database",
                    metric_path="performance.avg_query_time",
                    condition=">",
                    threshold=100,
                    severity="warning",
                    description="Average query time is above 100ms",
                ),
            ]
        )

    def start(self):
        """Запускает мониторинг."""
        if self.running:
            return

        self.running = True

        # Создаем директорию для метрик, если она не существует
        os.makedirs(self.metrics_path, exist_ok=True)

        # Запускаем потоки сбора метрик
        for name, collector in self.collectors.items():
            thread = threading.Thread(target=self._collector_worker, args=(name, collector))
            thread.daemon = True
            thread.start()
            self.collector_threads[name] = thread

        # Запускаем поток обработки событий
        self.event_thread = threading.Thread(target=self._event_worker)
        self.event_thread.daemon = True
        self.event_thread.start()

        logger.info("System monitoring started")

    def stop(self):
        """Останавливает мониторинг."""
        if not self.running:
            return

        self.running = False

        # Ждем завершения потоков
        for thread in self.collector_threads.values():
            thread.join(timeout=5)

        if self.event_thread:
            self.event_thread.join(timeout=5)

        self.collector_threads = {}
        self.event_thread = None

        logger.info("System monitoring stopped")

    def _collector_worker(self, name: str, collector: MetricCollector):
        """
        Рабочий метод для сбора метрик.

        Args:
            name: Имя сборщика метрик
            collector: Сборщик метрик
        """
        logger.info(f"Collector {name} started")

        while self.running:
            try:
                # Собираем метрики
                metrics = collector.collect()

                # Проверяем метрики на соответствие правилам оповещений
                self.alert_manager.check_metrics(name, metrics)

                # Добавляем событие в очередь
                self.event_queue.put({"type": "metrics", "collector": name, "metrics": metrics})

                # Ждем до следующего сбора метрик
                time.sleep(collector.interval)
            except Exception as e:
                logger.error(f"Error in collector {name}: {e}")
                time.sleep(1)

        logger.info(f"Collector {name} stopped")

    def _event_worker(self):
        """Рабочий метод для обработки событий."""
        logger.info("Event worker started")

        while self.running:
            try:
                # Пытаемся получить событие из очереди с тайм-аутом
                try:
                    event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Обрабатываем событие
                if event["type"] == "metrics":
                    # Сохраняем метрики в файл
                    self._save_metrics(event["collector"], event["metrics"])

                # Сообщаем очереди, что событие обработано
                self.event_queue.task_done()
            except Exception as e:
                logger.error(f"Error in event worker: {e}")

        logger.info("Event worker stopped")

    def _save_metrics(self, collector_name: str, metrics: Dict[str, Any]):
        """
        Сохраняет метрики в файл.

        Args:
            collector_name: Имя сборщика метрик
            metrics: Метрики для сохранения
        """
        # Создаем путь для сохранения метрик
        collector_path = os.path.join(self.metrics_path, collector_name)
        os.makedirs(collector_path, exist_ok=True)

        # Формируем имя файла на основе времени
        timestamp = metrics.get("timestamp", time.time())
        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
        file_path = os.path.join(collector_path, f"{date_str}.jsonl")

        # Сохраняем метрики в файл
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Error saving metrics to file {file_path}: {e}")

    def get_metrics(
        self,
        collector_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает метрики.

        Args:
            collector_name: Имя сборщика метрик
            start_time: Начальное время
            end_time: Конечное время

        Returns:
            Метрики
        """
        if collector_name not in self.collectors:
            return {"status": "error", "message": f"Collector {collector_name} not found"}

        collector = self.collectors[collector_name]
        timestamps, metrics = collector.get_metrics_history(start_time, end_time)

        return {
            "status": "success",
            "collector": collector_name,
            "timestamps": timestamps,
            "metrics": metrics,
        }

    def get_latest_metrics(self, collector_name: str) -> Dict[str, Any]:
        """
        Возвращает последние метрики.

        Args:
            collector_name: Имя сборщика метрик

        Returns:
            Последние метрики
        """
        if collector_name not in self.collectors:
            return {"status": "error", "message": f"Collector {collector_name} not found"}

        collector = self.collectors[collector_name]
        metrics = collector.get_latest_metrics()

        return {"status": "success", "collector": collector_name, "metrics": metrics}

    def get_metrics_statistics(
        self,
        collector_name: str,
        metric_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает статистику метрики.

        Args:
            collector_name: Имя сборщика метрик
            metric_name: Имя метрики
            start_time: Начальное время
            end_time: Конечное время

        Returns:
            Статистика метрики
        """
        if collector_name not in self.collectors:
            return {"status": "error", "message": f"Collector {collector_name} not found"}

        collector = self.collectors[collector_name]
        statistics = collector.calculate_statistics(metric_name, start_time, end_time)

        return {
            "status": "success",
            "collector": collector_name,
            "metric": metric_name,
            "statistics": statistics,
        }

    def get_all_collectors(self) -> Dict[str, Any]:
        """
        Возвращает список всех сборщиков метрик.

        Returns:
            Список сборщиков метрик
        """
        collectors = {}

        for name, collector in self.collectors.items():
            collectors[name] = {
                "name": collector.name,
                "interval": collector.interval,
                "metrics_count": len(collector.metrics),
            }

        return {"status": "success", "collectors": collectors}

    def get_alerts(
        self,
        severity: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает список оповещений.

        Args:
            severity: Фильтр по уровню важности
            start_time: Начальное время
            end_time: Конечное время

        Returns:
            Список оповещений
        """
        alerts = self.alert_manager.get_alerts(severity, start_time, end_time)

        return {"status": "success", "alerts": alerts, "count": len(alerts)}

    def add_alert_rule(self, rule: AlertRule) -> Dict[str, Any]:
        """
        Добавляет правило оповещения.

        Args:
            rule: Правило оповещения

        Returns:
            Результат добавления
        """
        self.alert_manager.add_rule(rule)

        return {"status": "success", "message": f"Alert rule {rule.name} added"}

    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        """
        Добавляет обработчик оповещений.

        Args:
            handler: Обработчик оповещений

        Returns:
            Результат добавления
        """
        self.alert_manager.add_alert_handler(handler)

        return {"status": "success", "message": "Alert handler added"}

    def record_request(self, successful: bool = True, time_ms: Optional[float] = None):
        """
        Записывает запрос.

        Args:
            successful: Успешен ли запрос
            time_ms: Время обработки запроса в миллисекундах
        """
        if "application" in self.collectors:
            collector = self.collectors["application"]
            collector.record_request(successful)

            if time_ms is not None:
                collector.record_processing_time(time_ms)

    def record_error(self):
        """Записывает ошибку."""
        if "application" in self.collectors:
            collector = self.collectors["application"]
            collector.record_error()

    def record_query(self, successful: bool = True, time_ms: Optional[float] = None):
        """
        Записывает запрос к базе данных.

        Args:
            successful: Успешен ли запрос
            time_ms: Время выполнения запроса в миллисекундах
        """
        if "database" in self.collectors:
            collector = self.collectors["database"]
            collector.record_query(successful, time_ms)

    def load_metrics_from_file(self, collector_name: str, date_str: str) -> Dict[str, Any]:
        """
        Загружает метрики из файла.

        Args:
            collector_name: Имя сборщика метрик
            date_str: Дата в формате YYYYMMDD

        Returns:
            Загруженные метрики
        """
        # Формируем путь к файлу метрик
        file_path = os.path.join(self.metrics_path, collector_name, f"{date_str}.jsonl")

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"Metrics file {file_path} not found"}

        try:
            # Загружаем метрики из файла
            metrics = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        metric = json.loads(line.strip())
                        metrics.append(metric)
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing metric from file {file_path}")

            return {
                "status": "success",
                "collector": collector_name,
                "date": date_str,
                "metrics": metrics,
                "count": len(metrics),
            }
        except Exception as e:
            logger.error(f"Error loading metrics from file {file_path}: {e}")

            return {"status": "error", "message": f"Error loading metrics: {str(e)}"}

    def get_available_metrics_files(self) -> Dict[str, Any]:
        """
        Возвращает список доступных файлов метрик.

        Returns:
            Список файлов метрик
        """
        files = {}

        for collector_name in self.collectors.keys():
            collector_path = os.path.join(self.metrics_path, collector_name)

            if os.path.exists(collector_path):
                collector_files = [f for f in os.listdir(collector_path) if f.endswith(".jsonl")]
                files[collector_name] = collector_files

        return {"status": "success", "files": files}
