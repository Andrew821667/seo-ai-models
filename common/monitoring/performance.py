# monitoring/performance.py

import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Мониторинг производительности модели"""
    def __init__(self):
        self.metrics_history = {
            'inference_time': [],
            'memory_usage': [],
            'request_count': 0,
            'error_count': 0
        }
        self.start_time = datetime.now()

    def track_inference(self, start_time: datetime, end_time: datetime):
        """
        Отслеживание времени инференса
        Args:
            start_time: время начала
            end_time: время окончания
        """
        duration = (end_time - start_time).total_seconds()
        self.metrics_history['inference_time'].append(duration)
        logger.info(f"Inference time: {duration:.3f} seconds")

    def track_memory(self):
        """Отслеживание использования памяти"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            self.metrics_history['memory_usage'].append({
                'allocated': memory_allocated,
                'cached': memory_cached
            })
            logger.info(f"Memory usage: {memory_allocated:.2f}MB allocated, {memory_cached:.2f}MB cached")

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Получение статистики производительности
        Returns:
            словарь со статистикой
        """
        stats = {
            'average_inference_time': np.mean(self.metrics_history['inference_time']),
            'max_inference_time': np.max(self.metrics_history['inference_time']),
            'total_requests': self.metrics_history['request_count'],
            'error_rate': self.metrics_history['error_count'] / max(self.metrics_history['request_count'], 1),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

        if self.metrics_history['memory_usage']:
            stats.update({
                'average_memory_usage_mb': np.mean([m['allocated'] for m in self.metrics_history['memory_usage']]),
                'peak_memory_usage_mb': np.max([m['allocated'] for m in self.metrics_history['memory_usage']])
            })

        return stats

    def log_performance_alert(self, metric: str, value: float, threshold: float):
        """
        Логирование предупреждений о производительности
        Args:
            metric: название метрики
            value: значение
            threshold: пороговое значение
        """
        if value > threshold:
            logger.warning(
                f"Performance alert: {metric} value {value:.2f} exceeds threshold {threshold:.2f}"
            )
