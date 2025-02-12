# model/monitoring/performance.py

import time
import psutil
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from collections import deque

from .logger import KeywordExtractorLogger

logger = KeywordExtractorLogger().get_logger()

@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[float]
    batch_processing_time: float
    throughput: float
    cuda_memory_allocated: Optional[float]
    cuda_memory_cached: Optional[float]

class PerformanceMonitor:
    """Мониторинг производительности модели"""
    
    def __init__(
        self,
        metrics_history_size: int = 1000,
        save_dir: Optional[Union[str, Path]] = None
    ):
        """
        Инициализация монитора
        
        Args:
            metrics_history_size: Размер истории метрик
            save_dir: Директория для сохранения метрик
        """
        self.metrics_history = deque(maxlen=metrics_history_size)
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # Проверка доступности CUDA
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.gpu_handle = torch.cuda.current_device()
            
    def start_batch(self) -> float:
        """
        Начало отслеживания обработки батча
        
        Returns:
            Время начала
        """
        if self.cuda_available:
            torch.cuda.synchronize()
        return time.perf_counter()
        
    def end_batch(
        self,
        start_time: float,
        batch_size: int
    ) -> Dict[str, float]:
        """
        Окончание отслеживания обработки батча
        
        Args:
            start_time: Время начала
            batch_size: Размер батча
            
        Returns:
            Метрики производительности
        """
        if self.cuda_available:
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            gpu_memory_used=self._get_gpu_memory_used() if self.cuda_available else None,
            batch_processing_time=processing_time,
            throughput=batch_size / processing_time,
            cuda_memory_allocated=torch.cuda.memory_allocated() if self.cuda_available else None,
            cuda_memory_cached=torch.cuda.memory_reserved() if self.cuda_available else None
        )
        
        self.metrics_history.append(asdict(metrics))
        return asdict(metrics)
        
    def _get_gpu_memory_used(self) -> float:
        """
        Получение использования памяти GPU
        
        Returns:
            Объем используемой памяти в GB
        """
        try:
            return torch.cuda.memory_allocated(self.gpu_handle) / 1024**3
        except Exception as e:
            logger.warning(f"Ошибка при получении статистики GPU: {e}")
            return 0.0
            
    def get_average_metrics(
        self,
        window_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Получение усредненных метрик
        
        Args:
            window_size: Размер окна для усреднения
            
        Returns:
            Усредненные метрики
        """
        if not self.metrics_history:
            return
            
        latest_metrics = self.metrics_history[-1]
        
        # Проверка CPU
        if latest_metrics['cpu_percent'] > cpu_threshold:
            logger.warning(
                f"Высокая загрузка CPU: {latest_metrics['cpu_percent']:.1f}%"
            )
            
        # Проверка памяти
        if latest_metrics['memory_percent'] > memory_threshold:
            logger.warning(
                f"Высокое использование памяти: {latest_metrics['memory_percent']:.1f}%"
            )
            
        # Проверка GPU если доступно
        if self.cuda_available and latest_metrics['gpu_memory_used'] is not None:
            gpu_memory_percent = (latest_metrics['gpu_memory_used'] / 
                                torch.cuda.get_device_properties(self.gpu_handle).total_memory) * 100
            if gpu_memory_percent > gpu_memory_threshold:
                logger.warning(
                    f"Высокое использование памяти GPU: {gpu_memory_percent:.1f}%"
                )
                
    def profile_batch_processing(
        self,
        batch_size: int,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Профилирование обработки батчей
        
        Args:
            batch_size: Размер батча
            num_iterations: Количество итераций
            
        Returns:
            Статистика производительности
        """
        processing_times = []
        
        for _ in range(num_iterations):
            start_time = self.start_batch()
            if self.cuda_available:
                torch.cuda.synchronize()
            # Имитация обработки
            time.sleep(0.01)  
            metrics = self.end_batch(start_time, batch_size)
            processing_times.append(metrics['batch_processing_time'])
        
        return {
            'avg_processing_time': float(np.mean(processing_times)),
            'std_processing_time': float(np.std(processing_times)),
            'min_processing_time': float(np.min(processing_times)),
            'max_processing_time': float(np.max(processing_times)),
            'avg_throughput': batch_size / float(np.mean(processing_times))
        }
        
    def monitor_memory_usage(
        self,
        interval: float = 1.0,
        duration: float = 60.0
    ) -> Dict[str, List[float]]:
        """
        Мониторинг использования памяти
        
        Args:
            interval: Интервал измерений в секундах
            duration: Продолжительность мониторинга в секундах
            
        Returns:
            История использования памяти
        """
        memory_history = {
            'timestamps': [],
            'cpu_memory': [],
            'gpu_memory': [] if self.cuda_available else None
        }
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            memory_history['timestamps'].append(time.time() - start_time)
            memory_history['cpu_memory'].append(psutil.virtual_memory().percent)
            
            if self.cuda_available:
                memory_history['gpu_memory'].append(self._get_gpu_memory_used())
                
            time.sleep(interval)
            
        return memory_history
        
    def export_performance_report(
        self,
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Создание отчета о производительности
        
        Args:
            output_file: Путь для сохранения отчета
            
        Returns:
            Данные отчета
        """
        if not self.metrics_history:
            return {}
            
        # Подготовка данных отчета
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024**3),
                'cuda_available': self.cuda_available
            },
            'performance_summary': self.get_average_metrics(),
            'metrics_history': list(self.metrics_history)
        }
        
        # Добавление информации о GPU
        if self.cuda_available:
            gpu_props = torch.cuda.get_device_properties(self.gpu_handle)
            report['system_info']['gpu_info'] = {
                'name': gpu_props.name,
                'total_memory': gpu_props.total_memory / (1024**3),
                'major': gpu_props.major,
                'minor': gpu_props.minor
            }
            
        # Сохранение отчета
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Отчет о производительности сохранен в {output_path}")
            
        return report self.metrics_history:
            return {}
            
        # Определение размера окна
        if window_size is None or window_size > len(self.metrics_history):
            window_size = len(self.metrics_history)
            
        # Получение последних метрик
        recent_metrics = list(self.metrics_history)[-window_size:]
        
        # Усреднение числовых метрик
        avg_metrics = {}
        for key in recent_metrics[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in recent_metrics if m[key] is not None]
                if values:
                    avg_metrics[key] = float(np.mean(values))
                    
        return avg_metrics
        
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """
        Сохранение метрик в файл
        
        Args:
            filename: Имя файла
        """
        if not self.save_dir:
            logger.warning("Директория для сохранения не указана")
            return
            
        if filename is None:
            filename = f"performance_metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
            
        file_path = self.save_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(list(self.metrics_history), f, indent=2)
            logger.info(f"Метрики сохранены в {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении метрик: {e}")
            
    def log_critical_metrics(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        gpu_memory_threshold: float = 90.0
    ) -> None:
        """
        Логирование критических метрик
        
        Args:
            cpu_threshold: Порог загрузки CPU
            memory_threshold: Порог использования памяти
            gpu_memory_threshold: Порог использования памяти GPU
        """
        if not
