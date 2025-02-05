import time
import psutil
import logging
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta
import torch
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

logger = logging.getLogger(__name__)

class DimReducerMonitor:
    """Система мониторинга для DimensionReducer"""
    
    def __init__(self):
        # Регистр метрик Prometheus
        self.registry = CollectorRegistry()
        
        # Метрики производительности
        self.inference_time = Histogram(
            'dim_reducer_inference_seconds',
            'Time spent processing requests',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.batch_size = Histogram(
            'dim_reducer_batch_size',
            'Distribution of batch sizes',
            buckets=[1, 5, 10, 20, 50, 100],
            registry=self.registry
        )
        
        self.request_counter = Counter(
            'dim_reducer_requests_total',
            'Total number of requests',
            ['endpoint', 'status'],
            registry=self.registry
        )
        
        # Метрики ресурсов
        self.gpu_memory_used = Gauge(
            'dim_reducer_gpu_memory_bytes',
            'GPU memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'dim_reducer_cpu_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.ram_usage = Gauge(
            'dim_reducer_ram_bytes',
            'RAM usage in bytes',
            registry=self.registry
        )
        
        # Метрики качества
        self.reconstruction_error = Histogram(
            'dim_reducer_reconstruction_error',
            'Distribution of reconstruction errors',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        self.compression_ratio = Gauge(
            'dim_reducer_compression_ratio',
            'Current compression ratio',
            registry=self.registry
        )
        
        # История для расчета скользящих средних
        self.history = {
            'inference_times': [],
            'reconstruction_errors': [],
            'request_timestamps': []
        }
        self.history_window = timedelta(minutes=5)
        
    def track_inference(
        self,
        start_time: datetime,
        end_time: datetime,
        endpoint: str = 'analyze',
        status: str = 'success'
    ):
        """Отслеживание времени инференса"""
        duration = (end_time - start_time).total_seconds()
        self.inference_time.observe(duration)
        self.request_counter.labels(endpoint=endpoint, status=status).inc()
        
        # Обновление истории
        self.history['inference_times'].append(duration)
        self.history['request_timestamps'].append(end_time)
        self._cleanup_history()
        
    def track_batch(self, size: int):
        """Отслеживание размера батча"""
        self.batch_size.observe(size)
        
    def track_reconstruction(self, error: float):
        """Отслеживание ошибки реконструкции"""
        self.reconstruction_error.observe(error)
        self.history['reconstruction_errors'].append(error)
        self._cleanup_history()
        
    def track_compression(self, input_dim: int, latent_dim: int):
        """Отслеживание коэффициента сжатия"""
        ratio = input_dim / latent_dim
        self.compression_ratio.set(ratio)
        
    def track_resources(self):
        """Отслеживание использования ресурсов"""
        try:
            # GPU память
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                self.gpu_memory_used.set(gpu_memory)
            
            # CPU и RAM
            cpu_percent = psutil.cpu_percent()
            ram_used = psutil.virtual_memory().used
            
            self.cpu_usage.set(cpu_percent)
            self.ram_usage.set(ram_used)
            
        except Exception as e:
            logger.error(f"Error tracking resources: {e}")
            
    def get_metrics(self) -> Dict[str, float]:
        """Получение текущих метрик"""
        try:
            current_metrics = {
                'avg_inference_time': np.mean(self.history['inference_times']) 
                    if self.history['inference_times'] else 0,
                'avg_reconstruction_error': np.mean(self.history['reconstruction_errors'])
                    if self.history['reconstruction_errors'] else 0,
                'requests_per_minute': self._calculate_request_rate(),
                'cpu_percent': psutil.cpu_percent(),
                'ram_percent': psutil.virtual_memory().percent
            }
            
            if torch.cuda.is_available():
                current_metrics['gpu_memory_percent'] = (
                    torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    if torch.cuda.max_memory_allocated() > 0 else 0
                )
                
            return current_metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
            
    def _calculate_request_rate(self) -> float:
        """Расчет количества запросов в минуту"""
        if not self.history['request_timestamps']:
            return 0.0
            
        now = datetime.now()
        recent_requests = [
            ts for ts in self.history['request_timestamps']
            if now - ts <= timedelta(minutes=1)
        ]
        return len(recent_requests)
        
    def _cleanup_history(self):
        """Очистка устаревших записей"""
        now = datetime.now()
        cutoff = now - self.history_window
        
        # Очистка временных меток
        self.history['request_timestamps'] = [
            ts for ts in self.history['request_timestamps']
            if ts > cutoff
        ]
        
        # Очистка других метрик
        max_history = len(self.history['request_timestamps'])
        self.history['inference_times'] = self.history['inference_times'][-max_history:]
        self.history['reconstruction_errors'] = self.history['reconstruction_errors'][-max_history:]
        
    def reset(self):
        """Сброс всех метрик"""
        for collector in self.registry.collect():
            if hasattr(collector, '_value'):
                collector._value.set(0)
                
        self.history = {
            'inference_times': [],
            'reconstruction_errors': [],
            'request_timestamps': []
        }
