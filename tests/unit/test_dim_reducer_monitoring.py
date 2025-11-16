import pytest
from datetime import datetime, timedelta
import time
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from seo_ai_models.models.dim_reducer.monitoring import DimReducerMonitor

@pytest.fixture
def monitor():
    """Фикстура для создания монитора"""
    return DimReducerMonitor()

@pytest.fixture
def sample_times():
    """Фикстура с тестовыми временными метками"""
    now = datetime.now()
    return {
        'start': now - timedelta(seconds=1),
        'end': now
    }

def test_monitor_initialization(monitor):
    """Тест инициализации монитора"""
    assert monitor.registry is not None
    assert monitor.history is not None
    assert isinstance(monitor.history_window, timedelta)
    assert monitor.history['inference_times'] == []
    assert monitor.history['reconstruction_errors'] == []

def test_track_inference(monitor, sample_times):
    """Тест отслеживания времени инференса"""
    monitor.track_inference(
        sample_times['start'],
        sample_times['end'],
        endpoint='analyze',
        status='success'
    )
    
    assert len(monitor.history['inference_times']) == 1
    assert len(monitor.history['request_timestamps']) == 1
    assert 0 < monitor.history['inference_times'][0] <= 1.0

def test_track_batch(monitor):
    """Тест отслеживания размера батча"""
    batch_size = 32
    with patch('prometheus_client.Histogram.observe') as mock_observe:
        monitor.track_batch(batch_size)
        mock_observe.assert_called_once_with(batch_size)

def test_track_reconstruction(monitor):
    """Тест отслеживания ошибки реконструкции"""
    error = 0.15
    monitor.track_reconstruction(error)
    
    assert len(monitor.history['reconstruction_errors']) == 1
    assert monitor.history['reconstruction_errors'][0] == error

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_track_resources_gpu(monitor):
    """Тест отслеживания ресурсов с GPU"""
    monitor.track_resources()
    metrics = monitor.get_metrics()
    
    assert 'gpu_memory_percent' in metrics
    assert isinstance(metrics['gpu_memory_percent'], (int, float))
    assert 0 <= metrics['gpu_memory_percent'] <= 100

def test_track_resources_cpu(monitor):
    """Тест отслеживания ресурсов CPU"""
    monitor.track_resources()
    metrics = monitor.get_metrics()
    
    assert 'cpu_percent' in metrics
    assert isinstance(metrics['cpu_percent'], (int, float))
    assert 0 <= metrics['cpu_percent'] <= 100
    assert 'ram_percent' in metrics
    assert 0 <= metrics['ram_percent'] <= 100

def test_compression_tracking(monitor):
    """Тест отслеживания коэффициента сжатия"""
    input_dim = 768
    latent_dim = 256
    
    with patch('prometheus_client.Gauge.set') as mock_set:
        monitor.track_compression(input_dim, latent_dim)
        mock_set.assert_called_once_with(input_dim / latent_dim)

def test_request_rate_calculation(monitor):
    """Тест расчета частоты запросов"""
    # Добавляем несколько запросов
    now = datetime.now()
    timestamps = [
        now - timedelta(seconds=10),
        now - timedelta(seconds=20),
        now - timedelta(seconds=30)
    ]
    
    monitor.history['request_timestamps'] = timestamps
    rate = monitor._calculate_request_rate()
    
    assert isinstance(rate, (int, float))
    assert rate == 3  # 3 запроса за последнюю минуту

def test_history_cleanup(monitor):
    """Тест очистки истории"""
    # Добавляем старые записи
    old_time = datetime.now() - monitor.history_window - timedelta(minutes=1)
    recent_time = datetime.now()
    
    monitor.history['request_timestamps'] = [old_time, recent_time]
    monitor.history['inference_times'] = [0.1, 0.2]
    monitor.history['reconstruction_errors'] = [0.01, 0.02]
    
    monitor._cleanup_history()
    
    # Проверяем, что старые записи удалены
    assert len(monitor.history['request_timestamps']) == 1
    assert len(monitor.history['inference_times']) == 1
    assert len(monitor.history['reconstruction_errors']) == 1

def test_metrics_calculation(monitor, sample_times):
    """Тест расчета метрик"""
    # Добавляем тестовые данные
    monitor.track_inference(sample_times['start'], sample_times['end'])
    monitor.track_reconstruction(0.1)
    
    metrics = monitor.get_metrics()
    
    assert 'avg_inference_time' in metrics
    assert 'avg_reconstruction_error' in metrics
    assert 'requests_per_minute' in metrics
    assert isinstance(metrics['avg_inference_time'], float)
    assert isinstance(metrics['requests_per_minute'], (int, float))

def test_reset_functionality(monitor, sample_times):
    """Тест сброса метрик"""
    # Добавляем данные
    monitor.track_inference(sample_times['start'], sample_times['end'])
    monitor.track_reconstruction(0.1)
    
    # Сбрасываем
    monitor.reset()
    
    assert len(monitor.history['inference_times']) == 0
    assert len(monitor.history['reconstruction_errors']) == 0
    assert len(monitor.history['request_timestamps']) == 0

def test_error_handling(monitor):
    """Тест обработки ошибок"""
    # Симулируем ошибку при получении метрик CPU
    with patch('psutil.cpu_percent', side_effect=Exception('Test error')):
        metrics = monitor.get_metrics()
        # Проверяем, что функция не упала и вернула пустой словарь
        assert isinstance(metrics, dict)

@pytest.mark.parametrize("batch_size", [1, 10, 50, 100])
def test_different_batch_sizes(monitor, batch_size):
    """Тест различных размеров батча"""
    with patch('prometheus_client.Histogram.observe') as mock_observe:
        monitor.track_batch(batch_size)
        mock_observe.assert_called_once_with(batch_size)

def test_concurrent_updates(monitor):
    """Тест параллельных обновлений метрик"""
    from concurrent.futures import ThreadPoolExecutor
    import random
    
    def random_update():
        time.sleep(random.random() * 0.1)
        start_time = datetime.now() - timedelta(seconds=1)
        end_time = datetime.now()
        monitor.track_inference(start_time, end_time)
    
    # Запускаем несколько параллельных обновлений
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(random_update) for _ in range(10)]
        for future in futures:
            future.result()
    
    # Проверяем, что все обновления были учтены
    assert len(monitor.history['inference_times']) == 10
