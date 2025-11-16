import pytest
import torch
import numpy as np
from pathlib import Path

from seo_ai_models.models.dim_reducer.model import DimensionReducer
from seo_ai_models.models.dim_reducer.inference import DimReducerInference
from seo_ai_models.common.config.dim_reducer_config import DimReducerConfig

@pytest.fixture
def config():
    """Фикстура для конфигурации"""
    return DimReducerConfig(
        input_dim=10,
        hidden_dim=8,
        latent_dim=4,
        max_length=128
    )

@pytest.fixture
def model_path(config, tmp_path):
    """Фикстура для создания и сохранения тестовой модели"""
    model = DimensionReducer(config)
    path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), path)
    return str(path)

@pytest.fixture
def inference(model_path, config):
    """Фикстура для инференса"""
    return DimReducerInference(model_path, config, device='cpu')

@pytest.fixture
def sample_features():
    """Фикстура для тестовых признаков"""
    return torch.randn(5, 10)  # 5 samples, 10 features

def test_inference_initialization(inference):
    """Тест инициализации инференса"""
    assert inference.model is not None
    assert inference.preprocessor is not None
    assert next(inference.model.parameters()).device.type == 'cpu'

def test_reduce_dimensions_tensor(inference, sample_features):
    """Тест сжатия размерности для тензора"""
    results = inference.reduce_dimensions(sample_features)
    
    assert isinstance(results, dict)
    assert 'latent_features' in results
    assert 'reconstructed' in results
    assert 'feature_importance' in results
    
    assert isinstance(results['latent_features'], np.ndarray)
    assert results['latent_features'].shape == (5, 4)  # batch_size=5, latent_dim=4
    assert results['reconstructed'].shape == (5, 10)  # batch_size=5, input_dim=10

def test_reduce_dimensions_numpy(inference):
    """Тест сжатия размерности для numpy массива"""
    features = np.random.randn(5, 10)
    results = inference.reduce_dimensions(features)
    
    assert isinstance(results['latent_features'], np.ndarray)
    assert isinstance(results['reconstructed'], np.ndarray)

def test_process_text(inference):
    """Тест обработки текста"""
    text = "Sample text for testing"
    results = inference.process_text(text)
    
    assert isinstance(results, dict)
    assert results['num_samples'] == 1
    assert 'latent_features' in results
    assert 'reconstructed' in results

def test_process_multiple_texts(inference):
    """Тест обработки нескольких текстов"""
    texts = ["First text", "Second text", "Third text"]
    results = inference.process_text(texts)
    
    assert results['num_samples'] == 3
    assert len(results['latent_features']) == 3

def test_analyze_feature_importance(inference, sample_features):
    """Тест анализа важности признаков"""
    results = inference.analyze_feature_importance(sample_features)
    
    assert 'mean_importance' in results
    assert 'std_importance' in results
    assert 'feature_ranking' in results
    
    assert results['mean_importance'].shape == (1,)  # Для каждого признака
    assert results['feature_ranking'].shape == (10,)  # Для всех входных признаков

def test_batch_process(inference):
    """Тест пакетной обработки"""
    features = np.random.randn(100, 10)  # Много семплов
    results = inference.batch_process(features, batch_size=32)
    
    assert len(results['latent_features']) == 100
    assert len(results['reconstructed']) == 100

def test_error_handling(inference):
    """Тест обработки ошибок"""
    with pytest.raises(Exception):
        # Передаем неправильную размерность
        invalid_features = torch.randn(5, 20)  # Неправильное количество признаков
        inference.reduce_dimensions(invalid_features)

def test_device_handling(inference):
    """Тест обработки устройств"""
    # Проверяем, что данные корректно перемещаются на CPU
    features = torch.randn(5, 10, device='cpu')
    results = inference.reduce_dimensions(features)
    
    assert isinstance(results['latent_features'], np.ndarray)
    assert isinstance(results['reconstructed'], np.ndarray)

def test_model_consistency(inference, sample_features):
    """Тест консистентности модели"""
    # Проверяем, что модель дает одинаковые результаты для одних и тех же входных данных
    results1 = inference.reduce_dimensions(sample_features)
    results2 = inference.reduce_dimensions(sample_features)
    
    np.testing.assert_array_almost_equal(
        results1['latent_features'],
        results2['latent_features']
    )
    np.testing.assert_array_almost_equal(
        results1['reconstructed'],
        results2['reconstructed']
    )
