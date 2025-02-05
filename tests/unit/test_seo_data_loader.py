import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from models.dim_reducer.data_loader import SEODataset, SEODataLoader
from models.dim_reducer.features import SEOFeaturesExtractor

@pytest.fixture
def sample_data():
    """Создание тестовых данных"""
    data = pd.DataFrame({
        'text': [
            'First sample text for testing.',
            'Second text with different content.',
            'Third sample with more variations.'
        ],
        'html': [
            '<html><body><h1>First</h1><p>Content</p></body></html>',
            '<html><body><h1>Second</h1><p>Content</p></body></html>',
            '<html><body><h1>Third</h1><p>Content</p></body></html>'
        ]
    })
    return data

@pytest.fixture
def temp_csv(sample_data, tmp_path):
    """Создание временного CSV файла"""
    csv_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def feature_extractor():
    """Создание экстрактора признаков"""
    return SEOFeaturesExtractor(max_features=50)

@pytest.fixture
def dataset(temp_csv, feature_extractor):
    """Создание тестового датасета"""
    return SEODataset(
        data_path=temp_csv,
        feature_extractor=feature_extractor,
        max_features=50,
        cache_features=True
    )

@pytest.fixture
def data_loader():
    """Создание загрузчика данных"""
    return SEODataLoader(
        batch_size=2,
        num_workers=0,  # Используем 0 для тестов
        max_features=50
    )

def test_dataset_initialization(dataset):
    """Тест инициализации датасета"""
    assert len(dataset) == 3
    assert isinstance(dataset.data, pd.DataFrame)
    assert dataset.max_features == 50
    assert dataset.cache_features is True

def test_feature_extraction(dataset):
    """Тест извлечения признаков"""
    features = dataset[0]  # Получаем признаки первого семпла
    
    assert isinstance(features, torch.Tensor)
    assert features.dim() == 1  # Одномерный тензор
    assert not torch.isnan(features).any()  # Нет NaN значений
    assert not torch.isinf(features).any()  # Нет бесконечных значений

def test_dataset_caching(dataset):
    """Тест кэширования признаков"""
    # Первый вызов - извлечение признаков
    features1 = dataset[0]
    
    # Второй вызов - должен использовать кэш
    features2 = dataset[0]
    
    assert torch.equal(features1, features2)
    assert 0 in dataset.features_cache

def test_data_loader_creation(data_loader, temp_csv):
    """Тест создания загрузчиков данных"""
    train_dataset, val_dataset, _ = data_loader.create_datasets(
        train_path=temp_csv,
        val_path=temp_csv  # Используем тот же файл для валидации
    )
    
    train_loader, val_loader, _ = data_loader.get_data_loaders(
        train_dataset,
        val_dataset
    )
    
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2
    assert not train_loader.shuffle == val_loader.shuffle  # Тренировочный перемешивается, валидационный - нет

def test_batch_iteration(data_loader, temp_csv):
    """Тест итерации по батчам"""
    train_dataset, _, _ = data_loader.create_datasets(train_path=temp_csv)
    train_loader, _, _ = data_loader.get_data_loaders(train_dataset)
    
    for batch in train_loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.size(0) <= 2  # Размер батча
        assert not torch.isnan(batch).any()
        break  # Проверяем только первый батч

def test_cache_saving_loading(dataset, tmp_path):
    """Тест сохранения и загрузки кэша"""
    # Извлекаем признаки, чтобы заполнить кэш
    _ = dataset[0]
    
    # Сохраняем кэш
    cache_path = tmp_path / 'cache.pt'
    dataset.save_cache(str(cache_path))
    assert cache_path.exists()
    
    # Создаем новый датасет и загружаем кэш
    new_dataset = SEODataset(
        dataset.data_path,
        dataset.feature_extractor,
        cache_features=True
    )
    new_dataset.load_cache(str(cache_path))
    
    assert torch.equal(dataset[0], new_dataset[0])

def test_error_handling(feature_extractor, tmp_path):
    """Тест обработки ошибок"""
    # Тест с несуществующим файлом
    with pytest.raises(Exception):
        SEODataset(
            tmp_path / 'nonexistent.csv',
            feature_extractor
        )
    
    # Тест с неподдерживаемым форматом файла
    invalid_path = tmp_path / 'test.txt'
    invalid_path.touch()
    with pytest.raises(ValueError):
        SEODataset(invalid_path, feature_extractor)

def test_multiprocessing(data_loader, temp_csv):
    """Тест многопроцессорной загрузки"""
    # Создаем загрузчик с несколькими рабочими процессами
    data_loader.num_workers = 2
    
    train_dataset, _, _ = data_loader.create_datasets(train_path=temp_csv)
    train_loader, _, _ = data_loader.get_data_loaders(train_dataset)
    
    # Проверяем, что загрузка работает
    for batch in train_loader:
        assert isinstance(batch, torch.Tensor)
        break

def test_feature_dimensions(dataset):
    """Тест размерностей признаков"""
    features = dataset[0]
    
    # Проверяем, что размерность соответствует ожидаемой
    expected_dim = 10 + dataset.max_features  # базовые признаки + TF-IDF
    assert features.size(0) == expected_dim

def test_empty_data_handling(feature_extractor, tmp_path):
    """Тест обработки пустых данных"""
    # Создаем пустой DataFrame
    empty_data = pd.DataFrame(columns=['text', 'html'])
    empty_csv = tmp_path / 'empty.csv'
    empty_data.to_csv(empty_csv, index=False)
    
    dataset = SEODataset(
        empty_csv,
        feature_extractor,
        max_features=50
    )
    
    assert len(dataset) == 0
