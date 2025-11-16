import pytest
import torch
import tempfile
from pathlib import Path

from seo_ai_models.models.dim_reducer.model import DimensionReducer
from seo_ai_models.models.dim_reducer.trainer import DimReducerTrainer
from seo_ai_models.common.config.dim_reducer_config import DimReducerConfig

@pytest.fixture
def config():
    """Фикстура для конфигурации"""
    return DimReducerConfig(
        input_dim=10,
        hidden_dim=8,
        latent_dim=4,
        batch_size=2,
        num_epochs=2
    )

@pytest.fixture
def model(config):
    """Фикстура для модели"""
    return DimensionReducer(config)

@pytest.fixture
def trainer(model, config):
    """Фикстура для тренера"""
    return DimReducerTrainer(model, config)

@pytest.fixture
def sample_batch():
    """Фикстура для тестового батча"""
    return torch.randn(2, 10)  # batch_size=2, input_dim=10

def test_trainer_initialization(trainer):
    """Тест инициализации тренера"""
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.reconstruction_loss is not None
    assert isinstance(trainer.history, dict)

def test_train_step(trainer, sample_batch):
    """Тест одного шага обучения"""
    metrics = trainer.train_step(sample_batch)
    
    assert isinstance(metrics, dict)
    assert 'total_loss' in metrics
    assert 'reconstruction_loss' in metrics
    assert 'sparsity_loss' in metrics
    assert 'importance_reg' in metrics
    
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(v >= 0 for v in metrics.values())

def test_validation(trainer, sample_batch):
    """Тест валидации"""
    # Создаем простой валидационный загрузчик
    val_loader = [sample_batch]
    
    metrics = trainer.validate(val_loader)
    
    assert isinstance(metrics, dict)
    assert 'val_loss' in metrics
    assert isinstance(metrics['val_loss'], float)
    assert metrics['val_loss'] >= 0

def test_checkpoint_saving_loading(trainer, sample_batch, tmp_path):
    """Тест сохранения и загрузки чекпоинтов"""
    # Выполняем один шаг обучения
    metrics = trainer.train_step(sample_batch)
    
    # Сохраняем чекпоинт
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path), epoch=0, metrics=metrics)
    
    assert checkpoint_path.exists()
    
    # Загружаем чекпоинт
    epoch, loaded_metrics = trainer.load_checkpoint(str(checkpoint_path))
    
    assert epoch == 0
    assert loaded_metrics == metrics

def test_train_epoch(trainer, sample_batch):
    """Тест обучения одной эпохи"""
    # Создаем простой тренировочный загрузчик
    train_loader = [sample_batch] * 3  # 3 итерации
    
    metrics = trainer.train_epoch(
        train_loader,
        epoch=0,
        val_loader=[sample_batch]
    )
    
    assert isinstance(metrics, dict)
    assert 'train_loss' in metrics
    assert 'val_loss' in metrics
    assert len(trainer.history['train_loss']) == 1

def test_model_device_moving(config):
    """Тест перемещения модели на устройство"""
    model = DimensionReducer(config)
    trainer = DimReducerTrainer(
        model,
        config,
        device='cpu'  # Явно указываем CPU для тестов
    )
    
    assert next(trainer.model.parameters()).device.type == 'cpu'

def test_gradient_clipping(trainer, sample_batch):
    """Тест ограничения градиентов"""
    # Выполняем шаг обучения
    trainer.train_step(sample_batch)
    
    # Проверяем, что градиенты параметров не превышают максимальное значение
    max_grad_norm = 1.0
    for param in trainer.model.parameters():
        if param.grad is not None:
            assert param.grad.norm() <= max_grad_norm * 1.1  # небольшой допуск

def test_history_tracking(trainer, sample_batch):
    """Тест отслеживания истории обучения"""
    train_loader = [sample_batch] * 2
    
    # Обучаем две эпохи
    for epoch in range(2):
        trainer.train_epoch(train_loader, epoch)
    
    assert len(trainer.history['train_loss']) == 2
    assert all(isinstance(loss, float) for loss in trainer.history['train_loss'])
