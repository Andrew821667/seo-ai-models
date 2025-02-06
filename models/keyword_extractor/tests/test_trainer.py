# tests/test_trainer.py

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from model.model import KeywordExtractorModel
from model.model.trainer import KeywordExtractorTrainer
from model.config.model_config import KeywordModelConfig, KeywordTrainingConfig

@pytest.fixture
def training_config():
    """Фикстура с конфигурацией обучения"""
    return KeywordTrainingConfig(
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=2,
        validation_split=0.2
    )

@pytest.fixture
def model():
    """Фикстура с моделью"""
    config = KeywordModelConfig(
        max_length=128,
        hidden_dim=256
    )
    return KeywordExtractorModel(config)

@pytest.fixture
def trainer(model, training_config):
    """Фикстура с тренером"""
    return KeywordExtractorTrainer(
        model=model,
        config=training_config
    )

@pytest.fixture
def sample_batch():
    """Фикстура с тестовым батчем данных"""
    return {
        'input_ids': torch.randint(0, 1000, (2, 128)),
        'attention_mask': torch.ones(2, 128),
        'keyword_labels': torch.randint(0, 2, (2, 128)),
        'trend_labels': torch.rand(2, 128)
    }

class TestKeywordExtractorTrainer:
    """Тесты для тренера модели"""
    
    def test_trainer_initialization(self, model, training_config):
        """Тест инициализации тренера"""
        trainer = KeywordExtractorTrainer(model, training_config)
        
        assert trainer.model == model
        assert trainer.config == training_config
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        
    def test_train_step(self, trainer, sample_batch):
        """Тест шага обучения"""
        # Выполнение шага обучения
        metrics = trainer.train_step(sample_batch)
        
        # Проверка метрик
        assert 'total_loss' in metrics
        assert 'keyword_loss' in metrics
        assert 'trend_loss' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        
    def test_validate(self, trainer, sample_batch):
        """Тест валидации"""
        # Создание валидационного загрузчика
        val_loader = [sample_batch] * 2
        
        # Выполнение валидации
        val_metrics = trainer.validate(val_loader)
        
        assert 'val_loss' in val_metrics
        assert isinstance(val_metrics['val_loss'], float)
        
    def test_checkpoint_saving_loading(self, trainer, tmp_path):
        """Тест сохранения и загрузки чекпоинтов"""
        # Сохранение чекпоинта
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(
            checkpoint_path,
            epoch=0,
            metrics={'loss': 0.5}
        )
        
        # Проверка существования файла
        assert checkpoint_path.exists()
        
        # Загрузка чекпоинта
        epoch, metrics = trainer.load_checkpoint(checkpoint_path)
        
        assert epoch == 0
        assert metrics['loss'] == 0.5
        
    def test_early_stopping(self, trainer, sample_batch):
        """Тест ранней остановки"""
        # Симуляция обучения с ухудшающимися метриками
        val_losses = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for loss in val_losses:
            should_stop = trainer.early_stopping(loss)
            if should_stop:
                break
                
        # Проверка, что обучение остановилось
        assert trainer.early_stopping.should_stop
        assert trainer.early_stopping.best_loss == 0.5
        
    def test_learning_rate_scheduler(self, trainer, sample_batch):
        """Тест планировщика скорости обучения"""
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Выполнение нескольких шагов
        for _ in range(3):
            trainer.train_step(sample_batch)
            
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Проверка изменения скорости обучения
        assert current_lr != initial_lr
        
    def test_gradient_clipping(self, trainer, sample_batch):
        """Тест ограничения градиентов"""
        # Выполнение шага обучения
        trainer.train_step(sample_batch)
        
        # Проверка градиентов
        max_grad_norm = max(
            p.grad.norm().item()
            for p in trainer.model.parameters()
            if p.grad is not None
        )
        
        assert max_grad_norm <= trainer.config.max_grad_norm
        
    @pytest.mark.parametrize(
        "batch_size,num_epochs",
        [(1, 1), (2, 2), (4, 1)]
    )
    def test_different_batch_sizes(
        self,
        model,
        batch_size,
        num_epochs
    ):
        """Тест различных размеров батча"""
        config = KeywordTrainingConfig(
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        trainer = KeywordExtractorTrainer(model, config)
        
        # Создание тестового батча
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, 128)),
            'attention_mask': torch.ones(batch_size, 128),
            'keyword_labels': torch.randint(0, 2, (batch_size, 128)),
            'trend_labels': torch.rand(batch_size, 128)
        }
        
        # Проверка обучения
        metrics = trainer.train_step(batch)
        assert all(isinstance(v, float) for v in metrics.values())
        
    def test_metrics_history(self, trainer, sample_batch):
        """Тест сохранения истории метрик"""
        # Выполнение нескольких шагов обучения
        for _ in range(3):
            trainer.train_step(sample_batch)
            
        # Проверка истории
        assert len(trainer.history['train_loss']) == 3
        assert len(trainer.history['learning_rates']) == 3
        
    def test_device_handling(self, training_config):
        """Тест обработки устройств"""
        config = KeywordModelConfig()
        model = KeywordExtractorModel(config)
        
        # Создание тренера с явным указанием устройства
        trainer_cpu = KeywordExtractorTrainer(
            model=model,
            config=training_config,
            device='cpu'
        )
        
        assert next(trainer_cpu.model.parameters()).device.type == 'cpu'
        
        # Проверка CUDA если доступно
        if torch.cuda.is_available():
            trainer_cuda = KeywordExtractorTrainer(
                model=model,
                config=training_config,
                device='cuda'
            )
            assert next(trainer_cuda.model.parameters()).device.type == 'cuda'
