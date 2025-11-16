import pytest
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from seo_ai_models.models.dim_reducer.trainer import DimReducerTrainer

class TestDimReducer:
    @pytest.fixture
    def trainer(self):
        return DimReducerTrainer(input_dim=10, output_dim=5)
    
    @pytest.fixture
    def sample_data(self):
        return torch.randn(32, 10)
    
    @pytest.fixture
    def data_loader(self, sample_data):
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=8)
    
    @pytest.fixture
    def model_path(self, tmp_path):
        return os.path.join(tmp_path, "model.pt")
    
    def test_train_epoch(self, trainer, data_loader):
        metrics = trainer.train_epoch(data_loader)
        assert 'train_loss' in metrics
        assert len(trainer.train_history) == 1
        
    def test_train_epoch_with_validation(self, trainer, data_loader):
        metrics = trainer.train_epoch(data_loader, data_loader)  # Используем тот же загрузчик для валидации
        assert 'train_loss' in metrics
        assert 'val_loss' in metrics
        assert len(trainer.train_history) == 1
        assert len(trainer.val_history) == 1
        
    def test_evaluate_loader(self, trainer, data_loader):
        metrics = trainer.evaluate_loader(data_loader)
        assert 'eval_loss' in metrics
        
    # ... остальные тесты остаются без изменений ...
