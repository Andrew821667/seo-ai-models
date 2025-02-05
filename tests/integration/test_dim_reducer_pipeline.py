import pytest
import torch
import numpy as np
from pathlib import Path

from models.dim_reducer.model import DimensionReducer
from models.dim_reducer.trainer import DimReducerTrainer
from models.dim_reducer.inference import DimReducerInference
from common.config.dim_reducer_config import DimReducerConfig
from common.utils.preprocessing import TextPreprocessor

@pytest.mark.integration
class TestDimReducerPipeline:
    @pytest.fixture(scope="class")
    def config(self):
        return DimReducerConfig(
            input_dim=10,
            hidden_dim=8,
            latent_dim=4,
            batch_size=2,
            num_epochs=2
        )
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        # Генерация тестовых данных
        X = torch.randn(100, 10)  # 100 samples, 10 features
        return X
    
    @pytest.fixture(scope="class")
    def trained_model_path(self, config, sample_data, tmp_path_factory):
        # Обучение модели и сохранение
        model = DimensionReducer(config)
        trainer = DimReducerTrainer(model, config)
        
        # Создаем простой даталоадер
        train_data = torch.utils.data.TensorDataset(sample_data)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        # Обучаем модель
        for epoch in range(config.num_epochs):
            trainer.train_epoch(train_loader, epoch)
        
        # Сохраняем модель
        path = tmp_path_factory.mktemp("models") / "trained_model.pt"
        torch.save(model.state_dict(), path)
        return str(path)
    
    def test_full_pipeline(self, config, sample_data, trained_model_path):
        """Тест полного пайплайна: обучение → инференс → анализ"""
        # Инициализация инференса
        inference = DimReducerInference(trained_model_path, config)
        
        # Проверяем сжатие размерности
        results = inference.reduce_dimensions(sample_data[:10])  # Берем первые 10 семплов
        assert 'latent_features' in results
        assert results['latent_features'].shape == (10, config.latent_dim)
        
        # Проверяем реконструкцию
        assert 'reconstructed' in results
        assert results['reconstructed'].shape == (10, config.input_dim)
        
        # Проверяем анализ важности признаков
        importance_results = inference.analyze_feature_importance(sample_data[:10])
        assert 'mean_importance' in importance_results
        assert 'feature_ranking' in importance_results
        
        # Проверяем качество реконструкции
        reconstruction_error = np.mean(
            (results['reconstructed'] - sample_data[:10].numpy()) ** 2
        )
        assert reconstruction_error < 1.0  # Пороговое значение для MSE
    
    def test_text_processing_pipeline(self, config, trained_model_path):
        """Тест пайплайна обработки текста"""
        inference = DimReducerInference(trained_model_path, config)
        
        # Тестовые тексты
        texts = [
            "First sample text for testing",
            "Second example with different content",
            "Third test sample with variations"
        ]
        
        # Обработка текстов
        results = inference.process_text(texts)
        
        assert 'latent_features' in results
        assert 'reconstructed' in results
        assert results['num_samples'] == len(texts)
        
        # Проверяем пакетную обработку
        batch_results = inference.batch_process(
            torch.randn(50, config.input_dim),
            batch_size=16
        )
        assert len(batch_results['latent_features']) == 50
    
    def test_model_persistence(self, config, trained_model_path, sample_data):
        """Тест сохранения и загрузки модели"""
        # Создаем два инстанса инференса с одной и той же моделью
        inference1 = DimReducerInference(trained_model_path, config)
        inference2 = DimReducerInference(trained_model_path, config)
        
        # Получаем предсказания
        results1 = inference1.reduce_dimensions(sample_data[:5])
        results2 = inference2.reduce_dimensions(sample_data[:5])
        
        # Проверяем идентичность результатов
        np.testing.assert_array_almost_equal(
            results1['latent_features'],
            results2['latent_features']
        )
    
    def test_error_handling_pipeline(self, config, trained_model_path):
        """Тест обработки ошибок в пайплайне"""
        inference = DimReducerInference(trained_model_path, config)
        
        # Проверяем обработку неверных входных данных
        with pytest.raises(Exception):
            inference.reduce_dimensions(torch.randn(10, config.input_dim + 1))
        
        with pytest.raises(Exception):
            inference.process_text([])
        
        with pytest.raises(Exception):
            inference.batch_process(torch.randn(10, config.input_dim), batch_size=0)
