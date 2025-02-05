import pytest
import torch
import pandas as pd
from pathlib import Path
import tempfile
import json

from models.dim_reducer.model import DimensionReducer
from models.dim_reducer.trainer import DimReducerTrainer
from models.dim_reducer.inference import DimReducerInference
from models.dim_reducer.data_loader import SEODataLoader
from models.dim_reducer.features import SEOFeaturesExtractor
from common.config.dim_reducer_config import DimReducerConfig

class TestDimReducerPipeline:
    @pytest.fixture(scope="class")
    def test_data(self):
        """Создание тестовых данных"""
        data = pd.DataFrame({
            'text': [
                'First test text for SEO analysis',
                'Second text with different content',
                'Third sample for complete testing'
            ],
            'html': [
                '<html><body><h1>First</h1><p>Content</p></body></html>',
                '<html><body><h1>Second</h1><p>Content</p></body></html>',
                '<html><body><h1>Third</h1><p>Content</p></body></html>'
            ]
        })
        return data

    @pytest.fixture(scope="class")
    def temp_data_path(self, test_data):
        """Сохранение тестовых данных во временный файл"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            return f.name

    @pytest.fixture(scope="class")
    def config(self):
        """Создание конфигурации"""
        return DimReducerConfig(
            input_dim=512,
            hidden_dim=256,
            latent_dim=128,
            batch_size=2,
            num_epochs=2
        )

    @pytest.fixture(scope="class")
    def model_path(self, config, temp_data_path):
        """Обучение и сохранение модели"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DimensionReducer(config)
            trainer = DimReducerTrainer(model, config)
            
            # Подготовка данных
            data_loader = SEODataLoader(
                batch_size=config.batch_size,
                num_workers=0
            )
            train_dataset, val_dataset, _ = data_loader.create_datasets(
                train_path=temp_data_path,
                val_path=temp_data_path
            )
            train_loader, val_loader, _ = data_loader.get_data_loaders(
                train_dataset, val_dataset
            )
            
            # Обучение
            for epoch in range(config.num_epochs):
                trainer.train_epoch(train_loader, epoch, val_loader)
            
            # Сохранение
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)
            return str(model_path)

    def test_full_pipeline(self, config, model_path, temp_data_path):
        """Тест полного пайплайна: обучение → инференс → API"""
        # 1. Загрузка данных
        data_loader = SEODataLoader(batch_size=config.batch_size)
        dataset, _, _ = data_loader.create_datasets(train_path=temp_data_path)
        
        # 2. Инференс
        inference = DimReducerInference(model_path, config)
        sample_text = "Test text for inference"
        results = inference.process_text(sample_text)
        
        assert 'latent_features' in results
        assert 'feature_importance' in results
        assert results['latent_features'].shape[1] == config.latent_dim
        
        # 3. Пакетная обработка
        batch_results = inference.batch_process(
            dataset[0].unsqueeze(0),
            batch_size=1
        )
        assert len(batch_results['latent_features']) == 1
        
        # 4. Проверка сохранения/загрузки
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "new_model.pt"
            torch.save(inference.model.state_dict(), new_path)
            
            # Загружаем новую модель
            new_inference = DimReducerInference(str(new_path), config)
            new_results = new_inference.process_text(sample_text)
            
            # Проверяем идентичность результатов
            torch.testing.assert_close(
                torch.tensor(results['latent_features']),
                torch.tensor(new_results['latent_features'])
            )

    def test_feature_extraction(self, test_data):
        """Тест извлечения признаков"""
        extractor = SEOFeaturesExtractor()
        
        for _, row in test_data.iterrows():
            features = extractor.extract_all_features(row['text'], row['html'])
            
            assert 'word_count' in features
            assert 'sentence_count' in features
            assert features['word_count'] > 0
            assert 'keywords' in features
            assert len(features['keywords']) > 0

    def test_model_training_stability(self, config, temp_data_path):
        """Тест стабильности обучения"""
        model = DimensionReducer(config)
        trainer = DimReducerTrainer(model, config)
        
        data_loader = SEODataLoader(batch_size=config.batch_size)
        train_dataset, _, _ = data_loader.create_datasets(train_path=temp_data_path)
        train_loader, _, _ = data_loader.get_data_loaders(train_dataset)
        
        # Проверяем несколько эпох
        losses = []
        for epoch in range(3):
            metrics = trainer.train_epoch(train_loader, epoch)
            losses.append(metrics['train_loss'])
        
        # Проверяем, что потери уменьшаются
        assert losses[-1] < losses[0]

    def test_error_handling(self, config, model_path):
        """Тест обработки ошибок"""
        inference = DimReducerInference(model_path, config)
        
        # Пустой текст
        with pytest.raises(Exception):
            inference.process_text("")
        
        # Некорректный размер входных данных
        with pytest.raises(Exception):
            inference.reduce_dimensions(
                torch.randn(1, config.input_dim + 1)
            )
            
        # Некорректный batch_size
        with pytest.raises(Exception):
            inference.batch_process(
                torch.randn(1, config.input_dim),
                batch_size=0
            )

    def test_model_persistence(self, config, model_path, test_data):
        """Тест сохранения состояния модели"""
        # Первый инференс
        inference1 = DimReducerInference(model_path, config)
        results1 = inference1.process_text(test_data['text'].iloc[0])
        
        # Второй инференс с той же моделью
        inference2 = DimReducerInference(model_path, config)
        results2 = inference2.process_text(test_data['text'].iloc[0])
        
        # Результаты должны быть идентичны
        torch.testing.assert_close(
            torch.tensor(results1['latent_features']),
            torch.tensor(results2['latent_features'])
        )
