import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
import time
from fastapi.testclient import TestClient

# Импорты компонентов DimensionReducer
from models.dim_reducer.model import DimensionReducer
from models.dim_reducer.trainer import DimReducerTrainer
from models.dim_reducer.inference import DimReducerInference
from models.dim_reducer.features import SEOFeaturesExtractor
from models.dim_reducer.data_loader import SEODataLoader
from models.dim_reducer.monitoring import DimReducerMonitor
from common.config.dim_reducer_config import DimReducerConfig
from common.api.routes import app

class TestDimReducerModule:
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Создание тестовых данных"""
        return pd.DataFrame({
            'text': [
                'First test text with SEO optimization keywords',
                'Second text about digital marketing strategies',
                'Third sample with metadata and analytics',
                'Fourth example discussing content marketing',
                'Fifth test case for complete validation'
            ],
            'html': [
                '<html><head><title>Test Title</title></head><body><h1>Header</h1><p>Content</p></body></html>'
                for _ in range(5)
            ]
        })

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
    def model(self, config):
        """Создание модели"""
        return DimensionReducer(config)

    @pytest.fixture
    def client(self):
        """Создание тестового клиента для API"""
        return TestClient(app)

    def test_data_loading(self, sample_data, tmp_path):
        """Тест загрузки данных"""
        # Сохраняем данные во временный файл
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)

        # Создаем загрузчик данных
        data_loader = SEODataLoader(batch_size=2)
        train_dataset, val_dataset, _ = data_loader.create_datasets(
            train_path=str(data_path),
            val_path=str(data_path)
        )

        # Проверяем загрузку
        assert len(train_dataset) == len(sample_data)
        assert len(val_dataset) == len(sample_data)

        # Проверяем батчи
        train_loader, val_loader, _ = data_loader.get_data_loaders(
            train_dataset, val_dataset
        )
        
        for batch in train_loader:
            assert batch.size(0) <= 2  # batch_size
            assert batch.size(1) == 512  # input_dim
            break

    def test_feature_extraction(self, sample_data):
        """Тест извлечения характеристик"""
        extractor = SEOFeaturesExtractor()
        
        # Проверяем обработку одного текста
        features = extractor.extract_all_features(
            sample_data['text'].iloc[0],
            sample_data['html'].iloc[0]
        )

        # Проверяем наличие всех необходимых характеристик
        assert 'word_count' in features
        assert 'keywords' in features
        assert 'tfidf_features' in features
        assert features['word_count'] > 0
        assert len(features['keywords']) > 0

        # Проверяем пакетную обработку
        batch_features = extractor.batch_process(sample_data['text'].tolist())
        assert len(batch_features) == len(sample_data)

    def test_model_training(self, model, config, sample_data, tmp_path):
        """Тест обучения модели"""
        # Создаем тренер
        trainer = DimReducerTrainer(model, config)

        # Подготавливаем данные
        data_path = tmp_path / "train_data.csv"
        sample_data.to_csv(data_path, index=False)

        data_loader = SEODataLoader(batch_size=config.batch_size)
        train_dataset, val_dataset, _ = data_loader.create_datasets(
            train_path=str(data_path),
            val_path=str(data_path)
        )
        train_loader, val_loader, _ = data_loader.get_data_loaders(
            train_dataset, val_dataset
        )

        # Обучение одной эпохи
        metrics = trainer.train_epoch(train_loader, 0, val_loader)
        
        assert 'train_loss' in metrics
        assert 'val_loss' in metrics
        assert metrics['train_loss'] > 0
        assert metrics['val_loss'] > 0

        # Сохранение модели
        model_path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(model_path), 0, metrics)
        assert model_path.exists()

    def test_inference(self, config, model, sample_data, tmp_path):
        """Тест инференса"""
        # Сохраняем модель
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        # Создаем инференс
        inference = DimReducerInference(str(model_path), config)

        # Тестируем обработку одного текста
        results = inference.process_text(sample_data['text'].iloc[0])
        assert 'latent_features' in results
        assert 'feature_importance' in results
        assert results['latent_features'].shape[1] == config.latent_dim

        # Тестируем пакетную обработку
        batch_results = inference.batch_process(
            torch.randn(3, config.input_dim)
        )
        assert len(batch_results['latent_features']) == 3

    def test_monitoring(self, model, config):
        """Тест системы мониторинга"""
        monitor = DimReducerMonitor()

        # Тест отслеживания времени
        start_time = time.time()
        time.sleep(0.1)  # Имитация обработки
        monitor.track_inference(start_time, time.time())

        # Тест отслеживания батча
        monitor.track_batch(32)

        # Тест отслеживания ресурсов
        monitor.track_resources()

        # Получение метрик
        metrics = monitor.get_metrics()
        assert 'avg_inference_time' in metrics
        assert 'requests_per_minute' in metrics
        assert 'cpu_percent' in metrics

    def test_api_endpoints(self, client, sample_data):
        """Тест API эндпоинтов"""
        # Тест анализа текста
        response = client.post(
            "/dim-reducer/analyze",
            json={
                "text": sample_data['text'].iloc[0],
                "html": sample_data['html'].iloc[0],
                "return_importance": True
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert 'latent_features' in result
        assert 'feature_importance' in result

        # Тест пакетной обработки
        response = client.post(
            "/dim-reducer/batch",
            json={
                "texts": sample_data['text'].tolist()[:2],
                "html_texts": sample_data['html'].tolist()[:2]
            }
        )
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2

        # Тест информации о модели
        response = client.get("/dim-reducer/info")
        assert response.status_code == 200
        info = response.json()
        assert 'config' in info
        assert 'device' in info

    def test_error_handling(self, model, config, client):
        """Тест обработки ошибок"""
        # Тест некорректных входных данных в API
        response = client.post(
            "/dim-reducer/analyze",
            json={"text": ""}  # Пустой текст
        )
        assert response.status_code in [400, 422]

        # Тест ошибок модели
        with pytest.raises(Exception):
            model(torch.randn(1, config.input_dim + 1))  # Неверная размерность

        # Тест ошибок загрузчика данных
        data_loader = SEODataLoader(batch_size=2)
        with pytest.raises(Exception):
            data_loader.create_datasets("nonexistent_file.csv")

    def test_full_pipeline(self, sample_data, config, tmp_path):
        """Тест полного пайплайна"""
        # 1. Подготовка данных
        data_path = tmp_path / "pipeline_data.csv"
        sample_data.to_csv(data_path, index=False)

        # 2. Загрузка данных
        data_loader = SEODataLoader(batch_size=config.batch_size)
        train_dataset, val_dataset, _ = data_loader.create_datasets(
            train_path=str(data_path),
            val_path=str(data_path)
        )

        # 3. Обучение модели
        model = DimensionReducer(config)
        trainer = DimReducerTrainer(model, config)
        train_loader, val_loader, _ = data_loader.get_data_loaders(
            train_dataset, val_dataset
        )
        metrics = trainer.train_epoch(train_loader, 0, val_loader)

        # 4. Сохранение модели
        model_path = tmp_path / "pipeline_model.pt"
        trainer.save_checkpoint(str(model_path), 0, metrics)

        # 5. Инференс
        inference = DimReducerInference(str(model_path), config)
        results = inference.process_text(sample_data['text'].iloc[0])

        # 6. Мониторинг
        monitor = DimReducerMonitor()
        monitor.track_inference(time.time() - 0.1, time.time())

        # Проверки
        assert model_path.exists()
        assert metrics['train_loss'] > 0
        assert len(results['latent_features']) > 0
        assert monitor.get_metrics()['requests_per_minute'] >= 0
