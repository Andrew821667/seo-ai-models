# tests/test_utils.py

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from model.utils.data_utils import (
    KeywordDataset,
    DataCollator,
    load_dataset,
    create_dataloaders
)
from model.utils.metrics import KeywordMetrics, KeywordEvaluator
from model.utils.visualization import (
    TrainingVisualizer,
    KeywordVisualizer
)
from model.utils.analysis import ErrorAnalyzer, PerformanceAnalyzer
from model.config.model_config import KeywordModelConfig
from model.model import KeywordExtractorModel

@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
    return {
        'texts': [
            "This is a sample text for testing.",
            "Another example text with keywords."
        ],
        'keyword_labels': [
            [0, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0]
        ],
        'trend_labels': [
            [0.1, 0.5, 0.2, 0.8, 0.3, 0.1],
            [0.2, 0.7, 0.6, 0.3, 0.9, 0.2]
        ]
    }

@pytest.fixture
def processor():
    """Фикстура с процессором"""
    config = KeywordModelConfig()
    model = KeywordExtractorModel(config)
    return model.processor

class TestDataUtils:
    """Тесты для утилит работы с данными"""
    
    def test_dataset_creation(self, sample_data, processor):
        """Тест создания датасета"""
        dataset = KeywordDataset(
            texts=sample_data['texts'],
            keyword_labels=sample_data['keyword_labels'],
            trend_labels=sample_data['trend_labels'],
            processor=processor
        )
        
        assert len(dataset) == len(sample_data['texts'])
        
        # Проверка первого элемента
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'keyword_labels' in item
        assert 'trend_labels' in item
        
    def test_data_collator(self, sample_data, processor):
        """Тест коллатора данных"""
        dataset = KeywordDataset(
            texts=sample_data['texts'],
            keyword_labels=sample_data['keyword_labels'],
            trend_labels=sample_data['trend_labels'],
            processor=processor
        )
        
        collator = DataCollator()
        batch = collator([dataset[0], dataset[1]])
        
        assert isinstance(batch, dict)
        assert all(isinstance(v, torch.Tensor) for v in batch.values())
        
    def test_dataloader_creation(self, sample_data, processor):
        """Тест создания загрузчиков данных"""
        dataset = KeywordDataset(
            texts=sample_data['texts'],
            keyword_labels=sample_data['keyword_labels'],
            trend_labels=sample_data['trend_labels'],
            processor=processor
        )
        
        train_loader, val_loader = create_dataloaders(
            dataset,
            dataset,
            batch_size=2
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0

class TestMetrics:
    """Тесты для метрик"""
    
    def test_keyword_metrics(self):
        """Тест расчета метрик ключевых слов"""
        metrics = KeywordMetrics()
        
        predictions = torch.tensor([[0.8, 0.2, 0.9], [0.3, 0.7, 0.4]])
        targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
        
        results = metrics.calculate_metrics(predictions, targets)
        
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert all(0 <= v <= 1 for v in results.values())
        
    def test_evaluator(self):
        """Тест оценщика"""
        evaluator = KeywordEvaluator()
        
        outputs = {
            'keyword_logits': torch.randn(2, 3, 2),
            'trend_scores': torch.sigmoid(torch.randn(2, 3))
        }
        
        targets = {
            'keyword_labels': torch.randint(0, 2, (2, 3)),
            'trend_labels': torch.rand(2, 3),
            'attention_mask': torch.ones(2, 3)
        }
        
        metrics = evaluator.evaluate_batch(outputs, targets)
        assert isinstance(metrics, dict)

class TestVisualization:
    """Тесты для визуализации"""
    
    def test_training_visualizer(self, tmp_path):
        """Тест визуализатора обучения"""
        visualizer = TrainingVisualizer(save_dir=tmp_path)
        
        history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'learning_rates': [0.001, 0.0008, 0.0005]
        }
        
        visualizer.plot_training_history(history, 'test.png')
        assert (tmp_path / 'test.png').exists()
        
    def test_keyword_visualizer(self, tmp_path):
        """Тест визуализатора ключевых слов"""
        visualizer = KeywordVisualizer(save_dir=tmp_path)
        
        keywords = [
            {'keyword': 'test', 'score': 0.8},
            {'keyword': 'example', 'score': 0.6}
        ]
        
        visualizer.plot_keyword_distribution(keywords, 'test.png')
        assert (tmp_path / 'test.png').exists()

class TestAnalysis:
    """Тесты для анализа"""
    
    def test_error_analyzer(self):
        """Тест анализатора ошибок"""
        analyzer = ErrorAnalyzer()
        
        predictions = [
            {'keyword': 'test', 'score': 0.8},
            {'keyword': 'example', 'score': 0.6}
        ]
        targets = ['test', 'sample']
        texts = ['test text', 'example text']
        
        analysis = analyzer.analyze_predictions(
            predictions,
            targets,
            texts
        )
        
        assert 'error_types' in analysis
        assert 'length_analysis' in analysis
        
    def test_performance_analyzer(self):
        """Тест анализатора производительности"""
        analyzer = PerformanceAnalyzer()
        
        outputs = [
            {
                'predictions': np.random.rand(10),
                'confidence': np.random.rand(10)
            }
        ]
        
        targets = [
            {
                'labels': np.random.randint(0, 2, 10),
                'weights': np.ones(10)
            }
        ]
        
        performance = analyzer.analyze_model_performance(
            outputs,
            targets
        )
        
        assert 'overall_metrics' in performance
        assert 'per_length_metrics' in performance
