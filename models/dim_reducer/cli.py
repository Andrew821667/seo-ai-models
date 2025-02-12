import click
import logging
from pathlib import Path
import torch
import yaml
import json
from typing import Optional
import sys

from common.config.dim_reducer_config import DimReducerConfig
from .model import DimensionReducer
from .trainer import DimReducerTrainer
from .data_loader import SEODataLoader
from .inference import DimReducerInference
from common.utils.visualization import VisualizationManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """CLI утилита для работы с DimensionReducer"""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('train_data', type=click.Path(exists=True))
@click.option('--val-data', type=click.Path(exists=True), help='Путь к данным для валидации')
@click.option('--output-dir', type=click.Path(), default='models/dim_reducer/checkpoints')
@click.option('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
def train(
    config_path: str,
    train_data: str,
    val_data: Optional[str],
    output_dir: str,
    device: str
):
    """Обучение модели"""
    try:
        # Загрузка конфигурации
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = DimReducerConfig(**config_dict)
        
        # Инициализация модели
        model = DimensionReducer(config)
        trainer = DimReducerTrainer(model, config, device=device)
        
        # Подготовка данных
        data_loader = SEODataLoader(
            batch_size=config.batch_size,
            num_workers=4,
            max_features=config.max_features
        )
        
        train_dataset, val_dataset, _ = data_loader.create_datasets(
            train_path=train_data,
            val_path=val_data
        )
        
        train_loader, val_loader, _ = data_loader.get_data_loaders(
            train_dataset,
            val_dataset
        )
        
        # Создание директории для сохранения
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Обучение
        logger.info("Starting training...")
        for epoch in range(config.num_epochs):
            metrics = trainer.train_epoch(train_loader, epoch, val_loader)
            
            # Логирование метрик
            logger.info(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Train Loss: {metrics['train_loss']:.4f}"
                + (f" - Val Loss: {metrics['val_loss']:.4f}" if 'val_loss' in metrics else "")
            )
            
            # Сохранение чекпоинта
            if (epoch + 1) % 5 == 0:
                checkpoint_path = output_path / f"checkpoint_epoch_{epoch + 1}.pt"
                trainer.save_checkpoint(str(checkpoint_path), epoch, metrics)
        
        # Сохранение финальной модели
        final_model_path = output_path / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Training completed. Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('input_data', type=click.Path(exists=True))
@click.option('--config-path', type=click.Path(exists=True), help='Путь к конфигурации')
@click.option('--output-dir', type=click.Path(), default='output')
@click.option('--batch-size', type=int, default=32)
@click.option('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
def predict(
    model_path: str,
    input_data: str,
    config_path: Optional[str],
    output_dir: str,
    batch_size: int,
    device: str
):
    """Применение обученной модели"""
    try:
        # Загрузка конфигурации
        if config_path:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            config = DimReducerConfig(**config_dict)
        else:
            config = DimReducerConfig()
        
        # Инициализация инференса
        inference = DimReducerInference(model_path, config, device)
        
        # Подготовка данных
        data_loader = SEODataLoader(
            batch_size=batch_size,
            num_workers=4,
            max_features=config.max_features
        )
        
        dataset, _, _ = data_loader.create_datasets(train_path=input_data)
        
        # Создание директории для результатов
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Обработка данных
        logger.info("Processing data...")
        all_results = []
        
        for batch in dataset:
            results = inference.reduce_dimensions(batch)
            all_results.append(results)
        
        # Сохранение результатов
        results_path = output_path / "predictions.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        sys.exit(1)

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('input_text')
@click.option('--config-path', type=click.Path(exists=True), help='Путь к конфигурации')
@click.option('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
def analyze_text(
    model_path: str,
    input_text: str,
    config_path: Optional[str],
    device: str
):
    """Анализ отдельного текста"""
    try:
        # Загрузка конфигурации
        if config_path:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            config = DimReducerConfig(**config_dict)
        else:
            config = DimReducerConfig()
        
        # Инициализация инференса
        inference = DimReducerInference(model_path, config, device)
        
        # Обработка текста
        results = inference.process_text(input_text)
        
        # Вывод результатов
        print("\nResults:")
        print(f"Latent dimensions: {results['latent_features'].shape}")
        print(f"Feature importance: {results['feature_importance'].mean():.4f}")
        
        # Визуализация
        viz = VisualizationManager()
        viz.plot_feature_importance(results['feature_importance'])
        
    except Exception as e:
        logger.error(f"Error during text analysis: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()
