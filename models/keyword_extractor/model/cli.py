# model/cli.py

import click
import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from .config.model_config import KeywordModelConfig, KeywordTrainingConfig
from .config.logging_config import get_logger
from .model.model import KeywordExtractorModel
from .monitoring.logger import KeywordExtractorLogger

logger = get_logger(__name__)

@click.group()
def cli():
    """Инструмент командной строки для модели извлечения ключевых слов"""
    pass

@cli.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Путь к файлу конфигурации модели'
)
@click.option(
    '--data-dir',
    type=click.Path(exists=True),
    required=True,
    help='Директория с данными для обучения'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    required=True,
    help='Директория для сохранения результатов'
)
@click.option(
    '--epochs',
    type=int,
    default=10,
    help='Количество эпох обучения'
)
@click.option(
    '--batch-size',
    type=int,
    default=32,
    help='Размер батча'
)
@click.option(
    '--learning-rate',
    type=float,
    default=1e-4,
    help='Скорость обучения'
)
@click.option(
    '--device',
    type=str,
    default='cuda',
    help='Устройство для обучения (cuda/cpu)'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Уровень логирования'
)
def train(
    config: Optional[str],
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    log_level: str
):
    """Обучение модели"""
    try:
        # Настройка логирования
        log_dir = Path(output_dir) / 'logs'
        logger_inst = KeywordExtractorLogger(
            log_level=log_level,
            log_dir=log_dir
        )
        logger = logger_inst.get_logger()
        
        # Загрузка конфигурации
        if config:
            logger.info(f"Загрузка конфигурации из {config}")
            with open(config) as f:
                config_data = json.load(f)
            model_config = KeywordModelConfig(**config_data['model'])
            training_config = KeywordTrainingConfig(**config_data['training'])
        else:
            logger.info("Использование конфигурации по умолчанию")
            model_config = KeywordModelConfig()
            training_config = KeywordTrainingConfig()
            
        # Обновление параметров обучения
        training_config.num_epochs = epochs
        training_config.batch_size = batch_size
        training_config.learning_rate = learning_rate
        
        logger.info("Запуск обучения...")
        # Импорт здесь для избежания циклических зависимостей
        from .train import train_model
        train_model(
            model_config=model_config,
            training_config=training_config,
            data_dir=data_dir,
            output_dir=output_dir,
            device=device
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option(
    '--model-path',
    type=click.Path(exists=True),
    required=True,
    help='Путь к сохраненной модели'
)
@click.option(
    '--input',
    type=click.Path(exists=True),
    required=True,
    help='Путь к входному файлу или директории'
)
@click.option(
    '--output',
    type=click.Path(),
    required=True,
    help='Путь для сохранения результатов'
)
@click.option(
    '--batch-size',
    type=int,
    default=32,
    help='Размер батча'
)
@click.option(
    '--device',
    type=str,
    default='cuda',
    help='Устройство для предсказаний (cuda/cpu)'
)
@click.option(
    '--threshold',
    type=float,
    default=0.5,
    help='Порог уверенности для ключевых слов'
)
def predict(
    model_path: str,
    input: str,
    output: str,
    batch_size: int,
    device: str,
    threshold: float
):
    """Извлечение ключевых слов из текста"""
    try:
        # Импорт здесь для избежания циклических зависимостей
        from .predict import run_predictions
        run_predictions(
            model_path=model_path,
            input_path=input,
            output_path=output,
            batch_size=batch_size,
            device=device,
            threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option(
    '--model-path',
    type=click.Path(exists=True),
    required=True,
    help='Путь к сохраненной модели'
)
@click.option(
    '--test-data',
    type=click.Path(exists=True),
    required=True,
    help='Путь к тестовым данным'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    required=True,
    help='Директория для сохранения результатов'
)
@click.option(
    '--batch-size',
    type=int,
    default=32,
    help='Размер батча'
)
def evaluate(
    model_path: str,
    test_data: str,
    output_dir: str,
    batch_size: int
):
    """Оценка качества модели"""
    try:
        # Импорт здесь для избежания циклических зависимостей
        from .evaluate import evaluate_model
        results = evaluate_model(
            model_path=model_path,
            test_data_path=test_data,
            batch_size=batch_size
        )
        
        # Сохранение результатов
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Результаты оценки сохранены в {results_file}")
        
    except Exception as e:
        logger.error(f"Ошибка при оценке модели: {e}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()
