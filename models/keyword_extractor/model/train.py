# model/train.py

import torch
from pathlib import Path
import logging
from typing import Optional, Union
import json
from datetime import datetime

from .config.model_config import KeywordModelConfig, KeywordTrainingConfig
from .config.logging_config import get_logger
from .model.model import KeywordExtractorModel
from .model.trainer import KeywordExtractorTrainer
from .monitoring.logger import KeywordExtractorLogger
from .monitoring.performance import PerformanceMonitor
from .utils.data_utils import load_dataset, create_dataloaders
from .utils.visualization import TrainingVisualizer

logger = get_logger(__name__)

def train_model(
    model_config: KeywordModelConfig,
    training_config: KeywordTrainingConfig,
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = 'cuda',
    resume_from: Optional[str] = None
) -> None:
    """
    Обучение модели
    
    Args:
        model_config: Конфигурация модели
        training_config: Конфигурация обучения
        data_dir: Директория с данными
        output_dir: Директория для результатов
        device: Устройство для обучения
        resume_from: Путь к чекпоинту для продолжения обучения
    """
    try:
        # Подготовка директорий
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoints_dir = output_dir / 'checkpoints'
        logs_dir = output_dir / 'logs'
        visualizations_dir = output_dir / 'visualizations'
        
        for dir_path in [checkpoints_dir, logs_dir, visualizations_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Настройка логирования
        file_logger = KeywordExtractorLogger(
            name="train",
            log_dir=logs_dir
        )
        logger = file_logger.get_logger()
        
        # Инициализация мониторинга
        monitor = PerformanceMonitor(
            save_dir=logs_dir
        )
        
        # Инициализация визуализатора
        visualizer = TrainingVisualizer(
            save_dir=visualizations_dir
        )
        
        # Загрузка данных
        logger.info("Загрузка данных...")
        model = KeywordExtractorModel(model_config)
        train_dataset, val_dataset = load_dataset(
            data_dir,
            model.processor,
            validation_split=training_config.validation_split
        )
        
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=training_config.batch_size
        )
        
        # Перемещение модели на устройство
        model = model.to(device)
        
        # Инициализация тренера
        trainer = KeywordExtractorTrainer(
            model=model,
            config=training_config,
            device=device
        )
        
        # Загрузка чекпоинта если указан
        start_epoch = 0
        if resume_from:
            logger.info(f"Загрузка чекпоинта: {resume_from}")
            start_epoch, _ = trainer.load_checkpoint(resume_from)
            
        # Обучение
        logger.info("Начало обучения...")
        for epoch in range(start_epoch, training_config.num_epochs):
            # Замер времени эпохи
            epoch_start = monitor.start_batch()
            
            # Обучение эпохи
            train_metrics = trainer.train_epoch(
                train_loader,
                epoch=epoch,
                val_dataloader=val_loader
            )
            
            # Замер производительности
            performance_metrics = monitor.end_batch(
                epoch_start,
                len(train_loader.dataset)
            )
            
            # Логирование метрик
            logger.info(
                f"Эпоха {epoch + 1}/{training_config.num_epochs} - "
                f"train_loss: {train_metrics['train_loss']:.4f}, "
                f"val_loss: {train_metrics.get('val_loss', 'N/A')}"
            )
            
            # Сохранение чекпоинта
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch + 1, train_metrics)
            
            # Обновление визуализаций
            if (epoch + 1) % 5 == 0:
                visualizer.plot_training_history(
                    trainer.history,
                    f"training_history_epoch_{epoch + 1}.png"
                )
                
        # Сохранение финальной модели
        final_model_path = output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Модель сохранена в {final_model_path}")
        
        # Сохранение конфигурации
        config = {
            'model': model_config.dict(),
            'training': training_config.dict()
        }
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Сохранение метрик производительности
        monitor.export_performance_report(
            output_dir / "performance_report.json"
        )
        
        logger.info("Обучение завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        raise

if __name__ == '__main__':
    # Пример запуска
    model_config = KeywordModelConfig()
    training_config = KeywordTrainingConfig()
    
    train_model(
        model_config=model_config,
        training_config=training_config,
        data_dir='data/train',
        output_dir='outputs',
        device='cuda'
    )
