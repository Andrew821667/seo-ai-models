# model/trainer.py

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime

from .config.model_config import KeywordModelConfig, KeywordTrainingConfig
from .config.logging_config import get_logger
from .model import KeywordExtractorModel

logger = get_logger(__name__)

class EarlyStopping:
    """Механизм ранней остановки обучения"""
    
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop

class KeywordExtractorTrainer:
    """Класс для обучения модели извлечения ключевых слов"""
    
    def __init__(
        self,
        model: KeywordExtractorModel,
        config: KeywordTrainingConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Инициализация тренера
        
        Args:
            model: Модель для обучения
            config: Конфигурация процесса обучения
            device: Устройство для обучения
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Планировщик скорости обучения
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=1  # Будет обновлено при обучении
        )
        
        # Функции потерь
        self.keyword_criterion = nn.CrossEntropyLoss()
        self.trend_criterion = nn.BCELoss()
        
        # Ранняя остановка
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience
        )
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': []
        }
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Один шаг обучения
        
        Args:
            batch: Батч данных
            
        Returns:
            Словарь с метриками
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # Перемещение данных на устройство
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            keyword_labels = batch['keyword_labels'].to(self.device)
            trend_labels = batch['trend_labels'].to(self.device)
            
            # Прямой проход
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Расчет потерь для ключевых слов
            keyword_loss = self.keyword_criterion(
                outputs['keyword_logits'].view(-1, 2),
                keyword_labels.view(-1)
            )
            
            # Расчет потерь для трендов
            trend_loss = self.trend_criterion(
                outputs['trend_scores'].view(-1),
                trend_labels.view(-1)
            )
            
            # Общие потери
            total_loss = keyword_loss + 0.5 * trend_loss
            
            # Обратное распространение
            total_loss.backward()
            
            # Ограничение градиентов
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Оптимизация
            self.optimizer.step()
            self.scheduler.step()
            
            return {
                'total_loss': total_loss.item(),
                'keyword_loss': keyword_loss.item(),
                'trend_loss': trend_loss.item()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении шага обучения: {e}")
            raise
            
    def validate(
        self,
        val_dataloader
    ) -> Dict[str, float]:
        """
        Валидация модели
        
        Args:
            val_dataloader: Загрузчик данных для валидации
            
        Returns:
            Словарь с метриками валидации
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    # Перемещение данных на устройство
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    keyword_labels = batch['keyword_labels'].to(self.device)
                    trend_labels = batch['trend_labels'].to(self.device)
                    
                    # Прямой проход
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Расчет потерь
                    keyword_loss = self.keyword_criterion(
                        outputs['keyword_logits'].view(-1, 2),
                        keyword_labels.view(-1)
                    )
                    
                    trend_loss = self.trend_criterion(
                        outputs['trend_scores'].view(-1),
                        trend_labels.view(-1)
                    )
                    
                    total_loss += (keyword_loss + 0.5 * trend_loss).item()
                    
                    # Расчет точности
                    predictions = torch.argmax(
                        outputs['keyword_logits'],
                        dim=-1
                    )
                    total_correct += (predictions == keyword_labels).sum().item()
                    total_samples += keyword_labels.numel()
                    
                except Exception as e:
                    logger.error(f"Ошибка при валидации батча: {e}")
                    continue
                    
        return {
            'val_loss': total_loss / len(val_dataloader),
            'val_accuracy': total_correct / total_samples
        }
        
    def train(
        self,
        train_dataloader,
        val_dataloader,
        save_dir: Optional[Path] = None
    ):
        """
        Обучение модели
        
        Args:
            train_dataloader: Загрузчик обучающих данных
            val_dataloader: Загрузчик данных для валидации
            save_dir: Директория для сохранения чекпоинтов
        """
        try:
            # Обновление количества шагов в планировщике
            self.scheduler.steps_per_epoch = len(train_dataloader)
            
            for epoch in range(self.config.num_epochs):
                # Обучение эпохи
                train_metrics = self._train_epoch(
                    train_dataloader,
                    epoch
                )
                
                # Валидация
                val_metrics = self.validate(val_dataloader)
                
                # Логирование результатов
                self._log_epoch_results(
                    epoch,
                    train_metrics,
                    val_metrics
                )
                
                # Сохранение чекпоинта
                if save_dir:
                    self.save_checkpoint(
                        save_dir / f'checkpoint_epoch_{epoch}.pt',
                        epoch,
                        val_metrics
                    )
                
                # Проверка ранней остановки
                if self.early_stopping(val_metrics['val_loss']):
                    logger.info("Применена ранняя остановка обучения")
                    break
                    
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise
            
    def _train_epoch(
        self,
        train_dataloader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Обучение одной эпохи
        
        Args:
            train_dataloader: Загрузчик обучающих данных
            epoch: Номер эпохи
            
        Returns:
            Словарь с метриками эпохи
        """
        total_loss = 0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f'Эпоха {epoch + 1}/{self.config.num_epochs}'
        )
        
        for batch in progress_bar:
            metrics = self.train_step(batch)
            total_loss += metrics['total_loss']
            
            # Обновление прогресс-бара
            progress_bar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
    def _log_epoch_results(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """
        Логирование результатов эпохи
        
        Args:
            epoch: Номер эпохи
            train_metrics: Метрики обучения
            val_metrics: Метрики валидации
        """
        # Обновление истории
        self.history['train_loss'].append(train_metrics['train_loss'])
        self.history['val_loss'].append(val_metrics['val_loss'])
        self.history['learning_rates'].append(train_metrics['learning_rate'])
        self.history['metrics'].append({**train_metrics, **val_metrics})
        
        # Вывод в лог
        logger.info(
            f"Эпоха {epoch + 1}: "
            f"train_loss={train_metrics['train_loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"val_accuracy={val_metrics['val_accuracy']:.4f}"
        )
        
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Сохранение чекпоинта
        
        Args:
            path: Путь для сохранения
            epoch: Номер эпохи
            metrics: Метрики модели
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Сохранен чекпоинт: {path}")
        
    def load_checkpoint(
        self,
        path: Path
    ) -> Tuple[int, Dict[str, float]]:
        """
        Загрузка чекпоинта
        
        Args:
            path: Путь к чекпоинту
            
        Returns:
            Номер эпохи и метрики
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        return checkpoint['epoch'], checkpoint['metrics']
