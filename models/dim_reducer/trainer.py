import torch
import torch.nn as nn
from typing import Dict, Optional, List
from tqdm import tqdm
import logging
from pathlib import Path

from common.config.dim_reducer_config import DimReducerConfig
from .model import DimensionReducer
from common.monitoring.metrics import MetricsTracker
from common.utils.visualization import VisualizationManager

logger = logging.getLogger(__name__)

class DimReducerTrainer:
    """Класс для обучения модели DimensionReducer"""
    
    def __init__(
        self,
        model: DimensionReducer,
        config: DimReducerConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Функции потерь
        self.reconstruction_loss = nn.MSELoss()
        
        # Мониторинг и визуализация
        self.metrics = MetricsTracker()
        self.visualizer = VisualizationManager()
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'feature_importance': []
        }
        
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Один шаг обучения"""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        outputs = self.model(batch)
        
        # Расчет различных компонентов потерь
        rec_loss = self.reconstruction_loss(outputs['reconstructed'], batch)
        sparsity_loss = torch.mean(torch.abs(outputs['latent']))
        importance_reg = torch.mean(torch.abs(outputs['importance']))
        
        # Общие потери
        total_loss = rec_loss + 0.1 * sparsity_loss + 0.05 * importance_reg
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': rec_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'importance_reg': importance_reg.item()
        }
        
    def train_epoch(
        self,
        train_loader,
        epoch: int,
        val_loader = None
    ) -> Dict[str, float]:
        """Обучение одной эпохи"""
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{self.config.num_epochs}'
        )
        
        for batch in progress_bar:
            metrics = self.train_step(batch)
            total_loss += metrics['total_loss']
            
            # Обновление прогресс-бара
            progress_bar.set_postfix(
                loss=f"{metrics['total_loss']:.4f}",
                rec_loss=f"{metrics['reconstruction_loss']:.4f}"
            )
            
        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        
        # Валидация
        if val_loader:
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['val_loss'])
            
            return {
                'train_loss': avg_loss,
                **val_metrics
            }
            
        return {'train_loss': avg_loss}
        
    def validate(self, val_loader) -> Dict[str, float]:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.reconstruction_loss(outputs['reconstructed'], batch)
                total_loss += loss.item()
                
        return {'val_loss': total_loss / len(val_loader)}
        
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Сохранение чекпоинта"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
    def load_checkpoint(self, path: str):
        """Загрузка чекпоинта"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        return checkpoint['epoch'], checkpoint['metrics']
