from typing import Dict, Any, Optional, List, Union
import torch
from torch import nn
import numpy as np
import os
from tqdm import tqdm
from common.monitoring.metrics import MetricsTracker

class DimReducerTrainer:
    def __init__(self, 
                 input_dim: int = 768,
                 output_dim: int = 128,
                 learning_rate: float = 1e-3,
                 device: Optional[str] = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = MetricsTracker()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        ).to(self.device)
        
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, input_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=learning_rate
        )
        self.criterion = nn.MSELoss()
        
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        
    def _process_batch(self, batch: Union[torch.Tensor, List, tuple]) -> torch.Tensor:
        """Обработка батча из DataLoader"""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # Берем первый элемент, так как у нас TensorDataset
        return batch.to(self.device)
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        self.encoder.train()
        self.decoder.train()
        epoch_losses = []
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch in pbar:
                batch = self._process_batch(batch)
                loss = self.train_step(batch)['loss']
                epoch_losses.append(loss)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
        avg_train_loss = np.mean(epoch_losses)
        self.train_history.append(avg_train_loss)
        metrics = {'train_loss': avg_train_loss}
        
        if val_loader is not None:
            val_loss = self.evaluate_loader(val_loader)['eval_loss']
            self.val_history.append(val_loss)
            metrics['val_loss'] = val_loss
            
        return metrics
    
    def evaluate_loader(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Оценка на всем датасете"""
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = self._process_batch(batch)
                encoded = self.encoder(batch)
                decoded = self.decoder(encoded)
                loss = self.criterion(decoded, batch)
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        self.metrics.add_metric('eval_loss', avg_loss)
        return {'eval_loss': avg_loss}
        
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Один шаг обучения"""
        batch = self._process_batch(batch)
        self.optimizer.zero_grad()
        
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        
        loss = self.criterion(decoded, batch)
        loss.backward()
        self.optimizer.step()
        
        self.metrics.add_metric('train_loss', loss.item())
        return {'loss': loss.item()}
    
    def evaluate(self, data: torch.Tensor) -> Dict[str, float]:
        """Оценка на тензоре данных"""
        data = self._process_batch(data)
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            encoded = self.encoder(data)
            decoded = self.decoder(encoded)
            loss = self.criterion(decoded, data)
            
        self.metrics.add_metric('eval_loss', loss.item())
        return {'eval_loss': loss.item()}
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Кодирование данных"""
        data = self._process_batch(data)
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(data).cpu()
            
    def save_model(self, path: str) -> None:
        """Сохранение модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'device': self.device,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, path)
        
    def load_model(self, path: str) -> None:
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.input_dim = checkpoint['input_dim']
        self.output_dim = checkpoint['output_dim']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
