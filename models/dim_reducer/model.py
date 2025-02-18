import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np
from sklearn.metrics import silhouette_score

class DimensionReducer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Optional[list] = None,
                 device: str = "cpu"):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # Определяем архитектуру сети
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
            
        layers = []
        prev_dim = input_dim
        
        # Создаем слои энкодера
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        # Финальный слой для получения сжатого представления
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через модель"""
        return self.encoder(x)
        
    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            num_epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5) -> Dict[str, list]:
        """Обучение модели"""
        optimizer = torch.optim.Adam(self.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
        
        reconstruction_criterion = nn.MSELoss()
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                
                # Прямой проход
                encoded = self.forward(batch_x)
                
                # Вычисляем потери
                loss = reconstruction_criterion(encoded, batch_x)
                
                # Обратное распространение
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            history['loss'].append(avg_loss)
            
        return history
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Преобразование данных в пространство меньшей размерности"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
            
    def save_model(self, path: str):
        """Сохранение модели"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, path)
        
    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> 'DimensionReducer':
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def evaluate(self, data: torch.Tensor) -> Dict[str, float]:
        """Оценка качества редукции размерности"""
        self.eval()
        with torch.no_grad():
            reduced = self.transform(data)
            
            # Считаем метрики
            metrics = {}
            
            # Силуэтный коэффициент
            if reduced.shape[1] >= 2:  # Нужно минимум 2 измерения
                silhouette = silhouette_score(
                    reduced.cpu().numpy(), 
                    np.zeros(reduced.shape[0])  # Фиктивные метки
                )
                metrics['silhouette_score'] = silhouette
            
            # Сохранение дисперсии
            variance_ratio = torch.var(reduced) / torch.var(data)
            metrics['variance_ratio'] = variance_ratio.item()
            
            return metrics
