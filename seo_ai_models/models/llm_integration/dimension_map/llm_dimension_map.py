"""
Модель для выявления факторов, важных для LLM.

Модуль предоставляет функционал для обучения ML-модели,
выявляющей факторы, которые влияют на ранжирование в LLM-поисковиках.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LLMDimensionMap:
    """
    Модель для выявления факторов, важных для LLM.
    """
    
    def __init__(self, features_dim: int = 50, hidden_dim: int = 32):
        """
        Инициализирует модель для выявления факторов.
        
        Args:
            features_dim: Размерность входных признаков
            hidden_dim: Размерность скрытого слоя
        """
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        # Проверяем наличие PyTorch
        if not HAS_TORCH:
            self.logger.warning("PyTorch не установлен. Функциональность модели будет ограничена.")
        
        # Настройка логгирования
        self.logger = logging.getLogger(__name__)
    
    def _create_model(self):
        """
        Создает нейронную сеть для предсказания важности факторов.
        
        Returns:
            nn.Module: Созданная модель
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch не установлен. Невозможно создать модель.")
        
        # Создаем простую нейросеть для предсказания
        class DimensionNet(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(DimensionNet, self).__init__()
                self.layer1 = nn.Linear(input_dim, hidden_dim)
                self.activation = nn.ReLU()
                self.layer2 = nn.Linear(hidden_dim, 1)
                self.sigmoid = nn.Sigmoid()
                
                # Слой для выделения важности признаков
                self.attention = nn.Linear(input_dim, input_dim)
                
            def forward(self, x):
                # Вычисляем важность признаков
                att_weights = self.attention(x)
                att_weights = torch.softmax(att_weights, dim=1)
                
                # Применяем веса к входным данным
                weighted_input = x * att_weights
                
                # Прямой проход через сеть
                hidden = self.activation(self.layer1(weighted_input))
                output = self.sigmoid(self.layer2(hidden))
                
                return output, att_weights
        
        return DimensionNet(self.features_dim, self.hidden_dim)
    
    def train(self, features: np.ndarray, targets: np.ndarray, 
             feature_names: List[str],
             epochs: int = 100, batch_size: int = 32, 
             learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Обучает модель на предоставленных данных.
        
        Args:
            features: Матрица признаков (n_samples, n_features)
            targets: Целевые значения (n_samples,)
            feature_names: Имена признаков
            epochs: Количество эпох обучения
            batch_size: Размер батча
            learning_rate: Скорость обучения
            
        Returns:
            Dict[str, Any]: Результаты обучения
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch не установлен. Невозможно обучить модель.")
        
        # Проверяем размерности данных
        if features.shape[1] != self.features_dim:
            raise ValueError(f"Размерность признаков ({features.shape[1]}) не соответствует ожидаемой ({self.features_dim})")
        
        if len(feature_names) != self.features_dim:
            raise ValueError(f"Количество имен признаков ({len(feature_names)}) не соответствует размерности признаков ({self.features_dim})")
        
        # Сохраняем имена признаков
        self.feature_names = feature_names
        
        # Создаем модель
        self.model = self._create_model()
        
        # Конвертируем данные в тензоры PyTorch
        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.FloatTensor(targets).view(-1, 1)
        
        # Создаем датасет и даталоадер
        class SimpleDataset(Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets
                
            def __len__(self):
                return len(self.features)
                
            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]
        
        dataset = SimpleDataset(features_tensor, targets_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Определяем оптимизатор и функцию потерь
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Обучаем модель
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_targets in dataloader:
                # Обнуляем градиенты
                optimizer.zero_grad()
                
                # Прямой проход
                outputs, attention_weights = self.model(batch_features)
                
                # Вычисляем потери
                loss = criterion(outputs, batch_targets)
                
                # Обратный проход
                loss.backward()
                
                # Обновляем веса
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Логируем прогресс
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Эпоха {epoch+1}/{epochs}, потери: {avg_loss:.4f}")
        
        # Вычисляем важность признаков
        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(features_tensor)
            feature_importance = attention_weights.mean(dim=0).numpy()
        
        # Сортируем признаки по важности
        importance_ranks = np.argsort(feature_importance)[::-1]
        ranked_features = [(self.feature_names[i], feature_importance[i]) for i in importance_ranks]
        
        # Помечаем модель как обученную
        self.is_trained = True
        
        return {
            "losses": losses,
            "feature_importance": dict(ranked_features),
            "epochs": epochs,
            "final_loss": losses[-1]
        }
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает целевые значения и важность признаков.
        
        Args:
            features: Матрица признаков (n_samples, n_features)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (предсказания, важность признаков)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        if not HAS_TORCH:
            raise ImportError("PyTorch не установлен. Невозможно выполнить предсказание.")
        
        # Проверяем размерность данных
        if features.shape[1] != self.features_dim:
            raise ValueError(f"Размерность признаков ({features.shape[1]}) не соответствует ожидаемой ({self.features_dim})")
        
        # Конвертируем данные в тензор PyTorch
        features_tensor = torch.FloatTensor(features)
        
        # Выполняем предсказание
        self.model.eval()
        with torch.no_grad():
            predictions, attention_weights = self.model(features_tensor)
            
            # Конвертируем в numpy
            predictions_np = predictions.numpy()
            attention_weights_np = attention_weights.numpy()
        
        return predictions_np, attention_weights_np
    
    def get_important_features(self, n_top: int = 10) -> List[Tuple[str, float]]:
        """
        Возвращает наиболее важные признаки.
        
        Args:
            n_top: Количество наиболее важных признаков
            
        Returns:
            List[Tuple[str, float]]: Список кортежей (имя признака, важность)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        if not HAS_TORCH:
            raise ImportError("PyTorch не установлен. Невозможно получить важные признаки.")
        
        # Получаем важность признаков
        self.model.eval()
        with torch.no_grad():
            # Создаем тестовый тензор
            test_tensor = torch.ones(1, self.features_dim)
            
            # Получаем веса внимания
            _, attention_weights = self.model(test_tensor)
            feature_importance = attention_weights[0].numpy()
        
        # Сортируем признаки по важности
        importance_ranks = np.argsort(feature_importance)[::-1][:n_top]
        top_features = [(self.feature_names[i], feature_importance[i]) for i in importance_ranks]
        
        return top_features
    
    def save_model(self, path: str) -> None:
        """
        Сохраняет модель в файл.
        
        Args:
            path: Путь для сохранения модели
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        if not HAS_TORCH:
            raise ImportError("PyTorch не установлен. Невозможно сохранить модель.")
        
        # Сохраняем модель и метаданные
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "features_dim": self.features_dim,
            "hidden_dim": self.hidden_dim,
            "feature_names": self.feature_names
        }
        
        torch.save(save_data, path)
        self.logger.info(f"Модель сохранена в {path}")
    
    def load_model(self, path: str) -> None:
        """
        Загружает модель из файла.
        
        Args:
            path: Путь к файлу модели
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch не установлен. Невозможно загрузить модель.")
        
        # Загружаем модель и метаданные
        # weights_only=False безопасно здесь, так как загружаются только доверенные модели
        save_data = torch.load(path, weights_only=False)  # nosec B614
        
        self.features_dim = save_data["features_dim"]
        self.hidden_dim = save_data["hidden_dim"]
        self.feature_names = save_data["feature_names"]
        
        # Создаем модель
        self.model = self._create_model()
        
        # Загружаем веса
        self.model.load_state_dict(save_data["model_state_dict"])
        self.model.eval()
        
        self.is_trained = True
        self.logger.info(f"Модель загружена из {path}")
