
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

class RankPredictor(nn.Module):
    def __init__(self, input_size: int = 50):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Список факторов ранжирования
        self.ranking_factors = [
            'keyword_density',
            'content_length',
            'readability_score',
            'meta_tags_score',
            'header_structure_score'
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _prepare_features(self, page_features: Dict) -> torch.Tensor:
        # Преобразование признаков в тензор
        features = []
        for factor in self.ranking_factors:
            features.append(page_features.get(factor, 0.0))
        return torch.tensor(features, dtype=torch.float32)
    
    def predict_position(self, page_features: Dict) -> float:
        # Подготовка данных
        feature_tensor = self._prepare_features(page_features)
        
        # Прогноз позиции
        with torch.no_grad():
            position = self.forward(feature_tensor)
        
        return position.item()
    
    def evaluate_page_strength(self, page_features: Dict) -> Dict[str, float]:
        # Оценка важности различных факторов
        strengths = {}
        for factor in self.ranking_factors:
            value = page_features.get(factor, 0.0)
            if value > 0.7:
                strengths[factor] = "высокая"
            elif value > 0.4:
                strengths[factor] = "средняя"
            else:
                strengths[factor] = "низкая"
        return strengths
