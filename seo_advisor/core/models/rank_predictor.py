import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from ...config.advisor_config import ModelConfig

logger = logging.getLogger(__name__)

class RankPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Усреднение по токенам
        self.predictor = nn.Sequential(
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Усреднение по последовательности
        if attention_mask is not None:
            masked_features = features * attention_mask.unsqueeze(-1)
            averaged_features = masked_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            averaged_features = features.mean(dim=1)
            
        rank_prediction = self.predictor(averaged_features)
            
        return {
            'rank_score': rank_prediction,
            'confidence': torch.ones_like(rank_prediction)
        }
