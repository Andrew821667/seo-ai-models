import torch
import torch.nn as nn
import logging

from models.seo_advisor.config.model_config import ModelConfig

logger = logging.getLogger(__name__)

class SEOAdvisor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate),
            nn.ReLU()
        )
        
    def forward(self, x: dict) -> dict:
        features = self.feature_extractor(x)
        return {'features': features}
