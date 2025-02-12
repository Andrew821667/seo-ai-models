import torch
import torch.nn as nn
from transformers import AutoModel
import logging
from typing import Dict, Optional

from common.config.dim_reducer_config import DimReducerConfig
from common.utils.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

class SEOAttentionLayer(nn.Module):
    """SEO-специфичный слой внимания"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

class DimensionReducer(nn.Module):
    """Модель для сжатия SEO-характеристик"""
    def __init__(self, config: DimReducerConfig):
        super().__init__()
        self.config = config
        
        # Энкодер
        encoder_layers = []
        current_dim = config.input_dim
        for i in range(config.num_encoder_layers):
            next_dim = config.latent_dim if i == config.num_encoder_layers-1 else config.hidden_dim
            encoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim) if config.use_batch_norm else nn.Identity(),
                nn.LeakyReLU() if config.activation == "leaky_relu" else nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = next_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # SEO-специфичное внимание
        self.seo_attention = SEOAttentionLayer(config.latent_dim, config.num_attention_heads)
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LeakyReLU() if config.activation == "leaky_relu" else nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.input_dim)
        )
        
        # Оценка важности признаков
        self.feature_importance = nn.Linear(config.latent_dim, 1)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Сжатие входных данных"""
        encoded = self.encoder(x)
        attended = self.seo_attention(encoded)
        return attended
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Восстановление данных из сжатого представления"""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Прямой проход модели"""
        z = self.encode(x)
        importance = torch.sigmoid(self.feature_importance(z))
        reconstructed = self.decode(z)
        
        return {
            'latent': z,
            'reconstructed': reconstructed,
            'importance': importance
        }
