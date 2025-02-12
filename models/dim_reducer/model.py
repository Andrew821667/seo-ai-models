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
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity(),
            nn.LeakyReLU() if config.activation == "leaky_relu" else nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        self.seo_attention = SEOAttentionLayer(config.latent_dim, config.num_attention_heads)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        attended = self.seo_attention(z)
        return {
            'latent': attended,
            'attention': self.seo_attention
        }
