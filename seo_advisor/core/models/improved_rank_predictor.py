import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ImprovedRankPredictor(nn.Module):
    def __init__(self, content_dim=768, hidden_dim=256, num_heads=8):
        super().__init__()
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feedforward layers
        self.fc1 = nn.Linear(content_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, attention_mask=None):
        try:
            batch_size, seq_len, _ = x.shape
            
            # Преобразуем attention mask в правильную форму
            if attention_mask is not None:
                # Транспонируем маску, если необходимо
                if attention_mask.shape != (batch_size, seq_len):
                    attention_mask = attention_mask.transpose(0, 1)
                
                # Преобразуем в boolean маску и инвертируем
                attention_mask = attention_mask.bool()
                attention_mask = ~attention_mask
            
            # Применяем self-attention
            attended_output, _ = self.attention(
                x, x, x,
                key_padding_mask=attention_mask
            )
            
            # Усредняем по последовательности
            pooled = torch.mean(attended_output, dim=1)
            
            # Проходим через полносвязные слои
            x = self.fc1(pooled)
            x = self.relu(x)
            x = self.dropout(x)
            
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            x = self.fc3(x)
            
            return x
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании рейтинга: {str(e)}")
            raise
