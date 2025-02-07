import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union
import logging

from ...config.advisor_config import ModelConfig

logger = logging.getLogger(__name__)

class ContentAnalyzer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Используем DistilBERT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.base_model = AutoModel.from_pretrained('distilbert-base-uncased')
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
    def forward(self, inputs: Dict[str, Union[str, List[str]]]) -> Dict[str, torch.Tensor]:
        try:
            # Токенизация текста
            if isinstance(inputs['text'], str):
                texts = [inputs['text']]
            else:
                texts = inputs['text']
                
            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Получение эмбеддингов
            outputs = self.base_model(**tokens)
            
            return {
                'embeddings': outputs.last_hidden_state,
                'attention_mask': tokens['attention_mask']
            }
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise
