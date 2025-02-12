# models/keyword_extractor/model/model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional
import logging

from ..config.model_config import KeywordModelConfig
from ..config.logging_config import get_logger
from .processor import (
    KeywordDetectionHead,
    TrendAnalysisHead,
    CompetitionAnalysisHead
)

logger = get_logger(__name__)

class KeywordExtractorModel(nn.Module):
    """Основная модель для извлечения и анализа ключевых слов"""
    
    def __init__(
        self,
        config: KeywordModelConfig,
        cache_dir: Optional[str] = None
    ):
        """
        Инициализация модели
        
        Args:
            config: Конфигурация модели
            cache_dir: Директория для кэширования
        """
        super().__init__()
        self.config = config
        
        # Загрузка базовой модели
        try:
            self.transformer = AutoModel.from_pretrained(
                config.model_name,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки базовой модели: {e}")
            raise
            
        # Компоненты модели
        self.keyword_detector = KeywordDetectionHead(config)
        self.trend_analyzer = TrendAnalysisHead(config)
        self.competition_analyzer = CompetitionAnalysisHead(config)
        
        # Механизм внимания
        self.context_attention = nn.MultiheadAttention(
            config.input_dim,
            config.num_heads,
            dropout=config.dropout_rate
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход модели
        
        Args:
            input_ids: ID токенов
            attention_mask: Маска внимания
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            # Получение эмбеддингов из трансформера
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            sequence_output = outputs.last_hidden_state
            
            # Применение механизма внимания
            context_output, attention_weights = self.context_attention(
                sequence_output,
                sequence_output,
                sequence_output,
                key_padding_mask=~attention_mask.bool()
            )
            
            # Определение ключевых слов
            keyword_logits = self.keyword_detector(context_output)
            
            # Анализ трендов
            trend_scores = self.trend_analyzer(context_output)
            
            # Анализ конкуренции
            competition_scores = self.competition_analyzer(context_output)
            
            return {
                'keyword_logits': keyword_logits,
                'trend_scores': trend_scores,
                'competition_scores': competition_scores,
                'attention_weights': attention_weights,
                'context_embeddings': context_output
            }
            
        except Exception as e:
            logger.error(f"Ошибка при прямом проходе модели: {e}")
            raise
            
    def predict_keywords(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Предсказание ключевых слов
        
        Args:
            input_ids: ID токенов
            attention_mask: Маска внимания
            threshold: Порог для определения ключевых слов
            
        Returns:
            Словарь с предсказаниями
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # Получение вероятностей
            keyword_probs = torch.softmax(outputs['keyword_logits'], dim=-1)
            
            # Определение ключевых слов по порогу
            keyword_mask = keyword_probs[:, :, 1] > threshold
            
            return {
                'keyword_mask': keyword_mask,
                'keyword_probs': keyword_probs,
                'trend_scores': outputs['trend_scores'],
                'competition_scores': outputs['competition_scores']
            }
