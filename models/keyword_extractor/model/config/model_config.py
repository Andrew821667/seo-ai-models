# models/keyword_extractor/model/processor.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path

from ..config.logging_config import get_logger
from ..config.model_config import KeywordModelConfig

logger = get_logger(__name__)

class KeywordProcessor:
    """Процессор данных для модели извлечения ключевых слов"""
    
    def __init__(
        self,
        config: KeywordModelConfig,
        cache_dir: Optional[Path] = None
    ):
        """
        Инициализация процессора
        
        Args:
            config: Конфигурация модели
            cache_dir: Директория для кэширования
        """
        self.config = config
        self.cache_dir = cache_dir
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки токенизатора: {e}")
            raise
            
    def encode_texts(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Кодирование текстов
        
        Args:
            texts: Текст или список текстов
            
        Returns:
            Словарь с тензорами для модели
        """
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
        except Exception as e:
            logger.error(f"Ошибка кодирования текстов: {e}")
            raise
            
    def decode_keywords(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Декодирование токенов в ключевые слова
        
        Args:
            token_ids: Тензор с ID токенов
            skip_special_tokens: Пропускать ли специальные токены
            
        Returns:
            Список декодированных ключевых слов
        """
        try:
            return self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
        except Exception as e:
            logger.error(f"Ошибка декодирования токенов: {e}")
            raise

class KeywordDetectionHead(nn.Module):
    """Голова модели для определения ключевых слов"""
    
    def __init__(self, config: KeywordModelConfig):
        """
        Инициализация головы модели
        
        Args:
            config: Конфигурация модели
        """
        super().__init__()
        
        self.dense = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 2)  # Бинарная классификация
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входной тензор
            
        Returns:
            Логиты классификации
        """
        return self.dense(x)

class TrendAnalysisHead(nn.Module):
    """Голова модели для анализа трендов"""
    
    def __init__(self, config: KeywordModelConfig):
        """
        Инициализация головы модели
        
        Args:
            config: Конфигурация модели
        """
        super().__init__()
        
        self.trend_analyzer = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входной тензор
            
        Returns:
            Оценки трендов
        """
        return self.trend_analyzer(x)

class CompetitionAnalysisHead(nn.Module):
    """Голова модели для анализа конкуренции"""
    
    def __init__(self, config: KeywordModelConfig):
        """
        Инициализация головы модели
        
        Args:
            config: Конфигурация модели
        """
        super().__init__()
        
        self.competition_analyzer = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 3),  # Низкая/Средняя/Высокая конкуренция
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входной тензор
            
        Returns:
            Оценки уровня конкуренции
        """
        return self.competition_analyzer(x)
