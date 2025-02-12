# model/processor.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, List, Union, Optional
import logging
import re
from pathlib import Path

from .config.logging_config import get_logger
from .config.model_config import KeywordModelConfig

logger = get_logger(__name__)

class KeywordProcessor:
    """Процессор данных для модели извлечения ключевых слов"""
    
    def __init__(
        self,
        config: KeywordModelConfig,
        cache_dir: Optional[Path] = None
    ):
        self.config = config
        self.cache_dir = cache_dir
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir=cache_dir
            )
            logger.info(f"Токенизатор {config.model_name} успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка загрузки токенизатора: {e}")
            raise
            
    def preprocess_text(self, text: str) -> str:
        """
        Предварительная обработка текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Обработанный текст
        """
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление HTML-тегов
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Замена множественных пробелов и переносов строк
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление специальных символов, оставляем только буквы, цифры и пробелы
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
            
    def encode_texts(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Кодирование текстов
        
        Args:
            texts: Текст или список текстов
            return_tensors: Тип возвращаемых тензоров ('pt' для PyTorch)
            
        Returns:
            Словарь с тензорами для модели
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Предобработка текстов
        texts = [self.preprocess_text(text) for text in texts]
            
        try:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=return_tensors
            )
            
            # Добавление дополнительных метрик
            encoded['text_lengths'] = torch.tensor(
                [len(text.split()) for text in texts]
            )
            
            return encoded
            
        except Exception as e:
            logger.error(f"Ошибка кодирования текстов: {e}")
            raise
            
    def decode_keywords(
        self,
        token_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        min_length: int = 3,
        skip_special_tokens: bool = True
    ) -> Union[List[str], List[Dict[str, Union[str, float]]]]:
        """
        Декодирование токенов в ключевые слова
        
        Args:
            token_ids: Тензор с ID токенов
            scores: Опциональные оценки важности токенов
            min_length: Минимальная длина ключевого слова
            skip_special_tokens: Пропускать ли специальные токены
            
        Returns:
            Список ключевых слов или словарей с ключевыми словами и их оценками
        """
        try:
            # Декодирование токенов
            words = self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
            
            # Фильтрация и очистка
            words = [
                word.strip() 
                for word in words 
                if len(word.strip()) >= min_length
            ]
            
            # Если есть оценки, возвращаем слова с их оценками
            if scores is not None:
                return [
                    {
                        'keyword': word,
                        'score': float(score)
                    }
                    for word, score in zip(words, scores)
                ]
            
            return words
            
        except Exception as e:
            logger.error(f"Ошибка декодирования токенов: {e}")
            raise

class KeywordDetectionHead(nn.Module):
    """Голова модели для определения ключевых слов"""
    
    def __init__(self, config: KeywordModelConfig):
        super().__init__()
        
        self.dense = nn.Sequential(
            # Первый слой с нормализацией
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate),
            nn.GELU(),
            
            # Второй слой для уточнения признаков
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Dropout(config.dropout_rate),
            nn.GELU(),
            
            # Выходной слой для бинарной классификации
            nn.Linear(config.hidden_dim // 2, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)

class TrendAnalysisHead(nn.Module):
    """Голова модели для анализа трендов"""
    
    def __init__(self, config: KeywordModelConfig):
        super().__init__()
        
        self.trend_analyzer = nn.Sequential(
            # Слой анализа трендов
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            
            # Слой оценки динамики
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            
            # Выходной слой
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trend_analyzer(x)
