# model/model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional, List, Union
import logging
from pathlib import Path

from .config.logging_config import get_logger
from .config.model_config import KeywordModelConfig
from .processor import (
    KeywordProcessor,
    KeywordDetectionHead,
    TrendAnalysisHead
)

logger = get_logger(__name__)

class KeywordExtractorModel(nn.Module):
    """Модель для извлечения и анализа ключевых слов"""
    
    def __init__(
        self,
        config: KeywordModelConfig,
        cache_dir: Optional[Path] = None
    ):
        """
        Инициализация модели
        
        Args:
            config: Конфигурация модели
            cache_dir: Директория для кэширования
        """
        super().__init__()
        self.config = config
        self.processor = KeywordProcessor(config, cache_dir)
        
        try:
            # Загрузка базовой модели
            self.transformer = AutoModel.from_pretrained(
                config.model_name,
                cache_dir=cache_dir
            )
            logger.info(f"Базовая модель {config.model_name} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки базовой модели: {e}")
            raise
            
        # Инициализация компонентов
        self.keyword_detector = KeywordDetectionHead(config)
        self.trend_analyzer = TrendAnalysisHead(config)
        
        # Механизм внимания для анализа контекста
        self.context_attention = nn.MultiheadAttention(
            config.input_dim,
            config.num_heads,
            dropout=config.dropout_rate
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход модели
        
        Args:
            input_ids: ID токенов
            attention_mask: Маска внимания
            token_type_ids: ID типов токенов (опционально)
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            # Получение эмбеддингов из трансформера
            transformer_outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            sequence_output = transformer_outputs.last_hidden_state
            
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
            
            return {
                'keyword_logits': keyword_logits,
                'trend_scores': trend_scores,
                'attention_weights': attention_weights,
                'context_embeddings': context_output
            }
            
        except Exception as e:
            logger.error(f"Ошибка при прямом проходе модели: {e}")
            raise
            
    def extract_keywords(
        self,
        texts: Union[str, List[str]],
        threshold: float = 0.5,
        min_length: int = 3,
        return_scores: bool = True
    ) -> Union[List[str], List[Dict[str, Union[str, float]]]]:
        """
        Извлечение ключевых слов из текста
        
        Args:
            texts: Текст или список текстов
            threshold: Порог для определения ключевых слов
            min_length: Минимальная длина ключевого слова
            return_scores: Возвращать ли оценки слов
            
        Returns:
            Список ключевых слов или словарей с ключевыми словами и их оценками
        """
        self.eval()
        
        try:
            # Кодирование текстов
            encoded = self.processor.encode_texts(texts)
            
            with torch.no_grad():
                # Получение предсказаний модели
                outputs = self.forward(
                    encoded['input_ids'],
                    encoded['attention_mask']
                )
                
                # Получение вероятностей и маски ключевых слов
                keyword_probs = torch.softmax(outputs['keyword_logits'], dim=-1)
                keyword_mask = keyword_probs[:, :, 1] > threshold
                
                # Получение трендов для отфильтрованных слов
                trend_scores = outputs['trend_scores'][keyword_mask]
                
                # Получение ID токенов ключевых слов
                keyword_tokens = encoded['input_ids'][keyword_mask]
                
                # Декодирование в слова
                keywords = self.processor.decode_keywords(
                    keyword_tokens,
                    scores=trend_scores if return_scores else None,
                    min_length=min_length
                )
                
                return keywords
                
        except Exception as e:
            logger.error(f"Ошибка при извлечении ключевых слов: {e}")
            raise
            
    def save_pretrained(self, save_dir: Union[str, Path]):
        """
        Сохранение модели
        
        Args:
            save_dir: Директория для сохранения
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Сохранение конфигурации
            config_path = save_dir / "config.json"
            self.config.save(str(config_path))
            
            # Сохранение весов модели
            weights_path = save_dir / "pytorch_model.bin"
            torch.save(self.state_dict(), weights_path)
            
            # Сохранение токенизатора
            self.processor.tokenizer.save_pretrained(save_dir)
            
            logger.info(f"Модель успешно сохранена в {save_dir}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            raise
            
    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        cache_dir: Optional[Path] = None
    ) -> 'KeywordExtractorModel':
        """
        Загрузка предобученной модели
        
        Args:
            model_dir: Директория с сохраненной моделью
            cache_dir: Директория для кэширования
            
        Returns:
            Загруженная модель
            
        Returns:
            KeywordExtractorModel: Инстанс загруженной модели
        """
        model_dir = Path(model_dir)
        
        try:
            # Загрузка конфигурации
            config_path = model_dir / "config.json"
            config = KeywordModelConfig.load(str(config_path))
            
            # Создание модели
            model = cls(config, cache_dir)
            
            # Загрузка весов
            weights_path = model_dir / "pytorch_model.bin"
            model.load_state_dict(torch.load(weights_path))
            
            logger.info(f"Модель успешно загружена из {model_dir}")
            return model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
