import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union
import logging

from ...config.advisor_config import ModelConfig
from ...utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)

class ContentAnalyzer(nn.Module):
    """Анализатор контента для SEO"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.text_processor = TextProcessor()
        
        # Загрузка предобученной модели и токенизатора
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.base_model = AutoModel.from_pretrained(config.model_name)
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
            
        # Дополнительные слои для SEO-анализа
        self.content_layers = nn.Sequential(
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.content_dim)
        )
        
        # Слой внимания для выделения важных частей текста
        self.attention = nn.MultiheadAttention(
            config.content_dim,
            config.num_heads,
            dropout=config.dropout
        )
        
    def forward(
        self,
        content: Union[str, List[str], Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Анализ контента
        Args:
            content: текстовый контент или уже токенизированные данные
        Returns:
            словарь с результатами анализа
        """
        try:
            # Подготовка входных данных
            if isinstance(content, (str, list)):
                inputs = self._prepare_inputs(content)
            else:
                inputs = content
                
            # Получение эмбеддингов из базовой модели
            outputs = self.base_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            sequence_output = outputs.last_hidden_state
            
            # Применение дополнительных слоев
            enhanced_features = self.content_layers(sequence_output)
            
            # Применение механизма внимания
            attended_output, attention_weights = self.attention(
                enhanced_features,
                enhanced_features,
                enhanced_features,
                key_padding_mask=~inputs['attention_mask'].bool()
            )
            
            # Расчет текстовых метрик
            if isinstance(content, (str, list)):
                text_metrics = self._calculate_text_metrics(content)
            else:
                text_metrics = None
            
            return {
                'embeddings': attended_output,
                'attention_weights': attention_weights,
                'sequence_output': sequence_output,
                'text_metrics': text_metrics
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе контента: {e}")
            raise
            
    def _prepare_inputs(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Подготовка входных данных
        Args:
            texts: текст или список текстов
        Returns:
            токенизированные данные
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Предобработка текстов
        processed_texts = [
            self.text_processor.normalize_text(text)
            for text in texts
        ]
        
        # Токенизация
        return self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
    def _calculate_text_metrics(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Расчет метрик текста
        Args:
            texts: текст или список текстов
        Returns:
            словарь с метриками
        """
        if isinstance(texts, str):
            texts = [texts]
            
        metrics = []
        for text in texts:
            text_metrics = self.text_processor.calculate_metrics(text)
            metrics.append({
                'word_count': text_metrics.word_count,
                'char_count': text_metrics.char_count,
                'sentence_count': text_metrics.sentence_count,
                'avg_word_length': text_metrics.avg_word_length,
                'avg_sentence_length': text_metrics.avg_sentence_length
            })
            
        # Преобразование в тензоры
        return {
            key: torch.tensor([m[key] for m in metrics])
            for key in metrics[0].keys()
        }

    def extract_keywords(
        self,
        text: Union[str, List[str]],
        top_k: int = 10
    ) -> Union[List[str], List[List[str]]]:
        """
        Извлечение ключевых слов
        Args:
            text: текст или список текстов
            top_k: количество ключевых слов
        Returns:
            список ключевых слов для каждого текста
        """
        if isinstance(text, str):
            return self.text_processor.extract_keywords(text, top_k=top_k)
            
        return [
            self.text_processor.extract_keywords(t, top_k=top_k)
            for t in text
        ]
