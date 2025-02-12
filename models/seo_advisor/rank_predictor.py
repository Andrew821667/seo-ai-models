import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from ...config.advisor_config import ModelConfig

logger = logging.getLogger(__name__)

class RankPredictor(nn.Module):
    """Предсказание SEO-рейтинга"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Основной блок предсказания
        self.predictor = nn.Sequential(
            # Первый слой с дропаутом
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            
            # Второй слой с уменьшением размерности
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            
            # Выходной слой
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Слой оценки уверенности
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Предсказание рейтинга
        Args:
            features: входные признаки
            attention_mask: маска внимания
        Returns:
            словарь с предсказанием и уверенностью
        """
        try:
            # Получение промежуточного представления
            hidden = self.predictor[:-3](features)  # До последнего линейного слоя
            
            # Предсказание рейтинга
            rank_prediction = self.predictor[-3:](hidden)
            
            # Оценка уверенности
            confidence = self.confidence_estimator(hidden)
            
            # Если есть маска внимания, учитываем ее
            if attention_mask is not None:
                rank_prediction = rank_prediction * attention_mask.unsqueeze(-1)
                confidence = confidence * attention_mask.unsqueeze(-1)
            
            # Агрегация по последовательности
            rank_score = rank_prediction.mean(dim=1)
            confidence_score = confidence.mean(dim=1)
            
            return {
                'rank_score': rank_score,
                'confidence': confidence_score,
                'rank_prediction': rank_prediction,
                'hidden_features': hidden
            }
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании рейтинга: {e}")
            raise
            
    def explain_prediction(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Объяснение предсказания
        Args:
            features: входные признаки
            attention_mask: маска внимания
        Returns:
            словарь с объяснениями
        """
        with torch.no_grad():
            # Получение предсказания
            outputs = self.forward(features, attention_mask)
            
            # Расчет градиентов для объяснения
            gradients = torch.autograd.grad(
                outputs['rank_score'].sum(),
                features,
                create_graph=True
            )[0]
            
            # Нормализация градиентов
            feature_importance = torch.norm(gradients, dim=-1)
            if attention_mask is not None:
                feature_importance = feature_importance * attention_mask
            
            return {
                'feature_importance': feature_importance,
                'rank_score': outputs['rank_score'],
                'confidence': outputs['confidence']
            }
            
    def calibrate_confidence(
        self,
        confidence: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Калибровка оценки уверенности
        Args:
            confidence: исходная оценка уверенности
            temperature: температура для калибровки
        Returns:
            калиброванная оценка
        """
        return torch.sigmoid(confidence / temperature)
