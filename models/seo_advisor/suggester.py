import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import logging

from ...config.advisor_config import ModelConfig

logger = logging.getLogger(__name__)

class OptimizationSuggester(nn.Module):
    """Генератор рекомендаций по оптимизации"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Генератор рекомендаций
        self.suggestion_generator = nn.Sequential(
            # Первый слой с нормализацией
            nn.Linear(config.content_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            
            # Второй слой с повышением размерности
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            
            # Выходной слой для генерации рекомендаций
            nn.Linear(
                config.hidden_dim * 2,
                config.num_suggestions * config.hidden_dim
            )
        )
        
        # Оценка важности рекомендаций
        self.importance_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Анализ сложности внедрения
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 3)  # Легко/Средне/Сложно
        )
        
        # Оценка потенциального влияния
        self.impact_estimator = nn.Sequential(
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
        Генерация рекомендаций
        Args:
            features: входные признаки
            attention_mask: маска внимания
        Returns:
            словарь с рекомендациями и их характеристиками
        """
        try:
            batch_size = features.size(0)
            
            # Генерация базовых рекомендаций
            suggestions = self.suggestion_generator(features)
            suggestions = suggestions.view(
                batch_size,
                self.config.num_suggestions,
                -1
            )
            
            # Оценка важности каждой рекомендации
                        # Оценка важности каждой рекомендации
            importance_scores = self.importance_estimator(suggestions)
            
            # Анализ сложности внедрения
            complexity_scores = self.complexity_analyzer(suggestions)
            complexity_probs = torch.softmax(complexity_scores, dim=-1)
            
            # Оценка потенциального влияния
            impact_scores = self.impact_estimator(suggestions)
            
            # Если есть маска внимания, применяем её
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand_as(suggestions)
                suggestions = suggestions * mask
                importance_scores = importance_scores * mask[:, :, :1]
                complexity_scores = complexity_scores * mask[:, :, :3]
                impact_scores = impact_scores * mask[:, :, :1]
            
            return {
                'suggestions': suggestions,
                'importance_scores': importance_scores,
                'complexity_scores': complexity_probs,
                'impact_scores': impact_scores
            }
            
        except Exception as e:
            logger.error(f"Ошибка при генерации рекомендаций: {e}")
            raise
            
    def decode_suggestions(
        self,
        suggestions: torch.Tensor,
        tokenizer
    ) -> List[str]:
        """
        Декодирование рекомендаций в текст
        Args:
            suggestions: тензор рекомендаций
            tokenizer: токенизатор для декодирования
        Returns:
            список текстовых рекомендаций
        """
        try:
            # Преобразование в индексы токенов
            token_ids = torch.argmax(suggestions, dim=-1)
            
            # Декодирование каждой рекомендации
            texts = []
            for suggestion in token_ids:
                text = tokenizer.decode(suggestion, skip_special_tokens=True)
                texts.append(text.strip())
            
            return texts
            
        except Exception as e:
            logger.error(f"Ошибка при декодировании рекомендаций: {e}")
            raise
            
    def filter_suggestions(
        self,
        suggestions: Dict[str, torch.Tensor],
        min_importance: float = 0.5,
        max_complexity: float = 0.7
    ) -> Dict[str, torch.Tensor]:
        """
        Фильтрация рекомендаций по важности и сложности
        Args:
            suggestions: словарь с рекомендациями
            min_importance: минимальная важность
            max_complexity: максимальная сложность
        Returns:
            отфильтрованные рекомендации
        """
        try:
            # Создание маски для фильтрации
            importance_mask = suggestions['importance_scores'] >= min_importance
            complexity_mask = suggestions['complexity_scores'].max(dim=-1)[0] <= max_complexity
            
            # Применение маски ко всем компонентам
            filtered = {}
            for key, value in suggestions.items():
                if key in ['suggestions', 'importance_scores', 'impact_scores']:
                    filtered[key] = value[importance_mask & complexity_mask]
                elif key == 'complexity_scores':
                    filtered[key] = value[importance_mask & complexity_mask]
                    
            return filtered
            
        except Exception as e:
            logger.error(f"Ошибка при фильтрации рекомендаций: {e}")
            raise
            
    def prioritize_suggestions(
        self,
        suggestions: Dict[str, torch.Tensor],
        weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Приоритизация рекомендаций
        Args:
            suggestions: словарь с рекомендациями
            weights: веса для различных факторов
        Returns:
            приоритизированные рекомендации
        """
        if weights is None:
            weights = {
                'importance': 0.4,
                'impact': 0.4,
                'complexity': 0.2
            }
            
        try:
            # Расчет общего скора для каждой рекомендации
            importance_score = suggestions['importance_scores'] * weights['importance']
            impact_score = suggestions['impact_scores'] * weights['impact']
            complexity_score = (1 - suggestions['complexity_scores'].max(dim=-1)[0]) * weights['complexity']
            
            total_score = importance_score + impact_score + complexity_score
            
            # Сортировка по общему скору
            _, indices = torch.sort(total_score, descending=True)
            
            # Сортировка всех компонентов
            prioritized = {}
            for key, value in suggestions.items():
                prioritized[key] = value[indices]
                
            return prioritized
            
        except Exception as e:
            logger.error(f"Ошибка при приоритизации рекомендаций: {e}")
            raise
