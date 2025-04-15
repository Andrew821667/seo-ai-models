import torch
import torch.nn as nn
from typing import Dict

class ImprovedRankPredictor:
    def __init__(self, industry: str = 'default'):
        self.industry = industry
        # Увеличиваем значимость ключевых метрик
        self.feature_weights = {
            'keyword_density': 0.25,      # Увеличили вес
            'content_length': 0.20,       # Увеличили вес
            'readability_score': 0.15,    # Увеличили вес
            'meta_tags_score': 0.10,
            'header_structure_score': 0.10,
            'multimedia_score': 0.05,
            'internal_linking_score': 0.05,
            'topic_relevance': 0.25,      # Увеличили вес
            'semantic_depth': 0.15,       # Увеличили вес
            'engagement_potential': 0.10
        }
        
        # Усиливаем индустриальные корректировки
        self.industry_adjustments = {
            'electronics': {
                'technical_depth': 1.5,    # Увеличили множитель
                'example_weight': 1.4,
                'spec_detail': 1.6         # Увеличили множитель
            }
        }
    
    def calculate_keyword_density(self, text: str, keywords: list) -> float:
        words = text.lower().split()
        total_words = len(words)
        
        keyword_count = 0
        for keyword in keywords:
            if ' ' in keyword:  # Для словосочетаний
                keyword_count += text.lower().count(keyword.lower()) * 2  # Удваиваем вес
            else:
                keyword_count += text.lower().count(keyword.lower())
                
        return (keyword_count / total_words) * 1.5 if total_words > 0 else 0
    
    def predict_position(self, features: Dict[str, float], text: str = None, keywords: list = None) -> Dict:
        if text and keywords:
            features['keyword_density'] = self.calculate_keyword_density(text, keywords)
            
        # Усиливаем влияние высоких показателей
        normalized_features = {
            k: min(max(v * 1.2, 0), 1) for k, v in features.items()  # Увеличили множитель
        }
        weighted_scores = {
            k: normalized_features[k] * self.feature_weights.get(k, 0) 
            for k in normalized_features
        }
        
        if self.industry in self.industry_adjustments:
            adj = self.industry_adjustments[self.industry]
            for factor, mult in adj.items():
                if factor in weighted_scores:
                    weighted_scores[factor] *= mult
        total_score = sum(weighted_scores.values())
        
        # Улучшенная формула расчета позиции
        position = max(1, min(100, 50 * (1 - total_score * 1.5)))  # Увеличили влияние скора
        
        return {
            'position': position,
            'feature_scores': normalized_features,
            'weighted_scores': weighted_scores
        }
    
    def generate_recommendations(self, features: Dict[str, float]) -> Dict[str, list]:
        recommendations = {}
        for feature, value in features.items():
            if value < 0.7:  # Повысили порог для рекомендаций
                recommendations[feature] = [
                    f"Улучшить {feature}",
                    f"Оптимизировать {feature}",
                    f"Повысить {feature}"
                ]
        return recommendations
