from typing import Dict, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass

@dataclass
class IndustryThresholds:
    content_length: Dict[str, int]
    keyword_density: Dict[str, float]
    readability_score: Dict[str, int]
    meta_tags_score: Dict[str, float]
    header_structure_score: Dict[str, float]
    backlinks_count: Dict[str, int]
    multimedia_score: Dict[str, float]
    internal_linking_score: Dict[str, float]
    
    @classmethod
    def create_default(cls):
        return cls(
            content_length={'low': 650, 'high': 2400, 'optimal_min': 950, 'optimal_max': 2600},
            keyword_density={'low': 0.01, 'high': 0.03},
            readability_score={'low': 40, 'high': 70},
            meta_tags_score={'low': 0.4, 'high': 0.7},
            header_structure_score={'low': 0.4, 'high': 0.7},
            backlinks_count={'low': 10, 'high': 100},
            multimedia_score={'low': 0.3, 'high': 0.7},
            internal_linking_score={'low': 0.3, 'high': 0.7}
        )

@dataclass
class IndustryWeights:
    keyword_density: float
    content_length: float
    readability_score: float
    meta_tags_score: float
    header_structure_score: float
    backlinks_count: float
    multimedia_score: float
    internal_linking_score: float
    
    @classmethod
    def create_default(cls):
        return cls(
            keyword_density=0.23,
            content_length=0.21,
            readability_score=0.17,
            meta_tags_score=0.13,
            header_structure_score=0.11,
            backlinks_count=0.07,
            multimedia_score=0.04,
            internal_linking_score=0.04
        )

class ImprovedRankPredictor:
    def __init__(self, industry: str = 'default'):
        self.industry = industry
        self.thresholds = self._get_industry_thresholds()
        self.weights = self._get_industry_weights()
        self.history = []
        
    def _get_industry_thresholds(self) -> IndustryThresholds:
        if self.industry == 'scientific_blog':
            return IndustryThresholds(
                content_length={'low': 1200, 'high': 3300, 'optimal_min': 1700, 'optimal_max': 3600},
                keyword_density={'low': 0.012, 'high': 0.028},
                readability_score={'low': 50, 'high': 85},
                meta_tags_score={'low': 0.55, 'high': 0.95},
                header_structure_score={'low': 0.55, 'high': 0.95},
                backlinks_count={'low': 30, 'high': 220},
                multimedia_score={'low': 0.35, 'high': 0.85},
                internal_linking_score={'low': 0.45, 'high': 0.85}
            )
        elif self.industry == 'blog':
            return IndustryThresholds(
                content_length={'low': 850, 'high': 2600, 'optimal_min': 1250, 'optimal_max': 2900},
                keyword_density={'low': 0.014, 'high': 0.032},
                readability_score={'low': 45, 'high': 72},
                meta_tags_score={'low': 0.45, 'high': 0.85},
                header_structure_score={'low': 0.45, 'high': 0.85},
                backlinks_count={'low': 20, 'high': 160},
                multimedia_score={'low': 0.35, 'high': 0.85},
                internal_linking_score={'low': 0.35, 'high': 0.75}
            )
        elif self.industry == 'ecommerce':
            return IndustryThresholds(
                content_length={'low': 450, 'high': 2050, 'optimal_min': 800, 'optimal_max': 2350},
                keyword_density={'low': 0.016, 'high': 0.038},
                readability_score={'low': 38, 'high': 65},
                meta_tags_score={'low': 0.5, 'high': 0.92},
                header_structure_score={'low': 0.4, 'high': 0.85},
                backlinks_count={'low': 35, 'high': 240},
                multimedia_score={'low': 0.45, 'high': 0.92},
                internal_linking_score={'low': 0.35, 'high': 0.75}
            )
        return IndustryThresholds.create_default()

    def _get_industry_weights(self) -> IndustryWeights:
        base = IndustryWeights.create_default()
        
        industry_multipliers = {
            'scientific_blog': {
                'content_length': 1.4,
                'header_structure_score': 1.3,
                'readability_score': 1.4,
                'meta_tags_score': 1.25,
                'internal_linking_score': 1.2
            },
            'ecommerce': {
                'meta_tags_score': 1.4,
                'header_structure_score': 1.3,
                'multimedia_score': 1.3,
                'backlinks_count': 1.3,
                'keyword_density': 1.25
            },
            'blog': {
                'content_length': 1.3,
                'readability_score': 1.25,
                'keyword_density': 1.2,
                'multimedia_score': 1.15,
                'internal_linking_score': 1.1
            }
        }
        
        if self.industry in industry_multipliers:
            weights_dict = {
                'keyword_density': base.keyword_density,
                'content_length': base.content_length,
                'readability_score': base.readability_score,
                'meta_tags_score': base.meta_tags_score,
                'header_structure_score': base.header_structure_score,
                'backlinks_count': base.backlinks_count,
                'multimedia_score': base.multimedia_score,
                'internal_linking_score': base.internal_linking_score
            }
            
            for attr, mult in industry_multipliers[self.industry].items():
                weights_dict[attr] *= mult
            
            total = sum(weights_dict.values())
            normalized_weights = {k: v/total for k, v in weights_dict.items()}
            
            return IndustryWeights(**normalized_weights)
                
        return base

    def _calculate_feature_score(self, feature_name: str, value: float) -> float:
        thresholds = getattr(self.thresholds, feature_name)
        
        if feature_name == 'content_length':
            if thresholds['optimal_min'] <= value <= thresholds['optimal_max']:
                return 1.0
            elif value < thresholds['low']:
                return max(0.5, (value / thresholds['low']) ** 0.7)
            elif value > thresholds['high']:
                excess = (value - thresholds['high']) / (thresholds['high'] * 0.5)
                return max(0.7, 1 - (excess ** 1.2))
        
        if value < thresholds['low']:
            return 0.4 + (0.3 * value / thresholds['low'])
        elif value > thresholds['high']:
            return 0.9 - (0.2 * (value - thresholds['high']) / thresholds['high'])
        
        position = (value - thresholds['low']) / (thresholds['high'] - thresholds['low'])
        return 0.7 + (0.2 * position)

    def predict_position(self, page_features: Dict[str, float]) -> Dict[str, Union[float, Dict]]:
        feature_scores = {}
        weighted_scores = {}
        total_score = 0.0
        
        weights_dict = {
            'keyword_density': self.weights.keyword_density,
            'content_length': self.weights.content_length,
            'readability_score': self.weights.readability_score,
            'meta_tags_score': self.weights.meta_tags_score,
            'header_structure_score': self.weights.header_structure_score,
            'backlinks_count': self.weights.backlinks_count,
            'multimedia_score': self.weights.multimedia_score,
            'internal_linking_score': self.weights.internal_linking_score
        }
        
        for feature_name, weight in weights_dict.items():
            if feature_name not in page_features:
                continue
                
            value = page_features[feature_name]
            feature_score = self._calculate_feature_score(feature_name, value)
            
            feature_scores[feature_name] = feature_score
            weighted_scores[feature_name] = feature_score * weight
            total_score += feature_score * weight
        
        position = max(1, min(100, 101 - (total_score * 100)))
        
        self.history.append({
            'timestamp': datetime.now(),
            'features': page_features,
            'position': position,
            'score': total_score
        })
        
        return {
            'position': position,
            'total_score': total_score,
            'feature_scores': feature_scores,
            'weighted_scores': weighted_scores
        }

    def generate_recommendations(self, page_features: Dict[str, float]) -> Dict[str, List[str]]:
        recommendations = {}
        prediction = self.predict_position(page_features)
        feature_scores = prediction['feature_scores']
        
        weights_dict = {
            'keyword_density': self.weights.keyword_density,
            'content_length': self.weights.content_length,
            'readability_score': self.weights.readability_score,
            'meta_tags_score': self.weights.meta_tags_score,
            'header_structure_score': self.weights.header_structure_score,
            'backlinks_count': self.weights.backlinks_count,
            'multimedia_score': self.weights.multimedia_score,
            'internal_linking_score': self.weights.internal_linking_score
        }
        
        feature_importance = {
            feature: weights_dict[feature] * (1 - score)
            for feature, score in feature_scores.items()
        }
        
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for feature, importance in sorted_features[:3]:
            value = page_features[feature]
            thresholds = getattr(self.thresholds, feature)
            
            if feature == 'content_length':
                if value < thresholds['optimal_min']:
                    recommendations[feature] = [
                        f"Увеличьте объем контента до {thresholds['optimal_min']} слов",
                        "Добавьте релевантные примеры и исследования",
                        "Расширьте описание ключевых концепций"
                    ]
                elif value > thresholds['optimal_max']:
                    recommendations[feature] = [
                        f"Сократите контент до {thresholds['optimal_max']} слов",
                        "Удалите повторяющуюся информацию",
                        "Сфокусируйтесь на ключевых моментах"
                    ]
            elif feature == 'keyword_density':
                if value < thresholds['low']:
                    recommendations[feature] = [
                        "Увеличьте плотность ключевых слов",
                        "Добавьте синонимы и связанные термины",
                        "Используйте ключевые слова в заголовках"
                    ]
                elif value > thresholds['high']:
                    recommendations[feature] = [
                        "Уменьшите плотность ключевых слов",
                        "Используйте больше LSI-keywords",
                        "Сделайте текст более естественным"
                    ]
            elif feature == 'readability_score':
                if value < thresholds['low']:
                    recommendations[feature] = [
                        "Упростите текст",
                        "Используйте более короткие предложения",
                        "Добавьте подзаголовки и списки"
                    ]
                    
        return recommendations

    def get_history_analytics(self) -> Dict[str, float]:
        if not self.history:
            return {}
            
        positions = [record['position'] for record in self.history]
        scores = [record['score'] for record in self.history]
        
        return {
            'avg_position': sum(positions) / len(positions),
            'min_position': min(positions),
            'max_position': max(positions),
            'avg_score': sum(scores) / len(scores),
            'predictions_count': len(self.history)
        }
