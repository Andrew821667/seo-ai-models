
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime

@dataclass
class CompetitorAnalysis:
    url: str
    content_metrics: Dict[str, float]
    keyword_metrics: Dict[str, float]
    strengths: List[str]
    overall_score: float

@dataclass
class IndustryMetrics:
    keyword_density_range: tuple[float, float]
    content_length_range: tuple[float, float]
    readability_range: tuple[float, float]
    feature_importance: Dict[str, float]
    competitive_threshold: float

class CalibratedRankPredictor:
    def __init__(self, industry: str = 'default'):
        self.industry = industry
        # Калиброванные веса на основе реальных данных
        self.feature_weights = {
            'keyword_density': 0.20,  # Снижен вес
            'content_length': 0.15,   # Снижен вес
            'readability_score': 0.12,
            'meta_tags_score': 0.08,
            'header_structure_score': 0.10,
            'multimedia_score': 0.07,
            'internal_linking_score': 0.08,
            'topic_relevance': 0.30,   # Повышен вес
            'semantic_depth': 0.20,    # Повышен вес
            'engagement_potential': 0.15,
            'competitive_edge': 0.25    # Новый фактор
        }
        
        # Отраслевые настройки
        self.industry_metrics = {
            'electronics': IndustryMetrics(
                keyword_density_range=(0.015, 0.035),
                content_length_range=(1200, 3500),
                readability_range=(45, 75),
                feature_importance={
                    'technical_depth': 1.4,
                    'spec_detail': 1.5,
                    'comparison_tables': 1.3
                },
                competitive_threshold=0.85
            ),
            'finance': IndustryMetrics(
                keyword_density_range=(0.02, 0.04),
                content_length_range=(1500, 4000),
                readability_range=(50, 80),
                feature_importance={
                    'trust_signals': 1.6,
                    'regulatory_compliance': 1.5,
                    'data_visualization': 1.3
                },
                competitive_threshold=0.90
            ),
            'healthcare': IndustryMetrics(
                keyword_density_range=(0.01, 0.03),
                content_length_range=(1800, 4500),
                readability_range=(55, 85),
                feature_importance={
                    'medical_accuracy': 1.7,
                    'source_credibility': 1.6,
                    'patient_accessibility': 1.4
                },
                competitive_threshold=0.92
            )
        }

    def analyze_competitors(self, competitors_data: List[Dict]) -> List[CompetitorAnalysis]:
        """Анализ конкурентов с учетом отраслевой специфики"""
        analyses = []
        industry_metrics = self.industry_metrics.get(self.industry)
        
        for competitor in competitors_data:
            content_metrics = self._calculate_content_metrics(competitor['content'])
            keyword_metrics = self._analyze_keywords(competitor['content'], competitor['keywords'])
            
            strengths = self._identify_strengths(
                content_metrics, 
                keyword_metrics,
                industry_metrics
            )
            
            overall_score = self._calculate_competitor_score(
                content_metrics,
                keyword_metrics,
                industry_metrics
            )
            
            analyses.append(CompetitorAnalysis(
                url=competitor['url'],
                content_metrics=content_metrics,
                keyword_metrics=keyword_metrics,
                strengths=strengths,
                overall_score=overall_score
            ))
        
        return analyses

    def _calculate_content_metrics(self, content: str) -> Dict[str, float]:
        """Расчет метрик контента"""
        words = content.split()
        sentences = content.split('.')
        
        return {
            'length': len(words),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'readability': 0.7,  # Заглушка, требуется реальная имплементация
            'structure_score': 0.8  # Заглушка, требуется реальная имплементация
        }

    def _analyze_keywords(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Анализ ключевых слов"""
        content_lower = content.lower()
        word_count = len(content_lower.split())
        
        keyword_counts = {
            keyword: content_lower.count(keyword.lower())
            for keyword in keywords
        }
        
        total_occurrences = sum(keyword_counts.values())
        density = total_occurrences / word_count if word_count > 0 else 0
        
        return {
            'density': density,
            'counts': keyword_counts,
            'distribution_score': 0.75  # Заглушка, требуется реальная имплементация
        }

    def _identify_strengths(
        self,
        content_metrics: Dict[str, float],
        keyword_metrics: Dict[str, float],
        industry_metrics: IndustryMetrics
    ) -> List[str]:
        """Определение сильных сторон контента"""
        strengths = []
        
        if content_metrics['length'] > industry_metrics.content_length_range[0]:
            strengths.append("Достаточная длина контента")
            
        if keyword_metrics['density'] >= industry_metrics.keyword_density_range[0]:
            strengths.append("Оптимальная плотность ключевых слов")
            
        if content_metrics['readability'] > 0.7:
            strengths.append("Хорошая читабельность")
            
        return strengths

    def _calculate_competitor_score(
        self,
        content_metrics: Dict[str, float],
        keyword_metrics: Dict[str, float],
        industry_metrics: IndustryMetrics
    ) -> float:
        """Расчет общего скора конкурента"""
        scores = []
        
        # Оценка длины контента
        length_score = self._normalize_value(
            content_metrics['length'],
            industry_metrics.content_length_range[0],
            industry_metrics.content_length_range[1]
        )
        scores.append(length_score * 0.3)
        
        # Оценка плотности ключевых слов
        density_score = self._normalize_value(
            keyword_metrics['density'],
            industry_metrics.keyword_density_range[0],
            industry_metrics.keyword_density_range[1]
        )
        scores.append(density_score * 0.3)
        
        # Оценка читабельности
        readability_score = self._normalize_value(
            content_metrics['readability'],
            industry_metrics.readability_range[0],
            industry_metrics.readability_range[1]
        )
        scores.append(readability_score * 0.2)
        
        # Структура контента
        scores.append(content_metrics['structure_score'] * 0.2)
        
        return sum(scores)

    def predict_position(
        self,
        features: Dict[str, float],
        competitor_analyses: Optional[List[CompetitorAnalysis]] = None
    ) -> Dict:
        """Улучшенное предсказание позиции с учетом конкурентов"""
        industry_metrics = self.industry_metrics.get(self.industry)
        if not industry_metrics:
            raise ValueError(f"Industry {self.industry} not supported")

        # Нормализация с учетом отраслевых диапазонов
        normalized_features = self._normalize_features(features, industry_metrics)
        
        # Учет конкурентного анализа
        if competitor_analyses:
            competitive_edge = self._calculate_competitive_edge(
                normalized_features,
                competitor_analyses,
                industry_metrics
            )
            normalized_features['competitive_edge'] = competitive_edge
        
        # Расчет взвешенных оценок с учетом отрасли
        weighted_scores = self._calculate_weighted_scores(
            normalized_features,
            industry_metrics
        )
        
        # Калиброванный расчет позиции
        total_score = sum(weighted_scores.values())
        position = self._calibrate_position(
            total_score,
            competitor_analyses if competitor_analyses else None
        )
        
        return {
            'position': position,
            'feature_scores': normalized_features,
            'weighted_scores': weighted_scores,
            'industry_benchmark': self._get_industry_benchmark(industry_metrics)
        }

    def _normalize_features(
        self,
        features: Dict[str, float],
        industry_metrics: IndustryMetrics
    ) -> Dict[str, float]:
        """Нормализация метрик с учетом отраслевых диапазонов"""
        normalized = {}
        
        # Нормализация плотности ключевых слов
        if 'keyword_density' in features:
            min_density, max_density = industry_metrics.keyword_density_range
            normalized['keyword_density'] = self._normalize_value(
                features['keyword_density'],
                min_density,
                max_density
            )
        
        # Нормализация длины контента
        if 'content_length' in features:
            min_length, max_length = industry_metrics.content_length_range
            normalized['content_length'] = self._normalize_value(
                features['content_length'],
                min_length,
                max_length
            )
        
        # Остальные метрики
        for key, value in features.items():
            if key not in normalized:
                normalized[key] = min(max(value, 0), 1)
        
        return normalized

    def _calculate_competitive_edge(
        self,
        features: Dict[str, float],
        competitor_analyses: List[CompetitorAnalysis],
        industry_metrics: IndustryMetrics
    ) -> float:
        """Расчет конкурентного преимущества"""
        competitor_scores = [analysis.overall_score for analysis in competitor_analyses]
        if not competitor_scores:
            return 0.5
        
        avg_competitor_score = np.mean(competitor_scores)
        max_competitor_score = max(competitor_scores)
        
        # Расчет относительного преимущества
        content_score = sum(
            features[key] * self.feature_weights[key]
            for key in features
            if key in self.feature_weights
        )
        
        relative_advantage = (content_score - avg_competitor_score) / (
            max_competitor_score - avg_competitor_score
            if max_competitor_score > avg_competitor_score
            else 1
        )
        
        return min(max(0.5 + relative_advantage, 0), 1)

    def _calculate_weighted_scores(
        self,
        features: Dict[str, float],
        industry_metrics: IndustryMetrics
    ) -> Dict[str, float]:
        """Расчет взвешенных оценок с учетом отрасли"""
        weighted_scores = {}
        
        for feature, value in features.items():
            if feature in self.feature_weights:
                weight = self.feature_weights[feature]
                
                # Применяем отраслевые корректировки, если есть
                if feature in industry_metrics.feature_importance:
                    weight *= industry_metrics.feature_importance[feature]
                    
                weighted_scores[feature] = value * weight
                
        return weighted_scores

    def _calibrate_position(
        self,
        total_score: float,
        competitor_analyses: Optional[List[CompetitorAnalysis]]
    ) -> float:
        """Калиброванный расчет позиции с учетом конкурентов"""
        base_position = 100 * (1 - total_score)
        
        if competitor_analyses:
            competitor_scores = [
                analysis.overall_score 
                for analysis in competitor_analyses
            ]
            if competitor_scores:
                # Учет распределения оценок конкурентов
                score_percentile = sum(
                    1 for score in competitor_scores 
                    if score < total_score
                ) / len(competitor_scores)
                
                # Корректировка позиции
                adjusted_position = (
                    base_position * (1 - score_percentile) +
                    score_percentile * max(1, base_position * 0.5)
                )
                return max(1, min(100, adjusted_position))
        
        return max(1, min(100, base_position))

    @staticmethod
    def _normalize_value(value: float, min_val: float, max_val: float) -> float:
        """Нормализация значения в диапазоне"""
        if max_val == min_val:
            return 0.5
        return min(max((value - min_val) / (max_val - min_val), 0), 1)

    def _get_industry_benchmark(self, industry_metrics: IndustryMetrics) -> Dict:
        """Получение отраслевых бенчмарков"""
        return {
            'keyword_density': {
                'min': industry_metrics.keyword_density_range[0],
                'max': industry_metrics.keyword_density_range[1]
            },
            'content_length': {
                'min': industry_metrics.content_length_range[0],
                'max': industry_metrics.content_length_range[1]
            },
            'readability': {
                'min': industry_metrics.readability_range[0],
                'max': industry_metrics.readability_range[1]
            },
            'competitive_threshold': industry_metrics.competitive_threshold
        }
