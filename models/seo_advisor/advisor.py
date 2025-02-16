
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor
from models.seo_advisor.content_analyzer import ContentAnalyzer
from models.seo_advisor.suggester import Suggester

@dataclass
class SEOAnalysisReport:
    """Полный отчет по SEO анализу"""
    timestamp: datetime
    content_metrics: Dict[str, float]
    keyword_analysis: Dict[str, Union[float, Dict]]
    predicted_position: float
    feature_scores: Dict[str, float]
    recommendations: Dict[str, List[str]]
    priorities: List[Dict[str, Union[str, float]]]
    industry: str

class SEOAdvisor:
    """Интегрированный SEO советник"""
    def __init__(self, industry: str = 'default'):
        self.industry = industry
        self.rank_predictor = ImprovedRankPredictor(industry=industry)
        self.content_analyzer = ContentAnalyzer()
        self.suggester = Suggester()
        self.analysis_history = []
    
    def analyze_content(self, content: str, target_keywords: List[str]) -> SEOAnalysisReport:
        """Полный анализ контента с рекомендациями"""
        content_metrics = self.content_analyzer.analyze_text(content)
        keyword_analysis = self.content_analyzer.extract_keywords(content, target_keywords)
        
        prediction_features = {
            'keyword_density': keyword_analysis['density'],
            'content_length': content_metrics['word_count'],
            'readability_score': content_metrics['readability'],
            'meta_tags_score': content_metrics['meta_score'],
            'header_structure_score': content_metrics['header_score'],
            'multimedia_score': content_metrics['multimedia_score'],
            'internal_linking_score': content_metrics['linking_score']
        }
        
        prediction = self.rank_predictor.predict_position(prediction_features)
        base_recommendations = self.rank_predictor.generate_recommendations(prediction_features)
        
        enhanced_recommendations = self.suggester.generate_suggestions(
            base_recommendations,
            prediction['feature_scores'],
            self.industry
        )
        
        priorities = self.suggester.prioritize_tasks(
            enhanced_recommendations,
            prediction['feature_scores'],
            prediction['weighted_scores']
        )
        
        report = SEOAnalysisReport(
            timestamp=datetime.now(),
            content_metrics=content_metrics,
            keyword_analysis=keyword_analysis,
            predicted_position=prediction['position'],
            feature_scores=prediction['feature_scores'],
            recommendations=enhanced_recommendations,
            priorities=priorities,
            industry=self.industry
        )
        
        self.analysis_history.append({
            'timestamp': report.timestamp,
            'position': report.predicted_position,
            'content_length': content_metrics['word_count'],
            'keyword_density': keyword_analysis['density']
        })
        
        return report
    
    def generate_improvement_plan(self, report: SEOAnalysisReport) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Генерация плана улучшений с улучшенной приоритизацией"""
        high_priority = []
        medium_priority = []
        low_priority = []
        
        metrics_weights = {
            'content_length': {
                'weight': 0.4,
                'is_critical': report.content_metrics['word_count'] < 500,
                'value': report.content_metrics['word_count'],
                'target': 1250
            },
            'readability': {
                'weight': 0.3,
                'is_critical': report.content_metrics['readability'] < 30,
                'value': report.content_metrics['readability'],
                'target': 70
            },
            'keyword_density': {
                'weight': 0.3,
                'is_critical': report.keyword_analysis['density'] < 0.01 or report.keyword_analysis['density'] > 0.04,
                'value': report.keyword_analysis['density'],
                'target': 0.02
            }
        }
        
        for task in report.priorities:
            base_impact = task['impact']
            task_text = task['task'].lower()
            adjusted_impact = base_impact
            
            for metric, info in metrics_weights.items():
                if metric in task_text:
                    if info['is_critical']:
                        gap = abs(info['target'] - info['value']) / info['target']
                        adjusted_impact = base_impact * (1 + gap) * info['weight'] * 2
                    break
            
            if adjusted_impact > 0.2:
                high_priority.append({
                    'task': task['task'],
                    'impact': round(adjusted_impact, 2),
                    'deadline': '3 дня',
                    'priority_score': task['priority_score']
                })
            elif adjusted_impact > 0.1:
                medium_priority.append({
                    'task': task['task'],
                    'impact': round(adjusted_impact, 2),
                    'deadline': '1 неделя',
                    'priority_score': task['priority_score']
                })
            else:
                low_priority.append({
                    'task': task['task'],
                    'impact': round(adjusted_impact, 2),
                    'deadline': '2 недели',
                    'priority_score': task['priority_score']
                })
        
        if len(high_priority) > 5:
            excess = high_priority[5:]
            high_priority = high_priority[:5]
            medium_priority = excess + medium_priority
        
        for priority_list in [high_priority, medium_priority, low_priority]:
            priority_list.sort(key=lambda x: x['impact'], reverse=True)
        
        return {
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority
        }
    
    def get_history_analytics(self) -> Dict[str, Union[float, int]]:
        """Получение аналитики по истории анализов"""
        if not self.analysis_history:
            return {}
            
        positions = [record['position'] for record in self.analysis_history]
        densities = [record['keyword_density'] for record in self.analysis_history]
        lengths = [record['content_length'] for record in self.analysis_history]
        
        return {
            'analyses_count': len(self.analysis_history),
            'avg_position': sum(positions) / len(positions),
            'best_position': min(positions),
            'worst_position': max(positions),
            'avg_keyword_density': sum(densities) / len(densities),
            'avg_content_length': sum(lengths) / len(lengths)
        }
