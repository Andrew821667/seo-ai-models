"""Основной класс SEO Advisor для анализа контента и оптимизации.

Этот класс предоставляет комплексный анализ контента с точки зрения SEO,
включая оценку качества контента, анализ ключевых слов, E-E-A-T метрики,
и предсказание позиций в поисковой выдаче.

Attributes:
    industry (str): Отрасль контента для специфичного анализа.
    rank_predictor (CalibratedRankPredictor): Предиктор ранжирования.
    content_analyzer (ContentAnalyzer): Анализатор контента.
    suggester (Suggester): Генератор рекомендаций.
    semantic_analyzer (SemanticAnalyzer): Семантический анализатор.
    eeat_analyzer (EEATAnalyzer): Анализатор E-E-A-T.
    analysis_history (list): История предыдущих анализов.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass

# Относительные импорты внутри модуля
from .analyzers.content_analyzer import ContentAnalyzer
from .analyzers.semantic_analyzer import SemanticAnalyzer
from .analyzers.eeat.eeat_analyzer import EEATAnalyzer
from .predictors.calibrated_rank_predictor import CalibratedRankPredictor
from .suggester.suggester import Suggester

@dataclass
class ContentQualityReport:
   """Отчет о качестве контента.
    
    Dataclass для хранения результатов анализа качества контента.
    
    Attributes:
        content_scores (Dict[str, float]): Оценки различных аспектов контента.
        strengths (List[str]): Сильные стороны контента.
        weaknesses (List[str]): Слабые стороны контента.
        potential_improvements (List[str]): Рекомендации по улучшению.
    """
   content_scores: Dict[str, float]
   strengths: List[str]
   weaknesses: List[str]
   potential_improvements: List[str]

@dataclass
class SEOAnalysisReport:
   """Расширенный отчет по SEO анализу.
    
    Dataclass для хранения результатов полного SEO анализа.
    
    Attributes:
        timestamp (datetime): Время выполнения анализа.
        content_metrics (Dict[str, float]): Метрики контента.
        keyword_analysis (Dict[str, Union[float, Dict]]): Анализ ключевых слов.
        predicted_position (float): Предсказанная позиция в выдаче.
        feature_scores (Dict[str, float]): Оценки отдельных факторов.
        content_quality (ContentQualityReport): Отчет о качестве контента.
        recommendations (Dict[str, List[str]]): Рекомендации по категориям.
        priorities (List[Dict[str, Union[str, float]]]): Задачи по приоритету.
        industry (str): Отрасль контента.
        position_probabilities (Optional[Dict[str, float]]): Вероятности позиций.
    """
   timestamp: datetime
   content_metrics: Dict[str, float]
   keyword_analysis: Dict[str, Union[float, Dict]]
   predicted_position: float
   feature_scores: Dict[str, float]
   content_quality: ContentQualityReport
   recommendations: Dict[str, List[str]]
   priorities: List[Dict[str, Union[str, float]]]
   industry: str
   position_probabilities: Optional[Dict[str, float]] = None

class SEOAdvisor:
   """Улучшенный SEO советник с расширенной аналитикой"""
   
   def __init__(self, industry: str = 'default'):
       self.industry = industry
       self.rank_predictor = CalibratedRankPredictor(industry=industry)
       self.content_analyzer = ContentAnalyzer()
       self.suggester = Suggester()
       self.semantic_analyzer = SemanticAnalyzer()
       self.eeat_analyzer = EEATAnalyzer()
       self.analysis_history = []
   
   def analyze_content(self, content: str, target_keywords: List[str]) -> SEOAnalysisReport:
       """Комплексный анализ контента с оценкой качества.
       
       Выполняет полный анализ контента, включая извлечение метрик,
       семантический анализ, анализ E-E-A-T, и генерацию рекомендаций.
       
       Args:
           content (str): Текстовое содержимое для анализа.
           target_keywords (List[str]): Целевые ключевые слова.
           
       Returns:
           SEOAnalysisReport: Полный отчет по SEO анализу.
       """
       # Получаем расширенные метрики
       content_metrics = self.content_analyzer.analyze_text(content)
       keyword_analysis = self.content_analyzer.extract_keywords(content, target_keywords)
       
       # Выполняем семантический анализ
       semantic_analysis = self.semantic_analyzer.analyze_text(content, target_keywords)
       
       # Добавляем семантические метрики в общий анализ
       content_metrics.update({
           "semantic_density": semantic_analysis["semantic_density"],
           "semantic_coverage": semantic_analysis["semantic_coverage"],
           "topical_coherence": semantic_analysis["topical_coherence"],
           "contextual_relevance": semantic_analysis["contextual_relevance"]
       })
       
       # Выполняем анализ E-E-A-T
       eeat_analysis = self.eeat_analyzer.analyze(content)
       
       # Добавляем E-E-A-T метрики в общий анализ
       content_metrics.update({
           "expertise_score": eeat_analysis["expertise_score"],
           "authority_score": eeat_analysis["authority_score"],
           "trust_score": eeat_analysis["trust_score"],
           "overall_eeat_score": eeat_analysis["overall_eeat_score"]
       })
       
       # Анализируем качество контента
       content_quality = self._evaluate_content_quality(content_metrics, keyword_analysis)
       
       # Подготавливаем данные для предсказания
       prediction_features = {
           'keyword_density': keyword_analysis['density'],
           'content_length': content_metrics['word_count'],
           'readability_score': content_metrics.get('readability', 0),
           'meta_tags_score': content_metrics.get('meta_score', 0),
           'header_structure_score': content_metrics.get('header_score', 0),
           'multimedia_score': content_metrics.get('multimedia_score', 0),
           'internal_linking_score': content_metrics.get('linking_score', 0),
           'topic_relevance': content_metrics.get('topic_relevance', 0),
           'semantic_depth': content_metrics.get('semantic_depth', 0),
           'engagement_potential': content_metrics.get('engagement_potential', 0),
           "expertise_score": content_metrics.get("expertise_score", 0),
           "authority_score": content_metrics.get("authority_score", 0),
           "trust_score": content_metrics.get("trust_score", 0),
           "overall_eeat_score": content_metrics.get("overall_eeat_score", 0)
       }
       
       # Получаем предсказание и рекомендации
       prediction = self.rank_predictor.predict_position(prediction_features)
       base_recommendations = self.rank_predictor.generate_recommendations(prediction_features)
       
       # Получаем улучшенные рекомендации
       enhanced_recommendations = self.suggester.generate_suggestions(
           base_recommendations,
           prediction['feature_scores'],
           self.industry
       )
       
       # Получаем семантические рекомендации
       semantic_recommendations = self.semantic_analyzer.generate_recommendations(semantic_analysis)
       
       # Объединяем с рекомендациями от E-E-A-T анализа
       eeat_recommendations = eeat_analysis["recommendations"]
       
       # Добавляем семантические и E-E-A-T рекомендации
       enhanced_recommendations["semantic_optimization"] = semantic_recommendations
       enhanced_recommendations["eeat_improvement"] = eeat_recommendations
       
       # Приоритизируем задачи
       priorities = self.suggester.prioritize_tasks(
           enhanced_recommendations,
           prediction['feature_scores'],
           prediction['weighted_scores']
       )
       
       # Создаем отчет
       report = SEOAnalysisReport(
           timestamp=datetime.now(),
           content_metrics=content_metrics,
           keyword_analysis=keyword_analysis,
           predicted_position=prediction['position'],
           feature_scores=prediction['feature_scores'],
           content_quality=content_quality,
           recommendations=enhanced_recommendations,
           priorities=priorities,
           industry=self.industry,
           position_probabilities=prediction.get('probability', {})
       )
       
       # Сохраняем в историю
       self._update_history(report)
       
       return report
   
   def _evaluate_content_quality(
       self, 
       metrics: Dict[str, float], 
       keyword_analysis: Dict[str, Union[float, Dict]]
   ) -> ContentQualityReport:
       """Оценка качества контента с расширенными рекомендациями"""
       scores = {}
       strengths = []
       weaknesses = []
       improvements = []
       
       # ... (здесь идет оригинальный код метода)
       # Для краткости опустим детали реализации
       
       return ContentQualityReport(
           content_scores=scores,
           strengths=strengths,
           weaknesses=weaknesses,
           potential_improvements=improvements
       )
   
   def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
       """Расчет общего качества контента"""
       weights = {
           'readability': 0.15,
           'meta_score': 0.1,
           'header_score': 0.1,
           'multimedia_score': 0.1,
           'linking_score': 0.1,
           'semantic_depth': 0.07,
           'topic_relevance': 0.1,
           'semantic_density': 0.07,
           'semantic_coverage': 0.06,
           'topical_coherence': 0.05,
           'overall_eeat_score': 0.1
       }
       
       weighted_sum = sum(
           metrics.get(metric, 0) * weight 
           for metric, weight in weights.items()
       )
       
       return min(max(weighted_sum, 0), 1)
       
   def _update_history(self, report: SEOAnalysisReport):
       """Обновление истории анализов"""
       history_record = {
           'timestamp': report.timestamp,
           'position': report.predicted_position,
           'content_length': report.content_metrics['word_count'],
           'keyword_density': report.keyword_analysis['density'],
           'overall_quality': report.content_quality.content_scores.get('overall_quality', 0.5)
       }
       
       self.analysis_history.append(history_record)
       
       # Ограничиваем историю последними 100 анализами
       if len(self.analysis_history) > 100:
           self.analysis_history = self.analysis_history[-100:]
