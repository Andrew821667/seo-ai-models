
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor
from models.seo_advisor.content_analyzer import ContentAnalyzer
from models.seo_advisor.suggester import Suggester

@dataclass
class ContentQualityReport:
   """Отчет о качестве контента"""
   content_scores: Dict[str, float]
   strengths: List[str]
   weaknesses: List[str]
   potential_improvements: List[str]

@dataclass
class SEOAnalysisReport:
   """Расширенный отчет по SEO анализу"""
   timestamp: datetime
   content_metrics: Dict[str, float]
   keyword_analysis: Dict[str, Union[float, Dict]]
   predicted_position: float
   feature_scores: Dict[str, float]
   content_quality: ContentQualityReport
   recommendations: Dict[str, List[str]]
   priorities: List[Dict[str, Union[str, float]]]
   industry: str

class SEOAdvisor:
   """Улучшенный SEO советник с расширенной аналитикой"""
   
   def __init__(self, industry: str = 'default'):
       self.industry = industry
       self.rank_predictor = ImprovedRankPredictor(industry=industry)
       self.content_analyzer = ContentAnalyzer()
       self.suggester = Suggester()
       self.analysis_history = []
   
   def analyze_content(self, content: str, target_keywords: List[str]) -> SEOAnalysisReport:
       """Комплексный анализ контента с оценкой качества"""
       # Получаем расширенные метрики
       content_metrics = self.content_analyzer.analyze_text(content)
       keyword_analysis = self.content_analyzer.extract_keywords(content, target_keywords)
       
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
           'engagement_potential': content_metrics.get('engagement_potential', 0)
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
           industry=self.industry
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
       
       # Анализ длины контента
       word_count = metrics.get('word_count', 0)
       if word_count < 500:
           weaknesses.append("Недостаточная длина контента")
           improvements.extend([
               "Увеличьте объем контента минимум до 1000 слов",
               "Добавьте больше практических примеров",
               "Расширьте каждый раздел дополнительными деталями"
           ])
       elif word_count > 1500:
           strengths.append("Достаточный объем контента")
       
       # Анализ читабельности
       readability = metrics.get('readability', 0)
       if readability < 40:
           weaknesses.append("Низкая читабельность текста")
           improvements.extend([
               "Используйте более короткие предложения",
               "Добавьте подзаголовки для лучшей структуры",
               "Разбейте длинные параграфы"
           ])
       elif readability > 70:
           strengths.append("Хорошая читабельность текста")
       
       # Анализ технических аспектов
       meta_score = metrics.get('meta_score', 0)
       if meta_score > 0.7:
           strengths.append("Хорошая мета-оптимизация")
       elif meta_score < 0.5:
           weaknesses.append("Слабая мета-оптимизация")
           improvements.extend([
               "Оптимизируйте мета-описания",
               "Улучшите заголовки страницы",
               "Добавьте alt-теги для изображений"
           ])
       
       # Анализ вовлеченности
       engagement = metrics.get('engagement_potential', 0)
       if engagement < 0.3:
           weaknesses.append("Низкий уровень вовлечения")
           improvements.extend([
               "Добавьте призывы к действию",
               "Включите интерактивные элементы",
               "Задайте вопросы читателям"
           ])
       
       # Анализ ключевых слов
       keyword_density = keyword_analysis.get('density', 0)
       if keyword_density < 0.01:
           weaknesses.append("Недостаточная плотность ключевых слов")
           improvements.extend([
               "Увеличьте частоту использования ключевых слов",
               "Добавьте синонимы ключевых слов",
               "Используйте ключевые слова в заголовках"
           ])
       elif keyword_density > 0.04:
           weaknesses.append("Слишком высокая плотность ключевых слов")
           improvements.append("Уменьшите частоту использования ключевых слов")
       
       # Рассчитываем общие скоры
       scores = {
           'overall_quality': self._calculate_overall_quality(metrics),
           'content_depth': metrics.get('semantic_depth', 0),
           'user_engagement': metrics.get('engagement_potential', 0),
           'technical_seo': (meta_score + metrics.get('header_score', 0)) / 2,
           'readability_score': readability / 100 if readability else 0,
           'keyword_optimization': min(max((keyword_density - 0.01) * 25, 0), 1) if keyword_density else 0
       }
       
       return ContentQualityReport(
           content_scores=scores,
           strengths=strengths,
           weaknesses=weaknesses,
           potential_improvements=improvements
       )
   
   def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
       """Расчет общего качества контента"""
       weights = {
           'readability': 0.2,
           'meta_score': 0.15,
           'header_score': 0.15,
           'multimedia_score': 0.15,
           'linking_score': 0.15,
           'semantic_depth': 0.1,
           'topic_relevance': 0.1
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
           'overall_quality': report.content_quality.content_scores['overall_quality']
       }
       
       self.analysis_history.append(history_record)
       
       # Ограничиваем историю последними 100 анализами
       if len(self.analysis_history) > 100:
           self.analysis_history = self.analysis_history[-100:]
