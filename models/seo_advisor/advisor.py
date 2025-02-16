
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
            'readability_score': content_metrics['readability'],
            'meta_tags_score': content_metrics['meta_score'],
            'header_structure_score': content_metrics['header_score'],
            'multimedia_score': content_metrics['multimedia_score'],
            'internal_linking_score': content_metrics['linking_score'],
            'topic_relevance': content_metrics['topic_relevance'],
            'semantic_depth': content_metrics['semantic_depth'],
            'engagement_potential': content_metrics['engagement_potential']
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
        """Оценка качества контента на основе метрик"""
        scores = {}
        strengths = []
        weaknesses = []
        improvements = []
        
        # Оцениваем каждый аспект контента
        if metrics['word_count'] < 500:
            weaknesses.append("Недостаточная длина контента")
            improvements.append("Увеличьте объем контента минимум до 1000 слов")
        elif metrics['word_count'] > 1500:
            strengths.append("Достаточный объем контента")
        
        if metrics['semantic_depth'] > 0.7:
            strengths.append("Высокая семантическая глубина")
        elif metrics['semantic_depth'] < 0.4:
            weaknesses.append("Низкая семантическая глубина")
            improvements.append("Добавьте более специализированную терминологию")
        
        if metrics['topic_relevance'] > 0.8:
            strengths.append("Высокая релевантность теме")
        elif metrics['topic_relevance'] < 0.5:
            weaknesses.append("Низкая релевантность теме")
            improvements.append("Сфокусируйтесь на основной теме")
        
        if metrics['engagement_potential'] > 0.7:
            strengths.append("Высокий потенциал вовлечения")
        elif metrics['engagement_potential'] < 0.4:
            weaknesses.append("Низкий потенциал вовлечения")
            improvements.append("Добавьте больше интерактивных элементов")
        
        # Рассчитываем общие скоры
        scores = {
            'overall_quality': self._calculate_overall_quality(metrics),
            'content_depth': (metrics['semantic_depth'] + metrics['topic_relevance']) / 2,
            'user_engagement': metrics['engagement_potential'],
            'technical_seo': (metrics['meta_score'] + metrics['header_score']) / 2
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
            'semantic_depth': 0.25,
            'topic_relevance': 0.25,
            'engagement_potential': 0.2,
            'content_uniqueness': 0.15,
            'information_density': 0.15
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
    
    def get_history_analytics(self) -> Dict[str, Union[float, List[Dict]]]:
        """Расширенная аналитика по истории анализов"""
        if not self.analysis_history:
            return {}
        
        positions = [record['position'] for record in self.analysis_history]
        qualities = [record['overall_quality'] for record in self.analysis_history]
        
        # Рассчитываем тренды
        position_trend = self._calculate_trend(positions)
        quality_trend = self._calculate_trend(qualities)
        
        return {
            'analyses_count': len(self.analysis_history),
            'avg_position': sum(positions) / len(positions),
            'best_position': min(positions),
            'worst_position': max(positions),
            'avg_quality': sum(qualities) / len(qualities),
            'position_trend': position_trend,
            'quality_trend': quality_trend,
            'latest_analyses': self.analysis_history[-5:]  # Последние 5 анализов
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Расчет тренда для метрики"""
        if len(values) < 2:
            return "neutral"
            
        # Сравниваем среднее первой и второй половины
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid
        second_half_avg = sum(values[mid:]) / (len(values) - mid)
        
        diff = second_half_avg - first_half_avg
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

if __name__ == "__main__":
    # Пример использования
    advisor = SEOAdvisor(industry='blog')
    
    test_content = """
    Machine Learning and AI Development
    
    Artificial Intelligence and Machine Learning are transforming various industries.
    This comprehensive guide explores the latest trends and best practices.
    We'll look at practical examples and case studies.
    
    Key Benefits:
    1. Improved efficiency
    2. Better decision making
    3. Automated processes
    
    Future prospects
    The field continues to evolve rapidly.
    New developments in machine learning show promising results.
    """
    
    target_keywords = ['machine learning', 'artificial intelligence', 'AI trends']
    
    report = advisor.analyze_content(test_content, target_keywords)
    
    print(f"\nPredicted Position: {report.predicted_position:.2f}")
    
    print("\nContent Quality Scores:")
    for metric, score in report.content_quality.content_scores.items():
        print(f"{metric}: {score:.2f}")
    
    print("\nStrengths:")
    for strength in report.content_quality.strengths:
        print(f"- {strength}")
    
    print("\nWeaknesses:")
    for weakness in report.content_quality.weaknesses:
        print(f"- {weakness}")
    
    print("\nRecommended Improvements:")
    for improvement in report.content_quality.potential_improvements:
        print(f"- {improvement}")
