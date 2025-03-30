
import logging
from typing import Dict, List, Optional, Union

# Импортируем необходимые модули
from models.seo_advisor.enhanced_eeat_analyzer import EnhancedEEATAnalyzer

logger = logging.getLogger(__name__)

class EEATIntegrator:
    """Класс для интеграции E-E-A-T анализа в SEO процессы"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация интегратора
        
        Args:
            model_path: Путь к обученной модели E-E-A-T
        """
        # Путь к модели по умолчанию
        if model_path is None:
            model_path = '/content/seo-ai-models/models/checkpoints/eeat_best_model.joblib'
            
        # Инициализация анализатора
        try:
            self.analyzer = EnhancedEEATAnalyzer(model_path=model_path)
            self.model_loaded = True
            logger.info(f"E-E-A-T анализатор инициализирован с моделью из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка инициализации E-E-A-T анализатора: {e}")
            self.model_loaded = False
            self.analyzer = EnhancedEEATAnalyzer()  # Базовый анализатор без модели
    
    def enhance_metrics(self, metrics: Dict[str, float], content: str, industry: str) -> Dict[str, float]:
        """
        Дополнение метрик контента результатами E-E-A-T анализа
        
        Args:
            metrics: Существующие метрики контента
            content: Текстовый контент для анализа
            industry: Отрасль контента
            
        Returns:
            Обновленные метрики с E-E-A-T показателями
        """
        # Выполняем анализ E-E-A-T
        eeat_results = self.analyzer.analyze(content, industry=industry)
        
        # Дополняем метрики результатами анализа
        enhanced_metrics = metrics.copy()
        enhanced_metrics.update({
            'expertise_score': eeat_results['expertise_score'],
            'authority_score': eeat_results['authority_score'],
            'trust_score': eeat_results['trust_score'],
            'semantic_coherence_score': eeat_results['semantic_coherence_score'],
            'overall_eeat_score': eeat_results['overall_eeat_score']
        })
        
        # Если есть структурные метрики, интегрируем их
        if 'structural_score' in eeat_results:
            enhanced_metrics['structural_score'] = eeat_results['structural_score']
        
        # Если есть специфичные для YMYL метрики, добавляем их
        if industry in ['finance', 'health', 'legal', 'medical']:
            enhanced_metrics['ymyl_status'] = 1
            enhanced_metrics['ymyl_category'] = industry
        
        return enhanced_metrics
    
    def get_recommendations(self, content: str, industry: str) -> List[str]:
        """
        Получение рекомендаций по улучшению E-E-A-T
        
        Args:
            content: Текстовый контент для анализа
            industry: Отрасль контента
            
        Returns:
            Список рекомендаций
        """
        # Выполняем анализ
        eeat_results = self.analyzer.analyze(content, industry=industry)
        
        # Возвращаем рекомендации
        return eeat_results.get('recommendations', [])
    
    def get_detailed_analysis(self, content: str, industry: str) -> Dict[str, Union[float, List[str]]]:
        """
        Получение полного анализа E-E-A-T
        
        Args:
            content: Текстовый контент для анализа
            industry: Отрасль контента
            
        Returns:
            Словарь с результатами анализа
        """
        # Возвращаем полный результат анализа
        return self.analyzer.analyze(content, industry=industry)
