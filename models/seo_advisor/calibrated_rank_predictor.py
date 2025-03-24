from typing import Dict, List, Optional
from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor

class CalibratedRankPredictor(ImprovedRankPredictor):
    """
    Откалиброванный предиктор ранжирования с оптимизированными весами и 
    учетом конкурентности ниши для более реалистичных предсказаний
    """
    
    def __init__(self, industry: str = 'default'):
        """
        Инициализация с оптимизированными весами
        
        Args:
            industry: Отрасль для специфичных настроек
        """
        super().__init__(industry)
        
        # Оптимизированные веса (сумма = 1.0)
        self.feature_weights = {
            'keyword_density': 0.030,      # Снижено: меньший акцент на плотность
            'content_length': 0.028,       # Снижено: меньший акцент на объем
            'readability_score': 0.159,    # Увеличено: фокус на пользовательский опыт
            'meta_tags_score': 0.086,      # Увеличено: важность мета-данных
            'header_structure_score': 0.122, # Увеличено: важность структуры 
            'multimedia_score': 0.073,     # Увеличено: значимость мультимедиа
            'internal_linking_score': 0.073, # Увеличено: важность внутренних ссылок
            'topic_relevance': 0.183,      # Увеличено: ключевой фактор релевантности
            'semantic_depth': 0.159,       # Увеличено: значимость семантики
            'engagement_potential': 0.086  # Увеличено: важность вовлечения
        }
        
        # Коэффициенты конкурентности ниш
        self.competition_factors = {
            'electronics': 1.4,    # Высокая конкуренция
            'health': 1.5,         # Очень высокая (YMYL)
            'finance': 1.6,        # Наивысшая (YMYL)
            'travel': 1.2,         # Средняя
            'education': 1.3,      # Средне-высокая
            'ecommerce': 1.35,     # Высокая
            'local_business': 1.0, # Ниже средней
            'blog': 1.1,           # Низкая
            'news': 1.3,           # Средне-высокая
            'default': 1.2         # По умолчанию
        }
    
    def predict_position(self, features: Dict[str, float], text: str = None, keywords: List[str] = None) -> Dict:
        """
        Предсказание позиции с учетом конкурентности ниши
        
        Args:
            features: Словарь с метриками контента
            text: Текст для анализа (опционально)
            keywords: Ключевые слова (опционально)
            
        Returns:
            Словарь с результатами предсказания
        """
        # Базовое предсказание с откалиброванными весами
        result = super().predict_position(features, text, keywords)
        
        # Получаем коэффициент конкурентности для данной отрасли
        competition_factor = self.competition_factors.get(
            self.industry, 
            self.competition_factors['default']
        )
        
        # Рассчитываем общий скор
        total_score = sum(result['weighted_scores'].values())
        
        # Применяем скорректированную формулу с учетом конкурентности
        position = max(1, min(100, 50 * (1 - (total_score / competition_factor) * 1.2)))
        
        # Обновляем результат
        result['position'] = position
        result['competition_factor'] = competition_factor
        result['total_score'] = total_score
        
        # Добавляем вероятности попадания в топ
        result['probability'] = {
            'top10': max(0, min(1, 1.5 - (position / 10))),
            'top20': max(0, min(1, 1.5 - (position / 20))),
            'top50': max(0, min(1, 1.5 - (position / 50)))
        }
        
        return result
        
    def generate_recommendations(self, features: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Генерация рекомендаций с учетом оптимизированных весов
        
        Args:
            features: Словарь с метриками контента
            
        Returns:
            Словарь с рекомендациями по улучшению контента
        """
        recommendations = super().generate_recommendations(features)
        
        # Фокусируемся на рекомендациях для самых важных метрик
        important_features = [
            'topic_relevance',
            'readability_score',
            'semantic_depth',
            'header_structure_score'
        ]
        
        # Добавляем приоритет рекомендациям
        for feature in recommendations:
            if feature in important_features:
                recommendations[feature] = [
                    f"ПРИОРИТЕТ: {rec}" for rec in recommendations[feature]
                ]
        
        return recommendations
