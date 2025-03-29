
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
        
        # ДОБАВЛЕНО: Словарь отраслей YMYL
        self.ymyl_industries = {
            'finance': True,
            'health': True,
            'law': True,
            'insurance': True,
            'medical': True
        }
    
    
    
    def predict_position(self, features: Dict[str, float], text: str = None, keywords: List[str] = None) -> Dict:
        """
        Предсказание позиции с учетом конкурентности ниши и E-E-A-T метрик
        
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
        
        # Рассчитываем общий скор (базовый контентный скор)
        total_score = sum(result['weighted_scores'].values())
        
        # Более консервативные коэффициенты для E-E-A-T метрик
        eeat_adjustment = 0.0
        is_ymyl = self.ymyl_industries.get(self.industry, False)
        
        # Для диагностики будем отслеживать каждый компонент
        eeat_components = {}
        
        if 'expertise_score' in features:
            # Значение экспертности для YMYL и не-YMYL
            eeat_multiplier = 0.5 if is_ymyl else 0.2
            expertise_contribution = features['expertise_score'] * eeat_multiplier
            eeat_adjustment += expertise_contribution
            eeat_components['expertise'] = expertise_contribution
            
        if 'authority_score' in features:
            # Значение авторитетности для YMYL и не-YMYL
            eeat_multiplier = 0.5 if is_ymyl else 0.2
            authority_contribution = features['authority_score'] * eeat_multiplier
            eeat_adjustment += authority_contribution
            eeat_components['authority'] = authority_contribution
            
        if 'trust_score' in features:
            # Значение доверия для YMYL и не-YMYL
            eeat_multiplier = 0.7 if is_ymyl else 0.3
            trust_contribution = features['trust_score'] * eeat_multiplier
            eeat_adjustment += trust_contribution
            eeat_components['trust'] = trust_contribution
            
        if 'overall_eeat_score' in features:
            # Используем общий E-E-A-T только если индивидуальные метрики не предоставлены
            if not any(k in features for k in ['expertise_score', 'authority_score', 'trust_score']):
                eeat_multiplier = 1.0 if is_ymyl else 0.4
                overall_contribution = features['overall_eeat_score'] * eeat_multiplier
                eeat_adjustment += overall_contribution
                eeat_components['overall'] = overall_contribution
        
        # НОВЫЙ ПОДХОД: Линейная зависимость вместо экспоненциальной
        # Чем выше eeat_adjustment, тем лучше позиция (ниже число)
        
        # Базовое значение позиции на основе общего скора
        base_position = 50 * (1 - (total_score / competition_factor))
        
        # Корректировка базовой позиции на основе E-E-A-T
        # Для YMYL отраслей E-E-A-T может улучшить позицию до 40 пунктов
        # Для не-YMYL отраслей - до 20 пунктов
        eeat_position_improvement = eeat_adjustment * (40 if is_ymyl else 20)
        
        # Позиция с учетом E-E-A-T
        adjusted_position = max(1, base_position - eeat_position_improvement)
        
        # Финальная позиция не должна быть хуже 50
        position = min(adjusted_position, 50)
        
        # Обновляем результат
        result['position'] = position
        result['competition_factor'] = competition_factor
        result['total_score'] = total_score
        result['base_position'] = base_position
        result['eeat_adjustment'] = eeat_adjustment
        result['eeat_components'] = eeat_components
        result['eeat_position_improvement'] = eeat_position_improvement
        
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
        
        # ДОБАВЛЕНО: Для YMYL отраслей приоритизируем E-E-A-T рекомендации
        is_ymyl = self.ymyl_industries.get(self.industry, False)
        if is_ymyl:
            if 'eeat_recommendations' in features and features['eeat_recommendations']:
                if 'eeat' not in recommendations:
                    recommendations['eeat'] = []
                
                for rec in features['eeat_recommendations']:
                    recommendations['eeat'].append(f"ВЫСОКИЙ ПРИОРИТЕТ: {rec}")
        
        # Добавляем приоритет рекомендациям
        for feature in recommendations:
            if feature in important_features:
                recommendations[feature] = [
                    f"ПРИОРИТЕТ: {rec}" for rec in recommendations[feature]
                ]
        
        return recommendations
