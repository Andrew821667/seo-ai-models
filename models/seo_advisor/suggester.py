
from typing import Dict, List, Optional, Tuple
from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor

class Suggester:
    def __init__(self, industry: str = 'default'):
        """
        Инициализация Suggester
        
        Args:
            industry (str): Тип индустрии ('default', 'blog', 'scientific_blog', 'ecommerce')
        """
        self.rank_predictor = ImprovedRankPredictor(industry=industry)
        self.industry = industry
        
    def analyze_content(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Анализ контента и генерация подробных рекомендаций
        
        Args:
            features (Dict[str, float]): Характеристики страницы
            
        Returns:
            Dict[str, any]: Результаты анализа и рекомендации
        """
        # Получаем базовый прогноз и рекомендации
        prediction = self.rank_predictor.predict_position(features)
        base_recommendations = self.rank_predictor.generate_recommendations(features)
        
        # Расширяем анализ
        detailed_analysis = self._perform_detailed_analysis(features, prediction)
        priority_tasks = self._generate_priority_tasks(detailed_analysis)
        competitor_insights = self._generate_competitor_insights(features)
        
        return {
            'current_position': prediction['position'],
            'score_analysis': detailed_analysis,
            'priority_tasks': priority_tasks,
            'base_recommendations': base_recommendations,
            'competitor_insights': competitor_insights
        }
    
    def _perform_detailed_analysis(
        self, 
        features: Dict[str, float], 
        prediction: Dict[str, any]
    ) -> Dict[str, Dict[str, any]]:
        """
        Подробный анализ всех факторов
        """
        analysis = {}
        feature_scores = prediction['feature_scores']
        weighted_scores = prediction['weighted_scores']
        
        for feature_name, score in feature_scores.items():
            current_value = features[feature_name]
            thresholds = getattr(self.rank_predictor.thresholds, feature_name)
            
            status = self._get_feature_status(feature_name, current_value, thresholds)
            impact = weighted_scores[feature_name] / prediction['total_score']
            
            analysis[feature_name] = {
                'current_value': current_value,
                'score': score,
                'weighted_score': weighted_scores[feature_name],
                'impact_percentage': impact * 100,
                'status': status,
                'thresholds': thresholds
            }
            
        return analysis
    
    def _get_feature_status(
        self, 
        feature_name: str, 
        value: float, 
        thresholds: Dict[str, float]
    ) -> str:
        """
        Определение статуса параметра
        """
        if feature_name == 'content_length':
            if thresholds['optimal_min'] <= value <= thresholds['optimal_max']:
                return 'optimal'
            elif value < thresholds['low']:
                return 'critical'
            elif value < thresholds['optimal_min']:
                return 'needs_improvement'
            elif value > thresholds['optimal_max']:
                return 'excessive'
        else:
            if value < thresholds['low']:
                return 'below_threshold'
            elif value > thresholds['high']:
                return 'above_threshold'
            else:
                return 'optimal'
                
        return 'unknown'
    
    def _generate_priority_tasks(
        self, 
        detailed_analysis: Dict[str, Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Генерация приоритетных задач на основе анализа
        """
        tasks = []
        
        # Сортируем факторы по их влиянию и статусу
        factors = [(name, data) for name, data in detailed_analysis.items()]
        factors.sort(key=lambda x: (
            x[1]['status'] != 'critical',  # Критические первые
            x[1]['status'] != 'needs_improvement',  # Потом требующие улучшения
            -x[1]['impact_percentage']  # Затем по влиянию (по убыванию)
        ))
        
        for feature_name, data in factors:
            if data['status'] in ['critical', 'needs_improvement', 'below_threshold']:
                task = self._create_improvement_task(feature_name, data)
                if task:
                    tasks.append(task)
                    
        return tasks
    
    def _create_improvement_task(
        self, 
        feature_name: str, 
        data: Dict[str, any]
    ) -> Optional[Dict[str, any]]:
        """
        Создание конкретной задачи по улучшению
        """
        task_templates = {
            'content_length': {
                'critical': {
                    'title': 'Критически низкий объем контента',
                    'description': f'Текущий объем ({data["current_value"]:.0f} слов) значительно ниже минимального порога. '
                                 f'Необходимо увеличить до как минимум {data["thresholds"]["low"]} слов.',
                    'priority': 'high'
                },
                'needs_improvement': {
                    'title': 'Недостаточный объем контента',
                    'description': f'Рекомендуется увеличить объем контента с {data["current_value"]:.0f} '
                                 f'до {data["thresholds"]["optimal_min"]} слов для оптимальных результатов.',
                    'priority': 'medium'
                }
            },
            'keyword_density': {
                'below_threshold': {
                    'title': 'Низкая плотность ключевых слов',
                    'description': f'Увеличьте плотность ключевых слов с {data["current_value"]*100:.1f}% '
                                 f'до {data["thresholds"]["low"]*100:.1f}%.',
                    'priority': 'medium'
                }
            },
            'readability_score': {
                'below_threshold': {
                    'title': 'Низкая читабельность текста',
                    'description': 'Упростите текст для лучшего восприятия.',
                    'priority': 'medium'
                }
            }
        }
        
        if feature_name in task_templates and data['status'] in task_templates[feature_name]:
            template = task_templates[feature_name][data['status']]
            return {
                'feature': feature_name,
                'title': template['title'],
                'description': template['description'],
                'priority': template['priority'],
                'impact': data['impact_percentage']
            }
            
        return None
    
    def _generate_competitor_insights(self, features: Dict[str, float]) -> List[str]:
        """
        Генерация инсайтов на основе сравнения с конкурентами
        """
        # В будущем здесь будет реальное сравнение с конкурентами
        # Пока возвращаем базовые рекомендации для индустрии
        if self.industry == 'blog':
            return [
                "У успешных блогов в вашей нише среднее время чтения составляет 7 минут",
                "Рекомендуется использовать больше визуального контента",
                "Популярные блоги используют структуру с 3-4 подзаголовками"
            ]
        elif self.industry == 'ecommerce':
            return [
                "Успешные конкуренты используют расширенные описания продуктов (300+ слов)",
                "Рекомендуется добавить секцию FAQ для ключевых продуктов",
                "Используйте больше качественных изображений продукта"
            ]
        elif self.industry == 'scientific_blog':
            return [
                "Добавьте больше ссылок на исследования и источники",
                "Используйте графики и диаграммы для визуализации данных",
                "Включите методологию исследования в контент"
            ]
        
        return [
            "Добавьте больше уникального контента",
            "Улучшите структуру заголовков",
            "Используйте больше релевантных ключевых слов"
        ]
