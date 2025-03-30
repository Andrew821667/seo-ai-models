
from typing import Dict, List, Union
from dataclasses import dataclass

@dataclass
class SuggestionPriority:
    task: str
    impact: float
    effort: float
    priority_score: float

class Suggester:
    """Генератор рекомендаций по улучшению SEO"""
    
    def __init__(self):
        self.priority_weights = {
            'content_length': 0.25,
            'keyword_density': 0.20,
            'readability': 0.15,
            'header_structure': 0.15,
            'meta_tags': 0.10,
            'multimedia': 0.08,
            'internal_linking': 0.07
        }
        
    def generate_suggestions(
        self, 
        basic_recommendations: Dict[str, List[str]],
        feature_scores: Dict[str, float],
        industry: str
    ) -> Dict[str, List[str]]:
        """
        Генерация расширенных рекомендаций
        """
        enhanced_suggestions = {}
        
        # Улучшаем базовые рекомендации в зависимости от индустрии
        industry_specific = self._get_industry_specific_suggestions(industry)
        
        for feature, suggestions in basic_recommendations.items():
            score = feature_scores.get(feature, 0)
            enhanced = suggestions.copy()
            
            # Добавляем индустри-специфичные рекомендации
            if feature in industry_specific:
                enhanced.extend(industry_specific[feature])
                
            # Добавляем конкретные действия в зависимости от скора
            if score < 0.3:
                enhanced.extend(self._get_critical_suggestions(feature))
            elif score < 0.6:
                enhanced.extend(self._get_improvement_suggestions(feature))
            
            enhanced_suggestions[feature] = enhanced
            
        return enhanced_suggestions
    
    def prioritize_tasks(
        self,
        suggestions: Dict[str, List[str]],
        feature_scores: Dict[str, float],
        weighted_scores: Dict[str, float]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Улучшенная приоритизация задач
        """
        priorities = []
        
        for feature, tasks in suggestions.items():
            feature_score = feature_scores.get(feature, 0)
            weighted_score = weighted_scores.get(feature, 0)
            weight = self.priority_weights.get(feature, 0)
            
            for task in tasks:
                # Оцениваем сложность задачи
                effort = self._estimate_task_effort(task)
                
                # Улучшенный расчет импакта
                impact = (1 - feature_score) * weight * 2  # Увеличиваем влияние
                
                # Бонус за критически важные метрики
                if feature in ['content_length', 'keyword_density'] and feature_score < 0.3:
                    impact *= 1.5
                    
                # Считаем приоритет с учетом срочности
                priority_score = (impact * 1.5) / effort
                if feature_score < 0.3:
                    priority_score *= 1.3  # Бонус за критические показатели
                
                priorities.append({
                    'task': task,
                    'impact': impact,
                    'effort': effort,
                    'priority_score': priority_score,
                    'feature': feature
                })
        
        # Сортируем по приоритету
        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)
    
    def _get_industry_specific_suggestions(self, industry: str) -> Dict[str, List[str]]:
        """
        Получение рекомендаций специфичных для индустрии
        """
        suggestions = {
            'blog': {
                'content_length': [
                    'Добавьте примеры из практики',
                    'Включите экспертные мнения',
                    'Разбейте текст на тематические секции'
                ],
                'readability': [
                    'Используйте больше подзаголовков',
                    'Добавьте маркированные списки',
                    'Включите определения терминов'
                ]
            },
            'scientific_blog': {
                'content_length': [
                    'Добавьте методологию исследования',
                    'Включите статистические данные',
                    'Опишите ограничения исследования'
                ],
                'readability': [
                    'Добавьте графики и диаграммы',
                    'Включите сравнительные таблицы',
                    'Объясните сложные термины'
                ]
            },
            'ecommerce': {
                'content_length': [
                    'Расширьте описание характеристик',
                    'Добавьте сравнение с аналогами',
                    'Включите отзывы пользователей'
                ],
                'multimedia': [
                    'Добавьте больше фотографий продукта',
                    'Включите видео-обзор',
                    'Добавьте инфографику'
                ]
            }
        }
        
        return suggestions.get(industry, {})
    
    def _get_critical_suggestions(self, feature: str) -> List[str]:
        """
        Получение критических рекомендаций для низких показателей
        """
        critical_suggestions = {
            'content_length': [
                'СРОЧНО: Расширьте контент минимум в 2 раза',
                'КРИТИЧНО: Добавьте детальное описание каждого аспекта'
            ],
            'keyword_density': [
                'СРОЧНО: Добавьте ключевые слова в заголовки',
                'КРИТИЧНО: Увеличьте частоту использования ключевых слов'
            ],
            'readability': [
                'СРОЧНО: Упростите сложные предложения',
                'КРИТИЧНО: Добавьте определения терминов'
            ]
        }
        
        return critical_suggestions.get(feature, [])
    
    def _get_improvement_suggestions(self, feature: str) -> List[str]:
        """
        Получение рекомендаций по улучшению средних показателей
        """
        improvement_suggestions = {
            'content_length': [
                'Добавьте дополнительные примеры',
                'Расширьте описание ключевых моментов'
            ],
            'keyword_density': [
                'Добавьте больше LSI-ключевых слов',
                'Используйте синонимы ключевых слов'
            ],
            'readability': [
                'Добавьте подзаголовки для структурирования',
                'Используйте маркированные списки'
            ]
        }
        
        return improvement_suggestions.get(feature, [])
    
    def _estimate_task_effort(self, task: str) -> float:
        """
        Оценка сложности выполнения задачи
        """
        # Улучшенная эвристика для оценки сложности
        effort = 0.5  # Базовая сложность
        
        if 'СРОЧНО' in task:
            effort += 0.4
        elif 'КРИТИЧНО' in task:
            effort += 0.3
        if 'добавьте' in task.lower():
            effort += 0.2
        if 'расширьте' in task.lower():
            effort += 0.2
        if 'создайте' in task.lower():
            effort += 0.3
            
        return min(1.0, effort)
