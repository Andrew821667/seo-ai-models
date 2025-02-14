
from typing import Dict, List, Optional, Tuple
from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor

class Suggester:
    def __init__(self, industry: str = 'default'):
        self.rank_predictor = ImprovedRankPredictor(industry=industry)
        self.industry = industry
        
    def analyze_content(self, features: Dict[str, float]) -> Dict[str, any]:
        prediction = self.rank_predictor.predict_position(features)
        base_recommendations = self.rank_predictor.generate_recommendations(features)
        
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
    
    def _perform_detailed_analysis(self, features: Dict[str, float], prediction: Dict[str, any]) -> Dict[str, Dict[str, any]]:
        analysis = {}
        feature_scores = prediction['feature_scores']
        
        # Используем таблицу весов из rank_predictor
        weights_dict = {
            'keyword_density': self.rank_predictor.weights.keyword_density,
            'content_length': self.rank_predictor.weights.content_length,
            'readability_score': self.rank_predictor.weights.readability_score,
            'meta_tags_score': self.rank_predictor.weights.meta_tags_score,
            'header_structure_score': self.rank_predictor.weights.header_structure_score,
            'backlinks_count': self.rank_predictor.weights.backlinks_count,
            'multimedia_score': self.rank_predictor.weights.multimedia_score,
            'internal_linking_score': self.rank_predictor.weights.internal_linking_score
        }
        
        for feature_name, score in feature_scores.items():
            if feature_name not in features:
                continue
            
            current_value = features[feature_name]
            thresholds = getattr(self.rank_predictor.thresholds, feature_name)
            
            status = self._get_feature_status(feature_name, current_value, thresholds)
            impact = weights_dict.get(feature_name, 0.1)
            
            analysis[feature_name] = {
                'current_value': current_value,
                'score': score,
                'impact_percentage': impact * 100,
                'status': status,
                'thresholds': thresholds
            }
            
        return analysis
    
    def _get_feature_status(self, feature_name: str, value: float, thresholds: Dict[str, float]) -> str:
        def check_value_range(value, low, high):
            if low is None or high is None:
                return 'unknown'
            if value < low:
                return 'below_threshold'
            elif value > high:
                return 'above_threshold'
            return 'optimal'
        
        # Обработка специальных случаев для content_length
        if feature_name == 'content_length':
            low = thresholds.get('low')
            high = thresholds.get('high')
            optimal_min = thresholds.get('optimal_min')
            optimal_max = thresholds.get('optimal_max')
            
            if low and high and optimal_min and optimal_max:
                if value < low:
                    return 'critical'
                elif value < optimal_min:
                    return 'needs_improvement'
                elif value > optimal_max:
                    return 'excessive'
                else:
                    return 'optimal'
        
        # Общий случай для остальных параметров
        low = thresholds.get('low')
        high = thresholds.get('high')
        return check_value_range(value, low, high)
                
    def _generate_priority_tasks(self, detailed_analysis: Dict[str, Dict[str, any]]) -> List[Dict[str, any]]:
        tasks = []
        
        # Сортировка факторов по важности и статусу
        factors = [(name, data) for name, data in detailed_analysis.items()]
        factors.sort(key=lambda x: (
            x[1]['status'] != 'critical',
            x[1]['status'] != 'needs_improvement',
            x[1]['status'] != 'below_threshold',
            -x[1]['impact_percentage']
        ))
        
        for feature_name, data in factors:
            if data['status'] in ['critical', 'needs_improvement', 'below_threshold', 'above_threshold']:
                task = self._create_improvement_task(feature_name, data)
                if task:
                    tasks.append(task)
                    
        return tasks
    
    def _create_improvement_task(self, feature_name: str, data: Dict[str, any]) -> Optional[Dict[str, any]]:
        task_templates = {
            'content_length': {
                'critical': {
                    'title': 'Критически низкий объем контента',
                    'description': lambda d: f'Текущий объем ({d["current_value"]:.0f} слов) значительно ниже минимального порога. '
                                           f'Необходимо увеличить до как минимум {d["thresholds"].get("low", 850)} слов.',
                    'priority': 'high'
                },
                'needs_improvement': {
                    'title': 'Недостаточный объем контента',
                    'description': lambda d: f'Рекомендуется увеличить объем контента с {d["current_value"]:.0f} '
                                           f'до {d["thresholds"].get("optimal_min", d["thresholds"].get("low", 950))} слов для оптимальных результатов.',
                    'priority': 'medium'
                },
                'excessive': {
                    'title': 'Избыточный объем контента',
                    'description': lambda d: f'Рекомендуется сократить объем контента с {d["current_value"]:.0f} '
                                           f'до {d["thresholds"].get("high", d["thresholds"].get("optimal_max", 2400))} слов.',
                    'priority': 'medium'
                }
            },
            'keyword_density': {
                'below_threshold': {
                    'title': 'Низкая плотность ключевых слов',
                    'description': lambda d: f'Увеличьте плотность ключевых слов с {d["current_value"]*100:.1f}% '
                                           f'до {d["thresholds"]["low"]*100:.1f}%.',
                    'priority': 'medium'
                },
                'above_threshold': {
                    'title': 'Избыточная плотность ключевых слов',
                    'description': lambda d: f'Уменьшите плотность ключевых слов с {d["current_value"]*100:.1f}% '
                                           f'до {d["thresholds"]["high"]*100:.1f}%.',
                    'priority': 'high'
                }
            },
            'readability_score': {
                'below_threshold': {
                    'title': 'Низкая читабельность текста',
                    'description': lambda d: f'Текущий показатель читабельности ({d["current_value"]:.0f}) ниже рекомендуемого. '
                                           f'Необходимо повысить до {d["thresholds"].get("low", 45)}.',
                    'priority': 'high'
                }
            },
            'meta_tags_score': {
                'below_threshold': {
                    'title': 'Улучшите мета-теги',
                    'description': lambda d: 'Оптимизируйте title, description и другие мета-теги.',
                    'priority': 'medium'
                }
            },
            'header_structure_score': {
                'below_threshold': {
                    'title': 'Улучшите структуру заголовков',
                    'description': lambda d: 'Используйте более четкую иерархию заголовков H1-H6.',
                    'priority': 'medium'
                }
            }
        }
        
        if (feature_name in task_templates and 
            data['status'] in task_templates[feature_name]):
            template = task_templates[feature_name][data['status']]
            return {
                'feature': feature_name,
                'title': template['title'],
                'description': template['description'](data),
                'priority': template['priority'],
                'impact': data['impact_percentage']
            }
            
        return None
    
    def _generate_competitor_insights(self, features: Dict[str, float]) -> List[str]:
        if self.industry == 'blog':
            return [
                "У успешных блогов в вашей нише среднее время чтения составляет 7 минут",
                "Рекомендуется использовать больше визуального контента",
                "Популярные блоги используют структуру с 3-4 подзаголовками"
            ]
        elif self.industry == 'scientific_blog':
            return [
                "Добавьте больше ссылок на исследования и источники",
                "Используйте графики и диаграммы для визуализации данных",
                "Включите методологию исследования в контент"
            ]
        elif self.industry == 'ecommerce':
            return [
                "Успешные конкуренты используют расширенные описания продуктов (300+ слов)",
                "Рекомендуется добавить секцию FAQ для ключевых продуктов",
                "Используйте больше качественных изображений продукта"
            ]
        
        return [
            "Добавьте больше уникального контента",
            "Улучшите структуру заголовков",
            "Используйте больше релевантных ключевых слов"
        ]
```

Ключевые изменения:
1. Сохранена полная структура вашего оригинального кода
2. Исправлен метод `_create_improvement_task` с использованием `.get()` 
3. Добавлена безопасная обработка различных сценариев
4. Сохранены все оригинальные методы и логика

### Рекомендации по использованию:

1. Замените полностью файл `suggester.py` этим кодом
2. Убедитесь, что импорт `ImprovedRankPredictor` корректен
3. Для отладки можете добавить дополнительный вывод в методах

### Пример тестового скрипта:

```python
# Пример использования
def test_suggester():
    # Тест 1: Блог с недостаточным контентом
    blog_suggester = Suggester(industry='blog')
    blog_features = {
        'keyword_density': 0.015,
        'content_length': 500,
        'readability_score': 65,
        'meta_tags_score': 0.6,
        'header_structure_score': 0.7,
        'backlinks_count': 25,
        'multimedia_score': 0.4,
        'internal_linking_score': 0.5
    }
    
    analysis = blog_suggester.analyze_content(blog_features)
    print("=== Анализ блога ===")
    print(f"Текущая позиция: {analysis['current_position']}")
    
    print("\nПриоритетные задачи:")
    for task in analysis['priority_tasks']:
        print(f"- {task['title']} (Приоритет: {task['priority']})")

# Запуск теста
test_suggester()
```

Этот код должен работать корректно и обрабатывать различные сценарии без ошибок.

Хотите, чтобы я помог вам с интеграцией и тестированием?
