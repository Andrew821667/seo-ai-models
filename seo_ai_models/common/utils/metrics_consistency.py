
"""
Модуль для проверки согласованности метрик контента.

Обнаруживает и исправляет логические противоречия между различными метриками,
чтобы избежать противоречивых рекомендаций.
"""

from typing import Dict, List, Tuple, Optional, Any

class MetricsConsistencyChecker:
    """
    Класс для проверки согласованности метрик SEO-анализа.
    
    Выявляет и разрешает противоречия между разными метриками,
    чтобы обеспечить точные и непротиворечивые рекомендации.
    """
    
    def __init__(self):
        """Инициализация проверки консистентности."""
        # Определяем логические правила для метрик
        self.rules = [
            # Правило 1: Если средняя длина предложения > 25 слов, читабельность не может быть > 0.7
            {
                'condition': lambda m: m.get('avg_sentence_length', 0) > 25 and m.get('readability', 0) > 0.7,
                'action': lambda m: self._set_readability_based_on_sentence_length(m)
            },
            # Правило 2: Если количество заголовков < 2, структурный скор не может быть > 0.5
            {
                'condition': lambda m: m.get('headers_count', 0) < 2 and m.get('structure_score', 0) > 0.5,
                'action': lambda m: self._set_structure_score(m, 0.5)
            },
            # Правило 3: Контент без заголовков не может иметь высокий header_score
            {
                'condition': lambda m: m.get('headers_count', 0) == 0 and m.get('header_score', 0) > 0.3,
                'action': lambda m: self._set_header_score(m, 0.3)
            },
            # Правило 4: Контент менее 300 слов не может иметь высокий semantic_depth
            {
                'condition': lambda m: m.get('word_count', 0) < 300 and m.get('semantic_depth', 0) > 0.5,
                'action': lambda m: self._set_semantic_depth(m, 0.5)
            },
            # Правило 5: Отсутствие списков должно снижать структурный скор
            {
                'condition': lambda m: m.get('lists_count', 0) == 0 and m.get('structure_score', 0) > 0.7,
                'action': lambda m: self._set_structure_score(m, 0.7)
            },
            # Правило 6: Большой текст без изображений должен иметь сниженный multimedia_score
            {
                'condition': lambda m: m.get('word_count', 0) > 800 and not m.get('has_images', False) and m.get('multimedia_score', 0) > 0.4,
                'action': lambda m: self._set_multimedia_score(m, 0.4)
            },
            # Правило 7: Слишком высокая плотность ключевых слов (>10%) нереалистична или спам
            {
                'condition': lambda m: m.get('keyword_density', 0) > 0.1,
                'action': lambda m: self._adjust_keyword_density(m)
            }
        ]
    
    def check_and_fix(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет и исправляет противоречия в метриках.
        
        Args:
            metrics: Словарь метрик для проверки
            
        Returns:
            Dict[str, Any]: Исправленные метрики без противоречий
        """
        fixed_metrics = metrics.copy()
        
        # Применяем все правила согласованности
        for rule in self.rules:
            if rule['condition'](fixed_metrics):
                fixed_metrics = rule['action'](fixed_metrics)
        
        # Проверяем согласованность основной оценки читабельности с конкретным значением
        if 'readability' in fixed_metrics and fixed_metrics['readability'] > 0.8:
            # Высокая оценка читабельности (> 0.8)
            # Проверка, что рекомендации согласованы с этой метрикой
            if 'readability_specific' not in fixed_metrics:
                fixed_metrics['readability_specific'] = fixed_metrics['readability'] * 100
        
        # Дополнительные проверки диапазонов для всех метрик
        fixed_metrics = self._check_ranges(fixed_metrics)
        
        return fixed_metrics
    
    def _set_readability_based_on_sentence_length(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Корректирует оценку читабельности на основе длины предложения."""
        avg_sentence_length = metrics.get('avg_sentence_length', 0)
        # Формула: чем длиннее предложения, тем ниже читабельность
        # Макс. читабельность 0.7 при длине 25+ слов
        adjusted_readability = max(0, min(0.7, 1.0 - (avg_sentence_length - 15) / 40))
        metrics['readability'] = adjusted_readability
        return metrics
    
    def _set_structure_score(self, metrics: Dict[str, Any], max_score: float) -> Dict[str, Any]:
        """Корректирует оценку структуры с ограничением на максимум."""
        metrics['structure_score'] = min(metrics.get('structure_score', 0), max_score)
        return metrics
    
    def _set_header_score(self, metrics: Dict[str, Any], max_score: float) -> Dict[str, Any]:
        """Корректирует оценку заголовков с ограничением на максимум."""
        metrics['header_score'] = min(metrics.get('header_score', 0), max_score)
        return metrics
    
    def _set_semantic_depth(self, metrics: Dict[str, Any], max_score: float) -> Dict[str, Any]:
        """Корректирует оценку семантической глубины с ограничением на максимум."""
        metrics['semantic_depth'] = min(metrics.get('semantic_depth', 0), max_score)
        return metrics
    
    def _set_multimedia_score(self, metrics: Dict[str, Any], max_score: float) -> Dict[str, Any]:
        """Корректирует оценку мультимедиа с ограничением на максимум."""
        metrics['multimedia_score'] = min(metrics.get('multimedia_score', 0), max_score)
        return metrics
    
    def _adjust_keyword_density(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Корректирует нереалистично высокую плотность ключевых слов."""
        current_density = metrics.get('keyword_density', 0)
        if current_density > 0.1:  # Больше 10% - вероятно ошибка или спам
            metrics['keyword_density'] = min(current_density, 0.1)
            # Добавляем пометку о корректировке
            metrics['keyword_density_adjusted'] = True
        return metrics
    
    def _check_ranges(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Проверяет, что все метрики находятся в допустимых диапазонах."""
        # Все относительные оценки должны быть от 0 до 1
        score_metrics = [
            'readability', 'header_score', 'structure_score', 'semantic_depth',
            'topic_relevance', 'engagement_potential', 'meta_score', 'multimedia_score',
            'linking_score'
        ]
        
        # Устанавливаем допустимые диапазоны для метрик
        for metric_name in score_metrics:
            if metric_name in metrics:
                metrics[metric_name] = max(0, min(1.0, metrics[metric_name]))
        
        return metrics
