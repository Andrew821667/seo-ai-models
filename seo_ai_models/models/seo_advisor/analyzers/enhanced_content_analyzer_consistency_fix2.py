
"""
Улучшенное исправление несогласованности в оценке читабельности между TextProcessor и ContentAnalyzer.
"""

from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer

# Получаем оригинальный метод
original_analyze_content = EnhancedContentAnalyzer.analyze_content

# Полностью переопределяем метод для согласования шкал
def consistent_analyze_content(self, content, html_content=None):
    # Вызываем оригинальный метод
    results = original_analyze_content(self, content, html_content)
    
    # Сохраняем оригинальное значение для отладки
    if 'readability' in results:
        results['readability_original'] = results['readability']
        
        # Получаем "правильное" значение из TextProcessor
        readability_from_tp = self.text_processor.calculate_enhanced_readability(content)
        tp_score = readability_from_tp.get('readability_score', 0.5)
        
        # Перезаписываем значение для согласованности
        results['readability'] = tp_score
        results['readability_note'] = 'Значение согласовано с TextProcessor (шкала: 0 = сложно, 1 = легко)'
    
    return results

# Заменяем метод
EnhancedContentAnalyzer.analyze_content = consistent_analyze_content
