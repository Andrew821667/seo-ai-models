
"""
Исправление несогласованности в оценке читабельности между TextProcessor и ContentAnalyzer.
"""

from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer

# Получаем оригинальный метод
original_analyze_content = EnhancedContentAnalyzer.analyze_content

# Определяем исправленный метод с согласованными шкалами читабельности
def consistent_analyze_content(self, content, html_content=None):
    # Вызываем оригинальный метод
    results = original_analyze_content(self, content, html_content)
    
    # Исправляем несогласованность в шкале читабельности
    # Если readability в результатах близка к 1.0, предполагаем, что шкала инвертирована
    if 'readability' in results and results['readability'] > 0.85:
        # Инвертируем шкалу, чтобы соответствовать шкале TextProcessor
        # Где 0 = сложно читать, 1 = легко читать
        original_value = results['readability']
        inverted_value = 1.0 - original_value
        
        # Обновляем значение и добавляем пояснение
        results['readability'] = inverted_value
        results['readability_original'] = original_value
        results['readability_note'] = 'Шкала согласована с TextProcessor: 0 = сложно, 1 = легко'
    
    return results

# Заменяем метод
EnhancedContentAnalyzer.analyze_content = consistent_analyze_content
