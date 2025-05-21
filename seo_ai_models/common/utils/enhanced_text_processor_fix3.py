
"""
Финальное исправление для EnhancedTextProcessor, добавляющее метод calculate_keyword_prominence
"""

from typing import Dict, List, Any
import re
from seo_ai_models.common.utils.enhanced_text_processor import EnhancedTextProcessor

def calculate_keyword_prominence(self, text: str, keywords: List[str]) -> Dict[str, float]:
    """
    Рассчитывает prominence (заметность) ключевых слов в тексте.
    
    Prominence учитывает позицию ключевого слова в тексте, его наличие в заголовках,
    первом и последнем абзацах, и другие факторы.
    
    Args:
        text: Анализируемый текст
        keywords: Список ключевых слов для анализа
        
    Returns:
        Dict[str, float]: Словарь с оценками prominence для каждого ключевого слова
    """
    if not text or not keywords:
        return {}
    
    # Нормализуем текст и ключевые слова
    text_lower = text.lower()
    keywords_lower = [keyword.lower() for keyword in keywords]
    
    # Разбиваем текст на компоненты
    paragraphs = text.split('\n\n')
    first_paragraph = paragraphs[0] if paragraphs else ""
    last_paragraph = paragraphs[-1] if len(paragraphs) > 1 else ""
    
    # Извлекаем заголовки
    headers = self.extract_headers(text)
    header_texts = [header.get('text', '').lower() for header in headers]
    
    # Словарь для хранения prominence оценок
    prominence_scores = {}
    
    # Анализируем каждое ключевое слово
    for keyword in keywords_lower:
        # Базовая оценка
        score = 0.0
        
        # Проверяем наличие в заголовках (высокий вес)
        for header_text in header_texts:
            if keyword in header_text:
                score += 0.3
                break
        
        # Проверяем наличие в первом абзаце (высокий вес)
        if keyword in first_paragraph.lower():
            score += 0.25
        
        # Проверяем наличие в последнем абзаце (средний вес)
        if keyword in last_paragraph.lower():
            score += 0.15
        
        # Проверяем общую частоту в тексте (низкий вес)
        occurrences = text_lower.count(keyword)
        frequency_score = min(occurrences / 10.0, 0.3)  # Максимум 0.3 за частоту
        score += frequency_score
        
        # Ищем первое вхождение (чем раньше, тем лучше)
        first_occurrence = text_lower.find(keyword)
        if first_occurrence >= 0:
            # Вычисляем позицию относительно длины текста (0 - в начале, 1 - в конце)
            relative_position = first_occurrence / len(text_lower)
            # Инвертируем: раннее вхождение дает больший бонус
            position_score = 0.2 * (1 - relative_position)
            score += position_score
        
        # Нормализуем итоговую оценку (максимум 1.0)
        prominence_scores[keyword] = min(score, 1.0)
    
    return prominence_scores

# Добавляем метод к классу
EnhancedTextProcessor.calculate_keyword_prominence = calculate_keyword_prominence
