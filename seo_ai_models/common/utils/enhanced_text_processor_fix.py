
"""
Исправление для EnhancedTextProcessor, добавляющее метод calculate_enhanced_readability
"""

from typing import Dict, List, Any
import re
from seo_ai_models.common.utils.enhanced_text_processor import EnhancedTextProcessor

# Добавляем недостающий метод
def calculate_enhanced_readability(self, text: str, language: str = None) -> Dict[str, float]:
    """
    Расширенный анализ читабельности текста с учетом языка.
    
    Args:
        text: Анализируемый текст
        language: Код языка (ru, en, и т.д.)
        
    Returns:
        Dict[str, float]: Метрики читабельности
    """
    if not text:
        return {
            'readability_score': 0.0,
            'flesch_reading_ease': 0.0,
            'complexity_level': 0.0
        }
    
    # Используем язык из объекта, если не указан явно
    lang = language or self.language or 'en'
    
    # Анализируем текст
    sentences = self.split_sentences(text)
    words = self.tokenize(text)
    syllables = self._count_syllables(text, lang)
    
    # Количество слов, предложений и слогов
    word_count = len(words)
    sentence_count = max(len(sentences), 1)  # Избегаем деления на ноль
    syllable_count = syllables
    
    # Считаем среднюю длину предложения
    avg_sentence_length = word_count / sentence_count
    
    # Считаем среднюю длину слова в слогах
    avg_syllables_per_word = syllable_count / max(word_count, 1)
    
    # Расчет индекса Flesch Reading Ease (адаптирован для разных языков)
    if lang == 'ru':
        # Адаптированная формула для русского
        flesch_score = 206.835 - (1.3 * avg_sentence_length) - (60.1 * avg_syllables_per_word)
    else:
        # Стандартная формула для английского
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Нормализация значения
    flesch_score = max(min(flesch_score, 100), 0)
    
    # Конвертация в оценку читабельности (0-1)
    readability_score = flesch_score / 100.0
    
    # Определение уровня сложности (0-1)
    complexity_level = 1.0 - readability_score
    
    return {
        'readability_score': readability_score,
        'flesch_reading_ease': flesch_score,
        'complexity_level': complexity_level,
        'avg_sentence_length': avg_sentence_length,
        'avg_syllables_per_word': avg_syllables_per_word,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'syllable_count': syllable_count
    }

# Вспомогательный метод для подсчета слогов
def _count_syllables(self, text: str, language: str = 'en') -> int:
    """
    Подсчет слогов в тексте с учетом языка.
    
    Args:
        text: Анализируемый текст
        language: Код языка
        
    Returns:
        int: Количество слогов
    """
    if not text:
        return 0
    
    # Токенизация на слова
    words = self.tokenize(text.lower())
    
    total_syllables = 0
    
    if language == 'ru':
        # Правила для русского языка: гласные как основа слогов
        vowels = set('аеёиоуыэюя')
        for word in words:
            syllable_count = sum(1 for char in word if char in vowels)
            # Минимум 1 слог в слове, если оно не пустое
            total_syllables += max(syllable_count, 1 if word else 0)
    else:
        # Правила для английского языка
        vowels = set('aeiouy')
        for word in words:
            syllable_count = 0
            prev_is_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    syllable_count += 1
                prev_is_vowel = is_vowel
            
            # Особые случаи для английского
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                syllable_count += 1
            
            # Минимум 1 слог в слове, если оно не пустое
            total_syllables += max(syllable_count, 1 if word else 0)
    
    return total_syllables

# Добавляем методы к классу
EnhancedTextProcessor.calculate_enhanced_readability = calculate_enhanced_readability
EnhancedTextProcessor._count_syllables = _count_syllables
