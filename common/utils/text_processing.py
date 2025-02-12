import re
from typing import List, Dict, Union
from bs4 import BeautifulSoup
from dataclasses import dataclass
from collections import Counter

@dataclass
class TextMetrics:
    """Метрики текста"""
    word_count: int
    char_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float
    keyword_density: Dict[str, float]

class TextProcessor:
    """Обработка и анализ текста"""
    
    def __init__(self):
        self.stop_words = set()  # Здесь можно добавить стоп-слова
        
    def clean_html(self, text: str) -> str:
        """
        Очистка HTML-тегов
        Args:
            text: исходный текст
        Returns:
            очищенный текст
        """
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ')
    
    def normalize_text(self, text: str) -> str:
        """
        Нормализация текста
        Args:
            text: исходный текст
        Returns:
            нормализованный текст
        """
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление специальных символов
        text = re.sub(r'[^\w\s]', ' ', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_keywords(
        self,
        text: str,
        min_length: int = 3,
        top_k: int = 10
    ) -> List[str]:
        """
        Извлечение ключевых слов
        Args:
            text: текст
            min_length: минимальная длина слова
            top_k: количество ключевых слов
        Returns:
            список ключевых слов
        """
        # Нормализация и токенизация
        words = self.normalize_text(text).split()
        
        # Фильтрация слов
        words = [
            word for word in words
            if len(word) >= min_length and word not in self.stop_words
        ]
        
        # Подсчет частоты слов
        word_freq = Counter(words)
        
        # Выбор top-k слов
        return [word for word, _ in word_freq.most_common(top_k)]
    
    def calculate_metrics(self, text: str) -> TextMetrics:
        """
        Расчет метрик текста
        Args:
            text: текст
        Returns:
            метрики текста
        """
        # Очистка и нормализация
        clean_text = self.clean_html(text)
        
        # Подсчет слов и символов
        words = clean_text.split()
        word_count = len(words)
        char_count = len(clean_text)
        
        # Подсчет предложений
        sentences = re.split(r'[.!?]+', clean_text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Средние значения
        avg_word_length = char_count / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Плотность ключевых слов
        keywords = self.extract_keywords(clean_text)
        word_freq = Counter(words)
        keyword_density = {
            keyword: word_freq[keyword] / word_count
            for keyword in keywords
        }
        
        return TextMetrics(
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            keyword_density=keyword_density
        )
