import os
import nltk
import numpy as np
from typing import Dict, List, Optional
import logging
from transformers import AutoTokenizer, AutoModel
import torch

from common.config.advisor_config import ModelConfig
from common.utils.text_processing import TextProcessor

class ContentAnalyzer:
    """Анализатор контента с использованием NLP и ML техник"""
    
    def __init__(self):
        """Инициализация анализатора"""
        self.model_config = ModelConfig()
        self.text_processor = TextProcessor()
        # Загружаем предобученную модель для анализа текста
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
    def analyze_text(self, content: str) -> Dict[str, float]:
        """
        Анализ текста и расчёт основных метрик
        
        Args:
            content (str): Текстовый контент для анализа
            
        Returns:
            Dict[str, float]: Словарь с метриками текста
        """
        # Базовые метрики
        words = self.text_processor.tokenize(content)
        sentences = self.text_processor.split_sentences(content)
        
        # Считаем базовые метрики
        word_count = len(words)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Анализ заголовков
        headers = self.text_processor.extract_headers(content)
        header_score = self._calculate_header_score(headers)
        
        # Анализ читабельности
        readability = self._calculate_readability(content)
        
        # Анализ мета-тегов
        meta_score = self._analyze_meta_tags(content)
        
        # Анализ мультимедиа
        multimedia_score = self._analyze_multimedia(content)
        
        # Анализ внутренних ссылок
        linking_score = self._analyze_internal_links(content)
        
        return {
            'word_count': word_count,
            'readability': readability,
            'meta_score': meta_score,
            'header_score': header_score,
            'multimedia_score': multimedia_score,
            'linking_score': linking_score,
            'avg_sentence_length': avg_sentence_length
        }
    
    def extract_keywords(self, content: str, target_keywords: List[str]) -> Dict[str, float]:
        """
        Извлечение и анализ ключевых слов
        
        Args:
            content (str): Текстовый контент
            target_keywords (List[str]): Целевые ключевые слова
            
        Returns:
            Dict[str, float]: Метрики ключевых слов
        """
        # Нормализуем контент и ключевые слова
        normalized_content = self.text_processor.normalize(content)
        normalized_keywords = [self.text_processor.normalize(kw) for kw in target_keywords]
        
        # Считаем встречаемость ключевых слов
        keyword_counts = {}
        total_words = len(normalized_content.split())
        
        for keyword in normalized_keywords:
            count = normalized_content.count(keyword)
            keyword_counts[keyword] = count
        
        # Рассчитываем общую плотность ключевых слов
        total_keyword_count = sum(keyword_counts.values())
        keyword_density = total_keyword_count / total_words if total_words > 0 else 0
        
        # Анализируем распределение ключевых слов
        keyword_distribution = self._analyze_keyword_distribution(content, target_keywords)
        
        return {
            'density': keyword_density,
            'counts': keyword_counts,
            'distribution': keyword_distribution
        }
    
    def _calculate_readability(self, text: str) -> float:
        """
        Расчёт показателя читабельности текста
        """
        words = self.text_processor.tokenize(text)
        sentences = self.text_processor.split_sentences(text)
        
        if not sentences:
            return 0.0
            
        # Базовый индекс Флеша-Кинкейда
        avg_sentence_length = len(words) / len(sentences)
        syllables = sum(self.text_processor.count_syllables(word) for word in words)
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        # Нормализуем значение от 0 до 100
        return max(0, min(100, readability))
    
    def _calculate_header_score(self, headers: Dict[str, List[str]]) -> float:
        """
        Оценка структуры заголовков
        """
        if not headers:
            return 0.0
            
        # Проверяем иерархию заголовков
        header_levels = sorted(headers.keys())
        hierarchy_score = 1.0
        
        for i in range(len(header_levels) - 1):
            if int(header_levels[i+1][1]) - int(header_levels[i][1]) > 1:
                hierarchy_score *= 0.8
        
        # Проверяем длину заголовков
        length_scores = []
        for level_headers in headers.values():
            for header in level_headers:
                words = len(header.split())
                if 2 <= words <= 8:
                    length_scores.append(1.0)
                else:
                    length_scores.append(0.7)
        
        avg_length_score = sum(length_scores) / len(length_scores) if length_scores else 0
        
        return (hierarchy_score * 0.6 + avg_length_score * 0.4)
    
    def _analyze_meta_tags(self, content: str) -> float:
        """
        Анализ мета-тегов
        """
        # В реальном приложении здесь будет анализ HTML
        # Для демонстрации возвращаем базовый скор
        return 0.7
    
    def _analyze_multimedia(self, content: str) -> float:
        """
        Анализ мультимедийного контента
        """
        # В реальном приложении здесь будет анализ изображений/видео
        # Для демонстрации возвращаем базовый скор
        return 0.6
    
    def _analyze_internal_links(self, content: str) -> float:
        """
        Анализ внутренних ссылок
        """
        # В реальном приложении здесь будет анализ ссылок
        # Для демонстрации возвращаем базовый скор
        return 0.5
    
    def _analyze_keyword_distribution(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """
        Анализ распределения ключевых слов по тексту
        """
        # Разбиваем текст на секции
        sections = self.text_processor.split_into_sections(content)
        
        distribution = {}
        for keyword in keywords:
            section_scores = []
            for section in sections:
                section_words = len(section.split())
                keyword_count = section.lower().count(keyword.lower())
                
                if section_words > 0:
                    score = keyword_count / section_words
                    section_scores.append(score)
            
            # Рассчитываем равномерность распределения
            if section_scores:
                variance = np.var(section_scores)
                distribution[keyword] = 1.0 / (1.0 + variance)
            else:
                distribution[keyword] = 0.0
                
        return distribution
