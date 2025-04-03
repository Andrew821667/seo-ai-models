"""Утилиты для обработки текста.

Модуль предоставляет функции для анализа и обработки текста, включая:
- токенизацию
- разделение на предложения
- извлечение заголовков
- очистку HTML
- расчет метрик читабельности
- определение языка
"""

from typing import Dict, List, Tuple, Set, Optional, Union
import re
import os
import string
from collections import Counter

class TextProcessor:
    """Класс для обработки и анализа текста.
    
    Предоставляет методы для токенизации, нормализации, извлечения метрик текста,
    анализа заголовков и структуры, а также расчета показателей читабельности.
    """
    
    def __init__(self):
        """Инициализация процессора текста."""
        # Список стоп-слов для русского языка
        self.stopwords_ru = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
            'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к',
            'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне',
            'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему'
        }
        
        # Список стоп-слов для английского языка
        self.stopwords_en = {
            'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'with',
            'for', 'as', 'are', 'on', 'was', 'be', 'this', 'by', 'not', 'or',
            'have', 'you', 'from', 'at', 'an', 'will', 'can', 'they', 'their',
            'but', 'we', 'he', 'she', 'all', 'has', 'been', 'when', 'who', 'which'
        }
            
    def detect_language(self, text: str) -> str:
        """Определение языка текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Код языка ('ru', 'en' или 'unknown')
        """
        if not text:
            return 'unknown'
            
        # Простая эвристика для определения языка
        text = text.lower()
        ru_chars = len(re.findall(r'[а-яё]', text))
        en_chars = len(re.findall(r'[a-z]', text))
        
        if ru_chars > en_chars:
            return 'ru'
        elif en_chars > ru_chars:
            return 'en'
        else:
            return 'unknown'
    
    def tokenize(self, text: str, remove_stopwords: bool = False, language: str = None) -> List[str]:
        """Разбивает текст на токены (слова).
        
        Args:
            text: Текст для токенизации
            remove_stopwords: Удалять ли стоп-слова
            language: Язык текста ('ru', 'en' или None для автоопределения)
            
        Returns:
            Список токенов
        """
        if not text:
            return []
            
        # Определяем язык, если не задан
        if language is None:
            language = self.detect_language(text)
            
        # Очищаем текст и разбиваем на слова
        text = text.lower()
        # Удаляем знаки пунктуации и цифры, заменяя их пробелами
        text = re.sub(r'[^\w\s]|\d', ' ', text)
        # Разбиваем на слова и удаляем пустые строки
        tokens = [token.strip() for token in text.split() if token.strip()]
            
        # Удаление стоп-слов если требуется
        if remove_stopwords:
            if language == 'ru':
                tokens = [token for token in tokens if token.lower() not in self.stopwords_ru]
            elif language == 'en':
                tokens = [token for token in tokens if token.lower() not in self.stopwords_en]
                
        return tokens
    
    def lemmatize(self, tokens: List[str], language: str = None) -> List[str]:
        """Упрощенная версия лемматизации (просто возвращает токены).
        
        Args:
            tokens: Список токенов для лемматизации
            language: Язык текста ('ru', 'en' или None для автоопределения)
            
        Returns:
            Список токенов (без лемматизации, в реальной системе здесь будет лемматизация)
        """
        # В рамках упрощенной версии просто возвращаем токены
        return tokens
    
    def normalize(self, text: str) -> str:
        """Нормализует текст (приводит к нижнему регистру).
        
        Args:
            text: Текст для нормализации
            
        Returns:
            Нормализованный текст
        """
        if not text:
            return ""
        # Приведение к нижнему регистру и удаление лишних пробелов
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_sentences(self, text: str, language: str = None) -> List[str]:
        """Разбивает текст на предложения.
        
        Args:
            text: Текст для разбиения
            language: Язык текста ('ru', 'en' или None для автоопределения)
            
        Returns:
            Список предложений
        """
        if not text:
            return []
            
        # Предварительная обработка: заменим переносы строк на пробелы
        text = re.sub(r'\n', ' ', text)
        
        # Простой алгоритм разбиения на предложения по знакам .!?
        # с учетом сокращений и других особых случаев
        sentences = []
        start = 0
        
        for match in re.finditer(r'[.!?]\s+', text):
            end = match.end()
            sentences.append(text[start:end].strip())
            start = end
            
        # Добавляем последнее предложение, если оно есть
        if start < len(text):
            sentences.append(text[start:].strip())
            
        return [s for s in sentences if s]
    
    def extract_headers(self, text: str, include_html: bool = False) -> List[Dict[str, str]]:
        """Извлекает заголовки из текста (Markdown и опционально HTML).
        
        Args:
            text: Текст для анализа
            include_html: Включать ли обработку HTML-заголовков
            
        Returns:
            Список заголовков с уровнем и текстом
        """
        if not text:
            return []
        
        headers = []
        
        # Извлечение Markdown-заголовков
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Подсчитываем количество символов #
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                header_text = line[level:].strip()
                if header_text and level <= 6:  # Проверяем валидность заголовка
                    headers.append({
                        'level': level,
                        'text': header_text,
                        'type': 'markdown'
                    })
        
        return headers
    
    def calculate_readability(self, text: str, language: str = None) -> Dict[str, float]:
        """Расчет различных метрик читабельности текста.
        
        Args:
            text: Текст для анализа
            language: Язык текста ('ru', 'en' или None для автоопределения)
            
        Returns:
            Словарь с метриками читабельности
        """
        if not text or len(text) < 20:
            return {'flesch_reading_ease': 0}
            
        # Определяем язык, если не задан
        if language is None:
            language = self.detect_language(text)
            
        words = self.tokenize(text)
        sentences = self.split_sentences(text)
        
        if not sentences or not words:
            return {'flesch_reading_ease': 0}
        
        # Средняя длина предложения в словах
        avg_sentence_length = len(words) / len(sentences)
        
        # Упрощенная формула Flesch Reading Ease
        readability_score = 206.835 - (1.3 * avg_sentence_length)
        
        # Нормализация оценки
        normalized_score = max(min(readability_score, 100), 0)
        
        return {'flesch_reading_ease': normalized_score}
    
    def extract_keywords(self, text: str, max_keywords: int = 10, 
                        min_word_length: int = 4, 
                        language: str = None) -> List[Tuple[str, float]]:
        """Извлечение ключевых слов из текста.
        
        Args:
            text: Текст для анализа
            max_keywords: Максимальное количество ключевых слов
            min_word_length: Минимальная длина слова
            language: Язык текста ('ru', 'en' или None для автоопределения)
            
        Returns:
            Список кортежей (слово, вес)
        """
        if not text:
            return []
            
        # Определяем язык, если не задан
        if language is None:
            language = self.detect_language(text)
            
        # Токенизируем текст и удаляем стоп-слова
        tokens = self.tokenize(text, remove_stopwords=True, language=language)
        
        # Фильтруем слова по длине
        tokens = [token.lower() for token in tokens if len(token) >= min_word_length]
        
        # Подсчитываем частоту
        counter = Counter(tokens)
        
        # Находим максимальную частоту для нормализации
        max_freq = max(counter.values()) if counter else 0
        
        # Нормализуем веса и сортируем
        if max_freq > 0:
            keywords = [(word, count / max_freq) for word, count in counter.most_common(max_keywords)]
        else:
            keywords = []
            
        return keywords
    
    def extract_ngrams(self, text: str, n: int = 2, max_ngrams: int = 10, 
                       language: str = None) -> List[Tuple[str, float]]:
        """Извлечение n-грамм из текста.
        
        Args:
            text: Текст для анализа
            n: Длина n-граммы (2 для биграмм, 3 для триграмм и т.д.)
            max_ngrams: Максимальное количество возвращаемых n-грамм
            language: Язык текста ('ru', 'en' или None для автоопределения)
            
        Returns:
            Список кортежей (n-грамма, вес)
        """
        if not text or n < 2:
            return []
            
        # Определяем язык, если не задан
        if language is None:
            language = self.detect_language(text)
            
        # Токенизируем текст
        tokens = self.tokenize(text, language=language)
        
        # Извлекаем n-граммы
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
            
        # Подсчитываем частоту
        counter = Counter(ngrams)
        
        # Находим максимальную частоту для нормализации
        max_freq = max(counter.values()) if counter else 0
        
        # Нормализуем веса и сортируем
        if max_freq > 0:
            top_ngrams = [(ngram, count / max_freq) for ngram, count in counter.most_common(max_ngrams)]
        else:
            top_ngrams = []
            
        return top_ngrams
    
    def analyze_text_structure(self, text: str) -> Dict[str, any]:
        """Анализ структуры текста: заголовки, списки, абзацы.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с метриками структуры текста
        """
        if not text:
            return {
                'headers_count': 0,
                'header_levels': [],
                'paragraphs_count': 0,
                'avg_paragraph_length': 0,
                'lists_count': 0,
                'has_conclusion': False,
                'has_introduction': False
            }
            
        # Извлекаем заголовки
        headers = self.extract_headers(text)
        header_levels = [h['level'] for h in headers]
        
        # Подсчитываем абзацы
        paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Средняя длина абзаца в символах
        avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        
        # Подсчитываем списки (маркированные и нумерованные)
        lists_count = 0
        list_markers = [
            r'^\s*[-*+]\s',  # Маркированные списки (-, *, +)
            r'^\s*\d+\.\s'  # Нумерованные списки (1., 2., etc)
        ]
        
        for paragraph in paragraphs:
            for line in paragraph.split('\n'):
                if any(re.match(marker, line) for marker in list_markers):
                    lists_count += 1
                    break
        
        # Проверка наличия введения и заключения
        has_introduction = False
        has_conclusion = False
        
        introduction_keywords = ['введение', 'introduction', 'вступление', 'начало', 'обзор']
        conclusion_keywords = ['заключение', 'вывод', 'итог', 'conclusion', 'summary', 'резюме']
        
        # Проверяем заголовки на наличие ключевых слов
        for header in headers:
            header_text = header['text'].lower()
            if any(keyword in header_text for keyword in introduction_keywords):
                has_introduction = True
            if any(keyword in header_text for keyword in conclusion_keywords):
                has_conclusion = True
        
        # Если не нашли в заголовках, проверяем первый и последний абзацы
        if paragraphs:
            if any(keyword in paragraphs[0].lower() for keyword in introduction_keywords):
                has_introduction = True
            if any(keyword in paragraphs[-1].lower() for keyword in conclusion_keywords):
                has_conclusion = True
        
        return {
            'headers_count': len(headers),
            'header_levels': header_levels,
            'paragraphs_count': len(paragraphs),
            'avg_paragraph_length': avg_paragraph_length,
            'lists_count': lists_count,
            'has_conclusion': has_conclusion,
            'has_introduction': has_introduction
        }
