"""
Статистический анализатор для микро-бизнеса.

Модуль предоставляет функциональность для анализа контента
на основе статистических методов, не требующих LLM.
"""

import logging
import re
import math
import string
from collections import Counter
from typing import Dict, List, Any, Optional, Set, Tuple

# Импорты из общих утилит
from seo_ai_models.common.utils.text_processing import (
    tokenize_text, extract_sentences, extract_paragraphs
)


class StatisticalAnalyzer:
    """
    Статистический анализатор для микро-бизнеса.
    
    Класс отвечает за статистический анализ контента без использования
    машинного обучения или LLM.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует статистический анализатор.
        
        Args:
            config: Конфигурация
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Настройки анализатора
        self.min_ngram_freq = self.config.get('min_ngram_freq', 2)
        self.max_ngram_size = self.config.get('max_ngram_size', 3)
        
        # Стоп-слова для исключения из анализа ключевых слов
        self.stopwords = self._load_stopwords()
        
        self.logger.info("StatisticalAnalyzer инициализирован")
    
    def _load_stopwords(self) -> Set[str]:
        """
        Загружает стандартные стоп-слова.
        
        Returns:
            Набор стоп-слов
        """
        # Базовый набор стоп-слов
        default_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'from', 'by', 'on', 'off', 'for', 'in', 'out', 'over', 'to',
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
            'can', 'could', 'may', 'might', 'must', 'ought',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their',
            'mine', 'yours', 'hers', 'ours', 'theirs',
            'this', 'that', 'these', 'those',
            'who', 'whom', 'whose', 'which', 'what',
            'where', 'when', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'many', 'some',
            'other', 'another', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very'
        }
        
        # Дополнительные стоп-слова из конфигурации
        custom_stopwords = set(self.config.get('stopwords', []))
        
        return default_stopwords.union(custom_stopwords)
    
    def analyze_content(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Анализирует контент с использованием статистических методов.
        
        Args:
            content: Текст для анализа
            keywords: Ключевые слова
            **kwargs: Дополнительные параметры
            
        Returns:
            Результаты анализа
        """
        self.logger.info("Начало статистического анализа контента")
        
        results = {
            'success': True,
            'message': 'Статистический анализ выполнен успешно',
        }
        
        # Извлечение n-грамм и ключевых слов
        results['ngrams'] = self.extract_ngrams(content)
        results['extracted_keywords'] = self.extract_keywords(content)
        
        # Анализ тематики
        results['topic_analysis'] = self.analyze_topic(content)
        
        # Анализ сложности текста
        results['complexity_analysis'] = self.analyze_complexity(content)
        
        # Анализ сентимента
        results['sentiment_analysis'] = self.analyze_sentiment(content)
        
        # Если указаны ключевые слова, анализируем их релевантность
        if keywords:
            results['keyword_relevance'] = self.analyze_keyword_relevance(
                content=content,
                keywords=keywords,
                extracted_keywords=results['extracted_keywords']
            )
            
        self.logger.info("Статистический анализ контента завершен")
        return results
    
    def extract_ngrams(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Извлекает наиболее часто встречающиеся n-граммы из текста.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Список n-грамм с частотой встречаемости
        """
        words = tokenize_text(content.lower())
        
        # Удаляем стоп-слова и пунктуацию
        filtered_words = [
            word for word in words
            if word not in self.stopwords and word not in string.punctuation
        ]
        
        # Извлекаем n-граммы разных размеров
        ngrams = {}
        
        for n in range(1, min(self.max_ngram_size + 1, len(filtered_words))):
            ngram_list = []
            
            # Формируем n-граммы
            for i in range(len(filtered_words) - n + 1):
                ngram = ' '.join(filtered_words[i:i+n])
                ngram_list.append(ngram)
                
            # Считаем частоту n-грамм
            counter = Counter(ngram_list)
            
            # Фильтруем по минимальной частоте
            frequent_ngrams = [
                {'ngram': ngram, 'frequency': count} 
                for ngram, count in counter.most_common(10)
                if count >= self.min_ngram_freq
            ]
            
            ngrams[f'{n}-grams'] = frequent_ngrams
            
        return ngrams
    
    def extract_keywords(self, content: str) -> List[Dict[str, Any]]:
        """
        Извлекает ключевые слова из текста на основе TF-IDF.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Список ключевых слов с оценками
        """
        words = tokenize_text(content.lower())
        
        # Удаляем стоп-слова и пунктуацию
        filtered_words = [
            word for word in words
            if word not in self.stopwords and word not in string.punctuation
        ]
        
        # Считаем частоту слов
        word_counts = Counter(filtered_words)
        
        # Общее количество слов
        total_words = len(filtered_words)
        
        # Приближение TF-IDF (без IDF, так как у нас один документ)
        # Используем лог-частоту
        keywords = []
        
        for word, count in word_counts.most_common(20):
            # Пропускаем слова короче 3 символов
            if len(word) < 3:
                continue
                
            # Вычисляем лог-частоту
            log_freq = 1 + math.log(count)
            
            # Вычисляем нормированную позицию первого вхождения
            first_pos = content.lower().find(word) / len(content)
            
            # Вычисляем итоговую оценку (больше вес у слов в начале документа)
            score = log_freq * (1 - 0.5 * first_pos)
            
            keywords.append({
                'keyword': word,
                'count': count,
                'frequency': count / total_words,
                'score': round(score, 4),
            })
            
        # Сортируем по оценке
        keywords.sort(key=lambda x: x['score'], reverse=True)
        
        return keywords
    
    def analyze_topic(self, content: str) -> Dict[str, Any]:
        """
        Анализирует тематику текста на основе частотности слов.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Результаты анализа тематики
        """
        # Извлекаем ключевые слова
        keywords = self.extract_keywords(content)
        
        # Берем топ-10 ключевых слов
        top_keywords = keywords[:10]
        
        # Вычисляем "тематическую плотность" - отношение
        # суммы частот топ-10 ключевых слов к общей длине текста
        total_frequency = sum(kw['frequency'] for kw in top_keywords)
        
        # Определяем "тематическую согласованность" - насколько топ-10
        # ключевых слов семантически близки (упрощенная метрика)
        # Здесь мы просто считаем, сколько раз ключевые слова 
        # встречаются близко друг к другу в тексте
        
        sentences = extract_sentences(content)
        coherence = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keywords_in_sentence = sum(
                1 for kw in top_keywords if kw['keyword'] in sentence_lower
            )
            # Если в предложении более одного ключевого слова, увеличиваем согласованность
            if keywords_in_sentence > 1:
                coherence += keywords_in_sentence - 1
                
        # Нормализуем согласованность
        normalized_coherence = min(1.0, coherence / (len(sentences) * 0.5))
        
        # Оценка тематической сфокусированности
        focus_score = (total_frequency + normalized_coherence) / 2
        
        return {
            'top_keywords': [kw['keyword'] for kw in top_keywords],
            'topic_density': round(total_frequency, 4),
            'topic_coherence': round(normalized_coherence, 4),
            'focus_score': round(focus_score, 4),
            'topic_focus_category': self._get_focus_category(focus_score),
        }
    
    def _get_focus_category(self, focus_score: float) -> str:
        """
        Определяет категорию тематической сфокусированности.
        
        Args:
            focus_score: Оценка сфокусированности
            
        Returns:
            Категория сфокусированности
        """
        if focus_score < 0.2:
            return "Очень низкая (много разных тем)"
        elif focus_score < 0.4:
            return "Низкая (несколько слабо связанных тем)"
        elif focus_score < 0.6:
            return "Средняя (одна основная тема с отступлениями)"
        elif focus_score < 0.8:
            return "Высокая (четкая тематическая линия)"
        else:
            return "Очень высокая (сильная тематическая фокусировка)"
    
    def analyze_complexity(self, content: str) -> Dict[str, Any]:
        """
        Анализирует сложность текста.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Результаты анализа сложности
        """
        words = tokenize_text(content)
        sentences = extract_sentences(content)
        paragraphs = extract_paragraphs(content)
        
        # Если нет слов или предложений, возвращаем базовые значения
        if not words or not sentences:
            return {
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'avg_paragraph_length': 0,
                'unique_words_ratio': 0,
                'complexity_score': 0,
                'complexity_category': "Не определено",
            }
        
        # Средняя длина слова в символах
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Средняя длина предложения в словах
        avg_sentence_length = len(words) / len(sentences)
        
        # Средняя длина параграфа в предложениях
        avg_paragraph_length = len(sentences) / max(len(paragraphs), 1)
        
        # Отношение уникальных слов к общему количеству
        unique_words = set(word.lower() for word in words)
        unique_words_ratio = len(unique_words) / len(words)
        
        # Вычисляем итоговую оценку сложности
        complexity_score = (
            (avg_word_length / 5) * 0.3 +  # Нормализованная длина слов
            (avg_sentence_length / 20) * 0.4 +  # Нормализованная длина предложений
            unique_words_ratio * 0.3  # Разнообразие словаря
        )
        
        # Определяем категорию сложности
        complexity_category = self._get_complexity_category(complexity_score)
        
        return {
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_paragraph_length': round(avg_paragraph_length, 2),
            'unique_words_ratio': round(unique_words_ratio, 2),
            'complexity_score': round(complexity_score, 2),
            'complexity_category': complexity_category,
        }
    
    def _get_complexity_category(self, complexity_score: float) -> str:
        """
        Определяет категорию сложности текста.
        
        Args:
            complexity_score: Оценка сложности
            
        Returns:
            Категория сложности
        """
        if complexity_score < 0.3:
            return "Очень простой (элементарный уровень)"
        elif complexity_score < 0.5:
            return "Простой (базовый уровень)"
        elif complexity_score < 0.7:
            return "Средний (общий уровень)"
        elif complexity_score < 0.85:
            return "Сложный (продвинутый уровень)"
        else:
            return "Очень сложный (экспертный уровень)"
    
    def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """
        Анализирует тональность текста.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Результаты анализа тональности
        """
        # Загружаем словари позитивных и негативных слов
        positive_words = self._get_positive_words()
        negative_words = self._get_negative_words()
        
        # Токенизируем текст
        words = tokenize_text(content.lower())
        
        # Считаем позитивные и негативные слова
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Общее количество слов
        total_words = len(words)
        
        # Вычисляем относительные частоты
        positive_ratio = positive_count / total_words if total_words > 0 else 0
        negative_ratio = negative_count / total_words if total_words > 0 else 0
        
        # Вычисляем общую оценку тональности (-1 до 1)
        sentiment_score = positive_ratio - negative_ratio
        
        # Определяем категорию тональности
        sentiment_category = self._get_sentiment_category(sentiment_score)
        
        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'positive_ratio': round(positive_ratio, 4),
            'negative_ratio': round(negative_ratio, 4),
            'sentiment_score': round(sentiment_score, 4),
            'sentiment_category': sentiment_category,
        }
    
    def _get_sentiment_category(self, sentiment_score: float) -> str:
        """
        Определяет категорию тональности текста.
        
        Args:
            sentiment_score: Оценка тональности
            
        Returns:
            Категория тональности
        """
        if sentiment_score < -0.3:
            return "Негативная"
        elif sentiment_score < -0.1:
            return "Умеренно негативная"
        elif sentiment_score < 0.1:
            return "Нейтральная"
        elif sentiment_score < 0.3:
            return "Умеренно позитивная"
        else:
            return "Позитивная"
    
    def _get_positive_words(self) -> Set[str]:
        """
        Возвращает набор позитивных слов.
        
        Returns:
            Набор позитивных слов
        """
        # Базовый набор позитивных слов
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'wonderful', 'amazing', 
            'fantastic', 'best', 'better', 'superior', 'perfect', 'ideal',
            'happy', 'glad', 'pleased', 'delighted', 'satisfied', 'enjoy',
            'love', 'like', 'appreciate', 'impressive', 'favorable', 'beneficial',
            'recommended', 'successful', 'outstanding', 'exceptional', 'remarkable',
            'superb', 'terrific', 'awesome', 'brilliant', 'splendid', 'marvelous',
            'quality', 'valuable', 'reliable', 'trusted', 'innovative', 'affordable'
        }
        
        # Добавляем пользовательские позитивные слова
        custom_positive = set(self.config.get('positive_words', []))
        
        return positive_words.union(custom_positive)
    
    def _get_negative_words(self) -> Set[str]:
        """
        Возвращает набор негативных слов.
        
        Returns:
            Набор негативных слов
        """
        # Базовый набор негативных слов
        negative_words = {
            'bad', 'poor', 'terrible', 'negative', 'awful', 'horrible',
            'worst', 'worse', 'inferior', 'inadequate', 'disappointing',
            'unhappy', 'sad', 'angry', 'upset', 'dissatisfied', 'hate',
            'dislike', 'against', 'unfavorable', 'harmful', 'not recommended',
            'unsuccessful', 'failure', 'problem', 'issue', 'difficult',
            'complicated', 'confusing', 'frustrating', 'annoying', 'irritating',
            'expensive', 'overpriced', 'unreliable', 'slow', 'broken', 'faulty'
        }
        
        # Добавляем пользовательские негативные слова
        custom_negative = set(self.config.get('negative_words', []))
        
        return negative_words.union(custom_negative)
    
    def analyze_keyword_relevance(
        self,
        content: str,
        keywords: List[str],
        extracted_keywords: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Анализирует релевантность ключевых слов к контенту.
        
        Args:
            content: Текст для анализа
            keywords: Заданные ключевые слова
            extracted_keywords: Извлеченные ключевые слова
            
        Returns:
            Результаты анализа релевантности
        """
        # Извлеченные ключевые слова в виде набора
        extracted_keywords_set = {kw['keyword'] for kw in extracted_keywords}
        
        # Вычисляем релевантность для каждого заданного ключевого слова
        keyword_relevance = {}
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Количество вхождений
            count = content.lower().count(keyword_lower)
            
            # Наличие в извлеченных ключевых словах
            in_extracted = any(
                kw['keyword'] in keyword_lower or keyword_lower in kw['keyword']
                for kw in extracted_keywords[:10]  # Проверяем только топ-10
            )
            
            # Позиция первого вхождения (нормализованная)
            first_pos = content.lower().find(keyword_lower)
            pos_score = 0
            if first_pos >= 0:
                pos_score = 1 - (first_pos / len(content))
                
            # Вычисляем общую оценку релевантности
            relevance_score = 0
            if count > 0:
                relevance_score = (
                    (min(count, 10) / 10) * 0.5 +  # Частота (не более 10)
                    (1 if in_extracted else 0) * 0.3 +  # Наличие в извлеченных
                    pos_score * 0.2  # Позиция
                )
                
            keyword_relevance[keyword] = {
                'count': count,
                'in_extracted_keywords': in_extracted,
                'position_score': round(pos_score, 2),
                'relevance_score': round(relevance_score, 2),
                'relevance_category': self._get_relevance_category(relevance_score),
            }
            
        # Общая релевантность контента к ключевым словам
        avg_relevance = sum(
            kr['relevance_score'] for kr in keyword_relevance.values()
        ) / len(keywords) if keywords else 0
        
        return {
            'keyword_relevance': keyword_relevance,
            'overall_relevance': round(avg_relevance, 2),
            'overall_category': self._get_relevance_category(avg_relevance),
        }
    
    def _get_relevance_category(self, relevance_score: float) -> str:
        """
        Определяет категорию релевантности.
        
        Args:
            relevance_score: Оценка релевантности
            
        Returns:
            Категория релевантности
        """
        if relevance_score < 0.2:
            return "Очень низкая"
        elif relevance_score < 0.4:
            return "Низкая"
        elif relevance_score < 0.6:
            return "Средняя"
        elif relevance_score < 0.8:
            return "Высокая"
        else:
            return "Очень высокая"
