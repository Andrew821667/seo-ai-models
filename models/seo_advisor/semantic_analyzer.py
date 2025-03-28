from typing import Dict, List, Optional, Set, Tuple, Any
import re
from collections import Counter

class SemanticAnalyzer:
    """Анализатор семантической структуры контента"""
    
    def __init__(self):
        # Стоп-слова для русского языка
        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
            'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к',
            'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне',
            'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему'
        }
        
    def analyze_text(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Семантический анализ текста
        
        Args:
            text: Анализируемый текст
            keywords: Список ключевых слов
            
        Returns:
            Словарь с результатами анализа
        """
        words = self._tokenize(text)
        
        # Результаты анализа
        results = {
            'semantic_fields': self._extract_semantic_fields(text, keywords),
            'semantic_density': self._calculate_semantic_density(words, keywords),
            'semantic_coverage': self._calculate_semantic_coverage(text, keywords),
            'topical_coherence': self._calculate_topical_coherence(words),
            'contextual_relevance': self._calculate_contextual_relevance(text, keywords),
            'related_terms': self._extract_related_terms(text, keywords)
        }
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        # Приводим к нижнему регистру и удаляем знаки пунктуации
        text = text.lower()
        text = re.sub(r'[^\wа-яА-Я\s]', ' ', text)
        
        # Разбиваем на слова и фильтруем стоп-слова
        words = [word.strip() for word in text.split() if word.strip()]
        words = [word for word in words if word not in self.stop_words]
        
        return words
        
    def _extract_semantic_fields(self, text: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Извлечение семантических полей для ключевых слов"""
        semantic_fields = {}
        words = self._tokenize(text)
        
        for keyword in keywords:
            # Находим связанные слова в контексте
            keyword_words = self._tokenize(keyword)
            related_words = []
            
            for i, word in enumerate(words):
                # Для каждого слова из ключевой фразы
                for kw in keyword_words:
                    # Если слово совпадает с ключевым
                    if kw in word:
                        # Добавляем слова вокруг (контекст)
                        context_size = 5
                        start = max(0, i - context_size)
                        end = min(len(words), i + context_size + 1)
                        
                        related_words.extend(words[start:i] + words[i+1:end])
            
            # Считаем частоту связанных слов
            counter = Counter(related_words)
            # Берем топ-10 наиболее частых
            most_common = counter.most_common(10)
            
            semantic_fields[keyword] = [word for word, count in most_common]
        
        return semantic_fields
    
    def _calculate_semantic_density(self, words: List[str], keywords: List[str]) -> float:
        """Расчет семантической плотности - включает ключевые слова и их вариации"""
        if not words:
            return 0.0
            
        # Нормализуем ключевые слова
        normalized_keywords = []
        for keyword in keywords:
            normalized_keywords.extend(self._tokenize(keyword))
        
        # Считаем вхождения
        matches = 0
        for word in words:
            for keyword in normalized_keywords:
                # Включаем частичные совпадения
                if keyword in word:
                    matches += 1
                    break
        
        return matches / len(words)
    
    def _calculate_semantic_coverage(self, text: str, keywords: List[str]) -> float:
        """Расчет покрытия семантики - насколько хорошо текст охватывает темы ключевых слов"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return 0.0
            
        # Для каждого параграфа определяем, содержит ли он ключевые слова
        coverage_scores = []
        
        for paragraph in paragraphs:
            para_words = self._tokenize(paragraph)
            para_coverage = []
            
            for keyword in keywords:
                keyword_words = self._tokenize(keyword)
                keyword_found = False
                
                for kw in keyword_words:
                    for word in para_words:
                        if kw in word:
                            keyword_found = True
                            break
                    if keyword_found:
                        break
                
                para_coverage.append(int(keyword_found))
            
            # Среднее покрытие ключевыми словами для параграфа
            if para_coverage:
                coverage_scores.append(sum(para_coverage) / len(keywords))
        
        # Среднее покрытие по всему тексту
        return sum(coverage_scores) / len(paragraphs) if coverage_scores else 0.0
    
    def _calculate_topical_coherence(self, words: List[str]) -> float:
        """Расчет тематической согласованности текста"""
        if len(words) < 5:
            return 0.0
            
        # Простая эвристика для подсчета когерентности
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Подсчитываем слова, которые повторяются более одного раза
        repeated_words = sum(1 for word, count in word_freq.items() if count > 1)
        
        # Нормализуем по количеству уникальных слов
        unique_words = len(word_freq)
        
        return min(repeated_words / unique_words if unique_words else 0, 1.0)
    
    def _calculate_contextual_relevance(self, text: str, keywords: List[str]) -> float:
        """Расчет контекстуальной релевантности"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs or not keywords:
            return 0.0
            
        # Оцениваем релевантность на основе распределения ключевых слов
        total_relevance = 0.0
        
        # Анализируем первый и последний параграф с повышенным весом
        first_paragraph = self._tokenize(paragraphs[0])
        last_paragraph = self._tokenize(paragraphs[-1])
        
        # Вес для первого и последнего параграфа
        first_weight = 1.5
        last_weight = 1.2
        
        # Релевантность первого параграфа
        first_relevance = self._calculate_paragraph_relevance(first_paragraph, keywords) * first_weight
        
        # Релевантность последнего параграфа
        last_relevance = self._calculate_paragraph_relevance(last_paragraph, keywords) * last_weight
        
        # Релевантность остальных параграфов
        middle_relevance = 0.0
        for i in range(1, len(paragraphs) - 1):
            para_words = self._tokenize(paragraphs[i])
            middle_relevance += self._calculate_paragraph_relevance(para_words, keywords)
        
        if len(paragraphs) > 2:
            middle_relevance /= (len(paragraphs) - 2)
        
        # Общая релевантность с учетом весов
        total_relevance = (first_relevance + middle_relevance + last_relevance) / (first_weight + 1 + last_weight)
        
        return min(max(total_relevance, 0.0), 1.0)
    
    def _calculate_paragraph_relevance(self, para_words: List[str], keywords: List[str]) -> float:
        """Расчет релевантности параграфа"""
        if not para_words or not keywords:
            return 0.0
            
        keyword_words = []
        for keyword in keywords:
            keyword_words.extend(self._tokenize(keyword))
        
        matches = 0
        for word in para_words:
            for kw in keyword_words:
                if kw in word:
                    matches += 1
                    break
        
        return matches / len(para_words)
    
    def _extract_related_terms(self, text: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Извлечение связанных терминов"""
        related_terms = {}
        words = self._tokenize(text)
        
        # Создаем окно контекста для каждого ключевого слова
        for keyword in keywords:
            keyword_words = self._tokenize(keyword)
            context_words = set()
            
            for i, word in enumerate(words):
                for kw in keyword_words:
                    if kw in word:
                        context_size = 5
                        start = max(0, i - context_size)
                        end = min(len(words), i + context_size + 1)
                        
                        context_words.update(words[start:i] + words[i+1:end])
            
            # Фильтруем и сортируем по релевантности
            filtered_terms = [w for w in context_words if w not in keyword_words]
            
            # Считаем, сколько раз каждое слово встречается рядом с ключевым
            term_relevance = {}
            for term in filtered_terms:
                term_relevance[term] = text.lower().count(term)
            
            # Сортируем по релевантности
            sorted_terms = sorted(term_relevance.items(), key=lambda x: x[1], reverse=True)
            
            # Берем топ-10
            related_terms[keyword] = [term for term, count in sorted_terms[:10]]
        
        return related_terms
        
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Генерация рекомендаций на основе семантического анализа
        
        Args:
            analysis_results: Результаты семантического анализа
            
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        # Рекомендации по семантической плотности
        semantic_density = analysis_results.get('semantic_density', 0)
        if semantic_density < 0.05:
            recommendations.extend([
                "Увеличьте семантическую плотность контента, добавив больше тематически связанных терминов",
                "Расширьте семантическое поле текста, используя связанные с темой термины и понятия"
            ])
        elif semantic_density > 0.15:
            recommendations.append(
                "Снизьте избыточную семантическую плотность для более естественного звучания текста"
            )
        
        # Рекомендации по семантическому покрытию
        semantic_coverage = analysis_results.get('semantic_coverage', 0)
        if semantic_coverage < 0.5:
            recommendations.extend([
                "Улучшите распределение ключевых терминов по тексту для более равномерного семантического покрытия",
                "Добавьте ключевые слова и связанные термины в разделы, где они отсутствуют"
            ])
        
        # Рекомендации по тематической связности
        topical_coherence = analysis_results.get('topical_coherence', 0)
        if topical_coherence < 0.3:
            recommendations.extend([
                "Повысьте тематическую связность текста, используя больше связанных между собой терминов",
                "Добавьте логические переходы между разделами для усиления тематической связности"
            ])
        
        # Рекомендации по контекстуальной релевантности
        contextual_relevance = analysis_results.get('contextual_relevance', 0)
        if contextual_relevance < 0.3:
            recommendations.extend([
                "Улучшите контекст вокруг ключевых слов, добавив больше релевантной информации",
                "Расширьте семантический контекст, используя термины из той же тематической области"
            ])
        
        # Рекомендации по связанным терминам
        related_terms = analysis_results.get('related_terms', {})
        if not related_terms or all(len(terms) < 3 for terms in related_terms.values()):
            recommendations.append(
                "Расширьте семантическое поле, добавив больше связанных терминов по каждому ключевому слову"
            )
        
        return recommendations
