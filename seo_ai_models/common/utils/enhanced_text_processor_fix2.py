
"""
Дополнительное исправление для EnhancedTextProcessor, добавляющее метод extract_main_topics
"""

from typing import Dict, List, Any
import re
from collections import Counter, defaultdict
from seo_ai_models.common.utils.enhanced_text_processor import EnhancedTextProcessor

def extract_main_topics(self, text: str, num_topics: int = 5) -> List[Dict[str, Any]]:
    """
    Извлекает основные темы из текста.
    
    Args:
        text: Анализируемый текст
        num_topics: Количество возвращаемых тем
        
    Returns:
        List[Dict[str, Any]]: Список основных тем с их весами и контекстом
    """
    if not text:
        return []
    
    # Предобработка: приведение к нижнему регистру, токенизация
    text_lower = text.lower()
    words = self.tokenize(text_lower)
    
    # Удаление стоп-слов (базовые для примера)
    stop_words = set(['и', 'в', 'на', 'с', 'для', 'по', 'к', 'у', 'о', 'из', 
                       'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with'])
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Подсчет частоты слов
    word_freq = Counter(filtered_words)
    
    # Извлечение n-грамм (биграмм и триграмм)
    bigrams = self._extract_ngrams(words, 2)
    trigrams = self._extract_ngrams(words, 3)
    
    # Объединение слов и n-грамм с их частотами
    topics = {}
    
    # Добавляем отдельные слова с высокой частотой
    for word, freq in word_freq.most_common(num_topics * 2):
        topics[word] = {
            'frequency': freq,
            'weight': freq / len(filtered_words) if filtered_words else 0,
            'type': 'word'
        }
    
    # Добавляем биграммы с высокой частотой
    for bigram, freq in bigrams.most_common(num_topics):
        if freq > 1:  # Игнорируем редкие биграммы
            bigram_text = ' '.join(bigram)
            topics[bigram_text] = {
                'frequency': freq,
                'weight': (freq * 1.5) / len(filtered_words) if filtered_words else 0,  # Больший вес для n-грамм
                'type': 'bigram'
            }
    
    # Добавляем триграммы с высокой частотой
    for trigram, freq in trigrams.most_common(num_topics):
        if freq > 1:  # Игнорируем редкие триграммы
            trigram_text = ' '.join(trigram)
            topics[trigram_text] = {
                'frequency': freq,
                'weight': (freq * 2) / len(filtered_words) if filtered_words else 0,  # Еще больший вес для триграмм
                'type': 'trigram'
            }
    
    # Определение контекста для каждой темы
    context = self._get_context_for_topics(text, topics.keys())
    
    # Формирование результата
    result = []
    for topic, metrics in sorted(topics.items(), key=lambda x: x[1]['weight'], reverse=True)[:num_topics]:
        result.append({
            'topic': topic,
            'frequency': metrics['frequency'],
            'weight': metrics['weight'],
            'type': metrics['type'],
            'context': context.get(topic, '')
        })
    
    return result

def _extract_ngrams(self, words: List[str], n: int) -> Counter:
    """
    Извлекает n-граммы из списка слов.
    
    Args:
        words: Список слов
        n: Длина n-граммы
        
    Returns:
        Counter: Счетчик частот n-грамм
    """
    ngrams = Counter()
    
    if len(words) < n:
        return ngrams
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        # Проверяем, что n-грамма не содержит стоп-слов или пунктуации
        if all(len(word) > 2 and word.isalnum() for word in ngram):
            ngrams[ngram] += 1
    
    return ngrams

def _get_context_for_topics(self, text: str, topics: List[str]) -> Dict[str, str]:
    """
    Извлекает контекстное окружение для каждой темы.
    
    Args:
        text: Исходный текст
        topics: Список тем
        
    Returns:
        Dict[str, str]: Словарь топик -> контекст
    """
    result = {}
    sentences = self.split_sentences(text)
    
    for topic in topics:
        topic_lower = topic.lower()
        for sentence in sentences:
            if topic_lower in sentence.lower():
                # Берем предложение, содержащее тему
                result[topic] = sentence.strip()
                break
    
    return result

def analyze_text_structure(self, text: str) -> Dict[str, Any]:
    """
    Анализирует структуру текста.
    
    Args:
        text: Анализируемый текст
        
    Returns:
        Dict[str, Any]: Структурные характеристики текста
    """
    if not text:
        return {
            'paragraphs_count': 0,
            'has_introduction': False,
            'has_conclusion': False,
            'headings_hierarchy': {},
            'lists_count': 0
        }
    
    # Разбиваем текст на абзацы
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    # Анализируем наличие введения
    first_paragraph = paragraphs[0] if paragraphs else ""
    intro_markers = ['введение', 'вступление', 'начало', 'предисловие', 'overview', 'introduction']
    has_intro = any(marker in first_paragraph.lower() for marker in intro_markers)
    
    # Если первый абзац не помечен явно как введение, проверяем его характеристики
    if not has_intro and first_paragraph:
        first_words = self.tokenize(first_paragraph.lower())
        intro_starter_words = ['в', 'this', 'the', 'our', 'we', 'я', 'мы', 'наш']
        if any(word in intro_starter_words for word in first_words[:3]):
            has_intro = True
    
    # Анализируем наличие заключения
    last_paragraph = paragraphs[-1] if paragraphs else ""
    conclusion_markers = ['заключение', 'вывод', 'итог', 'резюме', 'summary', 'conclusion']
    has_conclusion = any(marker in last_paragraph.lower() for marker in conclusion_markers)
    
    # Если последний абзац не помечен явно как заключение, проверяем его характеристики
    if not has_conclusion and last_paragraph:
        conclusion_phrases = ['таким образом', 'в итоге', 'в заключение', 'подводя итог', 
                              'in conclusion', 'to sum up', 'итак', 'therefore']
        if any(phrase in last_paragraph.lower() for phrase in conclusion_phrases):
            has_conclusion = True
    
    # Подсчет списков (маркированных и нумерованных)
    lists_count = 0
    bullet_markers = ['-', '*', '•']
    
    for paragraph in paragraphs:
        lines = paragraph.split('\n')
        if len(lines) > 1:
            # Проверяем на маркированный список
            if any(line.strip().startswith(marker) for line in lines for marker in bullet_markers):
                lists_count += 1
            # Проверяем на нумерованный список
            elif any(re.match(r'^\d+\.', line.strip()) for line in lines):
                lists_count += 1
    
    # Анализ заголовков
    headers = self.extract_headers(text)
    headings_hierarchy = defaultdict(list)
    
    for header in headers:
        level = header.get('level', 0)
        text = header.get('text', '')
        headings_hierarchy[level].append(text)
    
    return {
        'paragraphs_count': len(paragraphs),
        'has_introduction': has_intro,
        'has_conclusion': has_conclusion,
        'headings_hierarchy': dict(headings_hierarchy),
        'lists_count': lists_count
    }

# Добавляем методы к классу
EnhancedTextProcessor.extract_main_topics = extract_main_topics
EnhancedTextProcessor._extract_ngrams = _extract_ngrams
EnhancedTextProcessor._get_context_for_topics = _get_context_for_topics
EnhancedTextProcessor.analyze_text_structure = analyze_text_structure
